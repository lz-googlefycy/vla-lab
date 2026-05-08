#!/bin/bash
# ============================================================
# bootstrap-ssh.sh — one-shot setup for reverse-SSH tunnel to desktop.
#
# Rationale: the k8s pod loses all state between rebuilds.  SSH private
# keys should NOT live in the docker image (it is pushed to 3 public-ish
# registries). Instead they live on JuiceFS (/ad-alg/.../ssh_backup/)
# which is persistent across pod rebuilds.  This script:
#
#   1. Copies the saved SSH assets from JuiceFS back to /root/.ssh
#   2. Fixes permissions
#   3. Installs packages if somehow missing (autossh, tmux, openssh-server)
#   4. Starts the reverse tunnel via tmux+autossh to your desktop
#
# Run ONCE after each pod rebuild:
#   bash /opt/vla-lab/bootstrap-ssh.sh
# Idempotent — safe to run multiple times.
# ============================================================
set -euo pipefail

BACKUP_DIR="${SSH_BACKUP_DIR:-/ad-alg/planning-users/liuzhi7/.ssh_backup}"
SSH_DIR="${SSH_DIR:-/root/.ssh}"

log()  { echo "[bootstrap-ssh] $*"; }
die()  { echo "[bootstrap-ssh] ERROR: $*" >&2; exit 1; }

# ---- 1. Check JuiceFS backup is present ----
if [ ! -d "$BACKUP_DIR" ]; then
    die "SSH backup dir $BACKUP_DIR not found. First-time setup required: copy keys there manually."
fi
if [ ! -f "$BACKUP_DIR/authorized_keys" ] || [ ! -f "$BACKUP_DIR/autossh_volc_to_desktop" ]; then
    die "Required keys missing in $BACKUP_DIR (need authorized_keys + autossh_volc_to_desktop)."
fi

# ---- 2. Restore SSH assets ----
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

for f in authorized_keys autossh_volc_to_desktop autossh_volc_to_desktop.pub \
         id_rsa id_rsa.pub known_hosts; do
    if [ -f "$BACKUP_DIR/$f" ] && [ ! -f "$SSH_DIR/$f" ]; then
        cp "$BACKUP_DIR/$f" "$SSH_DIR/$f"
        log "restored $f"
    fi
done

# Fix perms
chmod 600 "$SSH_DIR/authorized_keys" 2>/dev/null || true
chmod 600 "$SSH_DIR/autossh_volc_to_desktop" 2>/dev/null || true
chmod 600 "$SSH_DIR/id_rsa" 2>/dev/null || true
chmod 644 "$SSH_DIR"/*.pub 2>/dev/null || true
chmod 644 "$SSH_DIR/known_hosts" 2>/dev/null || true

# ---- 3. Ensure packages exist ----
for pkg in autossh tmux openssh-server; do
    if ! dpkg -l | grep -q "^ii.*$pkg\s"; then
        log "installing $pkg ..."
        apt-get update -qq
        apt-get install -y --no-install-recommends "$pkg"
    fi
done

# ---- 4. Start sshd on 2222 (bound to localhost for tunnel) ----
# If the ML platform already starts sshd via /public-tools/start-sshd.sh,
# that usually listens on :22 (via container port mapping). We additionally
# run a localhost-only :2222 for the reverse tunnel.
if ! pgrep -f "sshd.*-p 2222" >/dev/null; then
    log "starting sshd on localhost:2222"
    /usr/sbin/sshd -p 2222 \
        -o ListenAddress=127.0.0.1 \
        -o PermitRootLogin=yes \
        -o PubkeyAuthentication=yes \
        -o PasswordAuthentication=no \
        -o AuthorizedKeysFile="$SSH_DIR/authorized_keys"
else
    log "sshd :2222 already running"
fi

# ---- 5. Start autossh reverse tunnel ----
TUNNEL_SCRIPT="$BACKUP_DIR/autossh_tunnel_cmd.sh"
if [ -f "$TUNNEL_SCRIPT" ]; then
    if tmux has-session -t autossh-4163 2>/dev/null; then
        log "autossh tunnel already running (tmux session 'autossh-4163')"
    else
        log "starting autossh reverse tunnel"
        bash "$TUNNEL_SCRIPT" || die "autossh tunnel failed to start"
    fi
else
    log "warning: no autossh_tunnel_cmd.sh in backup dir — skipping tunnel"
    log "         (you can create it manually, see docs/dev_machine_bootstrap.md)"
fi

# ---- 6. Summary ----
log ""
log "===== BOOTSTRAP COMPLETE ====="
log "SSH assets restored to $SSH_DIR"
log "sshd listening on localhost:2222"
log "reverse tunnel: ubuntu@desktop:4163 -> this_pod:2222"
log ""
log "Next: from your Ubuntu Desktop, test with:"
log "      ssh -p 4163 root@127.0.0.1"
log ""
log "To inspect the autossh tunnel:"
log "      tmux attach -t autossh-4163   (Ctrl+B then D to detach)"
