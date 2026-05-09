# 开发机 Pod 重建后快速恢复 SSH 指南

> 每次 ML 平台重建 pod 后，pod 内所有 `/` 根分区内容会丢失（包括 /root/.ssh）。
> 但 JuiceFS 挂载的 `/workspace/jfs/...` 是**跨 pod 持久化的**。
> 这份文档教你**如何一行命令恢复 SSH 反向隧道 + Spirit 环境**。

---

## 持久化 SSH 资产位置

所有 SSH 资产保存在 JuiceFS：

```
/workspace/jfs/.ssh_backup/
├── authorized_keys              # 你 desktop 能 ssh 进 pod 的公钥
├── id_rsa / id_rsa.pub          # pod 作为 client 用的 keypair
├── autossh_volc_to_desktop      # autossh 反向隧道 private key
├── autossh_volc_to_desktop.pub  # 对应 public key
├── known_hosts                  # desktop 的 fingerprint
├── autossh_tunnel_cmd.sh        # 反向隧道启动命令
└── bootstrap-ssh.sh             # 一键恢复脚本（本文的主角）
```

---

## 每次 pod 重建后的完整流程

### 1. 在 ML 平台 UI 创建新 pod

选镜像：

```
<private-registry>/planningmodel:spirit-v1.0-cu128-py310
```

或 sim 扩展（含 LIBERO + Maniskill）：

```
<private-registry>/planningmodel:spirit-sim-v1.0-cu128-py310
```

挂载 JuiceFS 路径 `/workspace/jfs/` 到 pod 内同样的位置（平台 UI 通常默认挂载）。

### 2. 进入 pod 后（通过 ML 平台的"在线 IDE"/ 相应工具），一行执行：

```bash
bash /workspace/jfs/.ssh_backup/bootstrap-ssh.sh
```

这个脚本会：
- 复制 SSH key / config 回 `/root/.ssh/` 并修好权限
- 检查 autossh/tmux/openssh-server 是否装了，没装就装
- 启动 sshd 监听 `localhost:2222`
- 启动 tmux+autossh 反向隧道到 `ubuntu@10.189.148.41:4163`

脚本是**幂等的**——可以重复跑，不会冲突。

### 3. 从你的 Ubuntu Desktop 连接：

```bash
ssh -p 4163 root@127.0.0.1
```

即可回到老熟悉的 4163 端口。

---

## 脚本内部做了什么

[`bootstrap-ssh.sh`](/workspace/jfs/.ssh_backup/bootstrap-ssh.sh)：

```bash
# 从 JuiceFS 备份恢复 SSH 资产
cp /workspace/jfs/.../.ssh_backup/{authorized_keys, id_rsa, autossh_volc_to_desktop, ...} \
   /root/.ssh/
chmod 600 /root/.ssh/{authorized_keys, id_rsa, autossh_volc_to_desktop}

# 确保包装上（镜像里已预装，双重保险）
apt install -y autossh tmux openssh-server

# 启动 localhost:2222 的 sshd（用于接收 desktop 反向隧道）
/usr/sbin/sshd -p 2222 -o ListenAddress=127.0.0.1 \
    -o PermitRootLogin=yes -o PubkeyAuthentication=yes \
    -o AuthorizedKeysFile=/root/.ssh/authorized_keys

# 启动 tmux + autossh 反向隧道
tmux new-session -d -s autossh-4163 \
    "autossh -M 0 -N -i /root/.ssh/autossh_volc_to_desktop \
     -R 4163:127.0.0.1:2222 ubuntu@10.189.148.41"
```

---

## 更新 SSH 资产（如果以后换 key）

直接更新 JuiceFS 备份：

```bash
# 在现有 pod 里（旧 key 还在）
cp /root/.ssh/authorized_keys /workspace/jfs/.ssh_backup/

# 下次 pod 重建时就会用新 key
```

---

## 镜像里为什么不打包 SSH private key

**安全原因**：`spirit-v1.0-cu128-py310` 镜像被 push 到 3 个公司内部 registry
(micr / volc / evad)。任何能 `docker pull` 这个镜像的人都能拿到镜像里的文件。
如果把 private key 打进镜像，等于**把隧道建立权限公开给所有能 pull 的人**。

因此约定：
- **镜像打包**：apt 工具（autossh, tmux, openssh-server）+ bootstrap 脚本
- **JuiceFS 放**：所有 `/root/.ssh/*` 敏感文件
- **启动时流程**：镜像提供工具 → JuiceFS 提供凭据 → bootstrap 脚本粘合

---

## 常见问题

### Q: `bash bootstrap-ssh.sh` 报 "autossh_tunnel_cmd.sh not found"

A: JuiceFS 备份里缺 `autossh_tunnel_cmd.sh`。手动创建，或跑下面这行：

```bash
cat > /workspace/jfs/.ssh_backup/autossh_tunnel_cmd.sh <<'EOF'
#!/bin/bash
tmux new-session -d -s autossh-4163 \
    "AUTOSSH_GATETIME=0 autossh -M 0 -N \
     -i /root/.ssh/autossh_volc_to_desktop \
     -o StrictHostKeyChecking=accept-new \
     -o ExitOnForwardFailure=yes \
     -o ServerAliveInterval=30 \
     -o ServerAliveCountMax=3 \
     -o TCPKeepAlive=yes \
     -R 4163:127.0.0.1:2222 \
     ubuntu@10.189.148.41 > /tmp/autossh-4163.log 2>&1"
EOF
chmod +x /workspace/jfs/.ssh_backup/autossh_tunnel_cmd.sh
```

### Q: 隧道启动了但 desktop 连不上

A: 看 `tmux attach -t autossh-4163` 里的错误输出，或 `tail /tmp/autossh-4163.log`。
常见原因：desktop 的 IP 变了（不再是 10.189.148.41），需要编辑
`autossh_tunnel_cmd.sh` 里的 `DESKTOP_HOST`。

### Q: desktop 上哪儿看到 pod 新端口的变化？

A: 如果你重建 pod 时选择了不同的反向端口（比如 4164），用
`DESKTOP_HOST=... FORWARD_PORT=4164 bash bootstrap-ssh.sh` 启动。

### Q: pod 里 sshd 已经在 :22 上了，为什么还要 :2222？

A: ML 平台的 start-sshd.sh 已经在 :22 启动了 sshd（让平台的 "在线 SSH" 功能用）。
我们的 :2222 是**独立的另一个 sshd 实例**，专门给 autossh 反向隧道用。
这样平台的 ssh 和我们的 autossh 互不影响。

---

## 相关文档

- [`docs/env_setup.md`](./env_setup.md) — 完整开发机 + 本机环境说明
- [`docs/troubleshooting.md`](./troubleshooting.md) — 踩坑记录
- [`docker-spirit/Dockerfile`](../docker-spirit/Dockerfile) — 镜像定义
- [`docker-spirit/bootstrap-ssh.sh`](../docker-spirit/bootstrap-ssh.sh) — 脚本源码
