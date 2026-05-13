#!/bin/bash
# 一键 SSH 进 dev pod / cloudml / 跑常用诊断
#
# 用法:
#   dev_ssh.sh devpod        # SSH 进世纪互联 dev pod (4321)
#   dev_ssh.sh cloudml       # SSH 进 cloudml dev (4163)
#   dev_ssh.sh status        # 两台一起查 GPU + 进程
#   dev_ssh.sh tail <suite>  # tail 当前 GRPO log (spatial/object/goal/long10)
#   dev_ssh.sh smoke         # dev pod 跑 GRPO smoke (5-10min, 验证镜像)
#   dev_ssh.sh setup         # dev pod 跑 setup.sh (镜像换完后用)
#   dev_ssh.sh push <suite>  # 把 cloudml 上某 suite 训练结果 rsync 到本机
#   dev_ssh.sh fixenv        # dev pod 重新跑 setup.sh + 验证

set -euo pipefail

DEV=4321
CLOUDML=4163
HOST=127.0.0.1
SSH_OPTS="-o StrictHostKeyChecking=no"

cmd="${1:-}"

case "$cmd" in
    devpod|dev)
        # 交互式 SSH 进 dev pod
        exec ssh -p $DEV $SSH_OPTS root@$HOST
        ;;

    cloudml|cm)
        # 交互式 SSH 进 cloudml
        exec ssh -p $CLOUDML $SSH_OPTS root@$HOST
        ;;

    status)
        echo "==========================================="
        echo "  dev pod (世纪互联 4321, 96GB H20)"
        echo "==========================================="
        ssh -p $DEV $SSH_OPTS -o ConnectTimeout=10 root@$HOST '
            hostname; date
            nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
            echo "--- procs ---"
            ps -ef | grep -E "train_grpo|train_dpo|eval_libero|debug_logp" | grep -v grep | head -5 | awk "{print \$2, \$10, \$11, \$12, \$13}"
        ' 2>&1 | tail -20

        echo ""
        echo "==========================================="
        echo "  cloudml (4163, 144GB H20-3e)"
        echo "==========================================="
        ssh -p $CLOUDML $SSH_OPTS -o ConnectTimeout=10 root@$HOST '
            hostname; date
            nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
            echo "--- procs ---"
            ps -ef | grep -E "train_grpo|train_dpo|eval_libero|cloudml_grpo" | grep -v grep | head -5 | awk "{print \$2, \$10, \$11, \$12, \$13}"
        ' 2>&1 | tail -20
        ;;

    tail)
        suite="${2:-spatial}"
        # Default to cloudml since GRPO runs there
        target="${3:-cloudml}"
        if [ "$target" = "cloudml" ]; then
            ssh -p $CLOUDML $SSH_OPTS root@$HOST "
                LOG=/ad-alg/planning-users/liuzhi7/ro_planning/output/cloudml_grpo_$suite/train.log
                if [ -f \$LOG ]; then
                    tail -30 \$LOG | grep -v Warning | tail -20
                else
                    echo 'no log at' \$LOG
                fi
            "
        else
            ssh -p $DEV $SSH_OPTS root@$HOST "
                LOG=/e2e-data/users/liuzhi7/vla_workspace/output/h20_grpo_$suite/train.log
                if [ -f \$LOG ]; then
                    tail -30 \$LOG | grep -v Warning | tail -20
                else
                    echo 'no log at' \$LOG
                fi
            "
        fi
        ;;

    smoke)
        ssh -p $DEV $SSH_OPTS -t root@$HOST 'bash /e2e-data/users/liuzhi7/.persist/grpo_smoke.sh'
        ;;

    setup|fixenv)
        ssh -p $DEV $SSH_OPTS -t root@$HOST 'bash /e2e-data/users/liuzhi7/.persist/setup.sh'
        ;;

    push)
        suite="${2:-spatial}"
        SRC="root@$HOST:/ad-alg/planning-users/liuzhi7/ro_planning/output/cloudml_grpo_${suite}_eval/"
        DST="$HOME/ro_planning/assets/paper_v1.5_eval/cloudml_grpo_${suite}_eval/"
        mkdir -p "$DST"
        rsync -az --info=progress2 -e "ssh -p $CLOUDML $SSH_OPTS" "$SRC" "$DST"
        echo "→ $DST"
        ;;

    log)
        # Tail any log on dev pod
        path="${2:-/e2e-data/users/liuzhi7/vla_workspace/output/devpod_grpo_smoke/smoke.log}"
        ssh -p $DEV $SSH_OPTS root@$HOST "tail -f $path"
        ;;

    *)
        cat << EOF
Usage:
  $0 devpod | dev          # SSH into dev pod (世纪互联 4321, 96GB H20)
  $0 cloudml | cm          # SSH into cloudml (4163, 144GB H20-3e)
  $0 status                # GPU + procs on both
  $0 tail <suite> [target] # tail GRPO train log (target = cloudml | dev)
  $0 smoke                 # run GRPO smoke on dev pod (validates image)
  $0 setup | fixenv        # run setup.sh on dev pod (after image change)
  $0 push <suite>          # rsync cloudml GRPO eval results back to local
  $0 log [path]            # tail -f arbitrary log on dev pod

Examples:
  $0 status
  $0 tail spatial
  $0 tail spatial cloudml
  $0 push goal
  $0 log /e2e-data/users/liuzhi7/vla_workspace/output/devpod_grpo_smoke/smoke.log
EOF
        ;;
esac
