#!/usr/bin/env bash
# 停止后台运行的 PaperStudio
set -euo pipefail

PID_FILE="server.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "未找到 $PID_FILE，服务可能未启动"
    exit 1
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "已发送 SIGTERM 给进程 (pid=$PID)"
    # 等待最多 5 秒让进程退出
    for i in {1..10}; do
        sleep 0.5
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "进程已退出"
            rm -f "$PID_FILE"
            exit 0
        fi
    done
    echo "进程未在 5 秒内退出，发送 SIGKILL..."
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "已强制终止 (pid=$PID)"
else
    echo "进程 (pid=$PID) 已不存在，清理 PID 文件"
    rm -f "$PID_FILE"
fi
