#!/usr/bin/env bash
# 后台启动 PaperStudio，PID 写入 server.pid，日志追加到 server.log
set -euo pipefail

PID_FILE="server.pid"
LOG_FILE="server.log"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "服务已在运行中 (pid=$OLD_PID)，请先执行 ./stop.sh"
        exit 1
    else
        echo "清理失效的 PID 文件 (pid=$OLD_PID)"
        rm -f "$PID_FILE"
    fi
fi

nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "PaperStudio 已启动 (pid=$(cat $PID_FILE))，日志: $LOG_FILE"
