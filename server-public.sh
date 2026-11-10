#!/usr/bin/env bash
# Second MCP instance: the AUTHENTICATED, public profile for ChatGPT Apps /
# Gemini Connected Apps. Runs the SAME code as server.sh, but on its own port
# (8101) with CX_PUBLIC=1, so it NEVER touches the open port-8100 instance that
# serves the engine's built-in LLM assist (/generate, /modify, ...). Those stay
# open and unauthenticated; only this instance requires a bearer token.
#
# The bearer token lives in .env.public (gitignored), not in this script. The
# public-profile overrides below beat the shared .env because server.py calls
# load_dotenv(override=False) — existing env wins.
#
# Usage: ./server-public.sh start | stop | restart | status
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$SCRIPT_DIR/server-public.pid"
LOGFILE="$SCRIPT_DIR/server-public.log"
PORT_NUM=8101

if   [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then PYTHON="$SCRIPT_DIR/../.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ];    then PYTHON="$SCRIPT_DIR/.venv/bin/python"
else PYTHON="$(command -v python3 || command -v python)"; fi

# Public-profile overrides (win over the shared .env).
export PORT="$PORT_NUM"
export CX_PUBLIC=1
export CX_DAILY_COST_LIMIT_USD="${CX_DAILY_COST_LIMIT_USD:-5}"
export CX_RATE_LIMIT_PER_MIN="${CX_RATE_LIMIT_PER_MIN:-60}"
# Token (required — CX_PUBLIC is fail-closed) and any extra overrides.
if [ -f "$SCRIPT_DIR/.env.public" ]; then set -a; . "$SCRIPT_DIR/.env.public"; set +a; fi

start() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running (pid $(cat "$PIDFILE"))"; exit 1
  fi
  cd "$SCRIPT_DIR"
  nohup $PYTHON "$SCRIPT_DIR/src/server.py" >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  echo "Started public instance on :$PORT_NUM (pid $!) — check $LOGFILE for the 'Auth: bearer REQUIRED' line"
}

stop() {
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then kill "$PID" && echo "Stopped (pid $PID)"; fi
    rm -f "$PIDFILE"
  else
    echo "Not running (no pidfile)"
  fi
  PORT_PID=$(lsof -ti:"$PORT_NUM" 2>/dev/null)
  if [ -n "$PORT_PID" ]; then
    kill "$PORT_PID" 2>/dev/null && echo "Killed orphan on :$PORT_NUM (pid $PORT_PID)"
  fi
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Running (pid $(cat "$PIDFILE"))"
  else
    echo "Not running"
  fi
}

case "$1" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  *)       echo "Usage: $0 start|stop|restart|status"; exit 1 ;;
esac
