#!/usr/bin/env bash
# Usage: ./server.sh start | stop | restart | status
PIDFILE="$(dirname "$0")/server.pid"
LOGFILE="$(dirname "$0")/server.log"
CMD="/Users/isaac/git/.venv/bin/python src/server.py"

start() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running (pid $(cat "$PIDFILE"))"
    exit 1
  fi
  nohup $CMD >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  echo "Started (pid $!)"
}

stop() {
  if [ ! -f "$PIDFILE" ]; then
    echo "Not running (no pidfile)"
    exit 1
  fi
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" && rm -f "$PIDFILE"
    echo "Stopped (pid $PID)"
  else
    rm -f "$PIDFILE"
  fi
  # Kill any orphaned process still holding port 8100
  PORT_PID=$(lsof -ti:8100 2>/dev/null)
  if [ -n "$PORT_PID" ]; then
    kill "$PORT_PID" 2>/dev/null
    echo "Killed orphaned process on port 8100 (pid $PORT_PID)"
  fi
}

_stop_quiet() {
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"
      echo "Stopped (pid $PID)"
    fi
    rm -f "$PIDFILE"
  fi
  # Kill any orphaned process still holding port 8100
  PORT_PID=$(lsof -ti:8100 2>/dev/null)
  if [ -n "$PORT_PID" ]; then
    kill "$PORT_PID" 2>/dev/null
    echo "Killed orphaned process on port 8100 (pid $PORT_PID)"
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
  restart) _stop_quiet; sleep 1; start ;;
  status)  status ;;
  *)       echo "Usage: $0 start|stop|restart|status"; exit 1 ;;
esac
