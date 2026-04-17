#!/usr/bin/env python3
"""
manage_call_log.py — CLI tool for the canvasxpress-mcp call log database.

Usage:
    python manage_call_log.py stats
    python manage_call_log.py export [--tool TOOL] [--rated-only] [--limit N] [--format json|csv] [--out FILE]
    python manage_call_log.py purge  [--tool TOOL] [--rated-only] [--yes]

Commands:
    stats        Print row counts by tool and rating.
    export       Export rows to JSON or CSV (stdout or --out FILE).
    purge        Delete rows (prompts for confirmation unless --yes is given).

Options:
    --tool TOOL      Filter by tool name (partial match supported).
    --rated-only     Only include/delete rows that have a rating (1 or -1).
    --limit N        Max rows to export (default: all rows).
    --format FMT     Output format: json (default) or csv.
    --out FILE       Write output to FILE instead of stdout.
    --yes            Skip confirmation prompt when purging.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the database
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "data" / "call_log.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        print("The server must have been started at least once to create the call log.", file=sys.stderr)
        sys.exit(1)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cmd_stats(args, db_path: Path) -> None:
    con = _connect(db_path)

    total = con.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    rated = con.execute("SELECT COUNT(*) FROM tool_calls WHERE rating IS NOT NULL").fetchone()[0]
    thumbs_up   = con.execute("SELECT COUNT(*) FROM tool_calls WHERE rating = 1").fetchone()[0]
    thumbs_down = con.execute("SELECT COUNT(*) FROM tool_calls WHERE rating = -1").fetchone()[0]

    print(f"\nCall log: {db_path}")
    print(f"{'─' * 50}")
    print(f"  Total calls   : {total:,}")
    print(f"  Rated         : {rated:,}  (unrated: {total - rated:,})")
    print(f"  Thumbs up  👍 : {thumbs_up:,}")
    print(f"  Thumbs down 👎: {thumbs_down:,}")

    rows = con.execute(
        """
        SELECT tool,
               COUNT(*)                                      AS total,
               SUM(CASE WHEN rating IS NOT NULL THEN 1 END) AS rated,
               SUM(CASE WHEN rating = 1         THEN 1 END) AS up,
               SUM(CASE WHEN rating = -1        THEN 1 END) AS down,
               MIN(ts)                                       AS first_call,
               MAX(ts)                                       AS last_call
        FROM tool_calls
        GROUP BY tool
        ORDER BY total DESC
        """
    ).fetchall()

    if rows:
        print(f"\n{'Tool':<40} {'Total':>6} {'Rated':>6} {'👍':>5} {'👎':>5}  First call")
        print(f"{'─' * 40} {'─' * 6} {'─' * 6} {'─' * 5} {'─' * 5}  {'─' * 24}")
        for r in rows:
            first = (r["first_call"] or "")[:19].replace("T", " ")
            print(
                f"{(r['tool'] or '?'):<40} {r['total']:>6,} "
                f"{(r['rated'] or 0):>6,} {(r['up'] or 0):>5,} {(r['down'] or 0):>5,}  {first}"
            )
    print()
    con.close()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def cmd_export(args, db_path: Path) -> None:
    con = _connect(db_path)

    clauses: list[str] = []
    params:  list      = []

    if args.tool:
        clauses.append("tool LIKE ?")
        params.append(f"%{args.tool}%")
    if args.rated_only:
        clauses.append("rating IS NOT NULL")

    where  = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    limit  = f"LIMIT {int(args.limit)}" if args.limit else ""
    query  = f"""
        SELECT id, tool, path, request, response, status, ts, rating, comment
        FROM tool_calls {where} ORDER BY ts DESC {limit}
    """

    rows = con.execute(query, params).fetchall()
    con.close()

    # Deserialise stored JSON strings
    records = []
    for row in rows:
        d = dict(row)
        for field in ("request", "response"):
            try:
                d[field] = json.loads(d[field]) if d[field] else None
            except Exception:
                pass
        records.append(d)

    fmt = (args.format or "json").lower()
    out_path = args.out

    if fmt == "csv":
        _write_csv(records, out_path)
    else:
        _write_json(records, out_path)

    dest = out_path or "stdout"
    count = len(records)
    print(f"Exported {count:,} row(s) → {dest} [{fmt}]", file=sys.stderr)


def _write_json(records: list[dict], out_path: str | None) -> None:
    text = json.dumps(records, indent=2, default=str)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def _write_csv(records: list[dict], out_path: str | None) -> None:
    if not records:
        if out_path:
            Path(out_path).write_text("", encoding="utf-8")
        return

    # Flatten nested request/response to strings for CSV
    flat = []
    for r in records:
        row = dict(r)
        for field in ("request", "response"):
            if isinstance(row.get(field), (dict, list)):
                row[field] = json.dumps(row[field])
        flat.append(row)

    fieldnames = list(flat[0].keys())
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------
def cmd_purge(args, db_path: Path) -> None:
    con = _connect(db_path)

    clauses: list[str] = []
    params:  list      = []

    if args.tool:
        clauses.append("tool LIKE ?")
        params.append(f"%{args.tool}%")
    if args.rated_only:
        clauses.append("rating IS NOT NULL")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    # Count first
    count = con.execute(f"SELECT COUNT(*) FROM tool_calls {where}", params).fetchone()[0]

    if count == 0:
        print("No rows match the given filters — nothing to purge.")
        con.close()
        return

    scope = "ALL" if not clauses else " + ".join(
        ([f"tool LIKE '%{args.tool}%'"] if args.tool else []) +
        (["rated only"] if args.rated_only else [])
    )
    print(f"\n⚠️  About to permanently delete {count:,} row(s) from {db_path}")
    print(f"   Scope : {scope}")

    if not args.yes:
        try:
            answer = input("   Type 'yes' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            con.close()
            return
        if answer != "yes":
            print("Aborted.")
            con.close()
            return

    cur = con.execute(f"DELETE FROM tool_calls {where}", params)
    con.commit()
    deleted = cur.rowcount
    con.close()
    print(f"Deleted {deleted:,} row(s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the canvasxpress-mcp call log database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db", default=str(DB_PATH),
        help=f"Path to call_log.db (default: {DB_PATH})"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- stats ---
    sub.add_parser("stats", help="Print row counts by tool and rating.")

    # --- export ---
    exp = sub.add_parser("export", help="Export rows to JSON or CSV.")
    exp.add_argument("--tool",       help="Filter: tool name (partial match).")
    exp.add_argument("--rated-only", action="store_true", help="Only rows with a rating.")
    exp.add_argument("--limit",      type=int, default=None, help="Max rows (default: all).")
    exp.add_argument("--format",     choices=["json", "csv"], default="json", help="Output format (default: json).")
    exp.add_argument("--out",        help="Write to FILE instead of stdout.")

    # --- purge ---
    prg = sub.add_parser("purge", help="Delete rows (with confirmation).")
    prg.add_argument("--tool",       help="Filter: tool name (partial match).")
    prg.add_argument("--rated-only", action="store_true", help="Only rows with a rating.")
    prg.add_argument("--yes",        action="store_true", help="Skip confirmation prompt.")

    args = parser.parse_args()
    db_path = Path(args.db)

    if args.command == "stats":
        cmd_stats(args, db_path)
    elif args.command == "export":
        cmd_export(args, db_path)
    elif args.command == "purge":
        cmd_purge(args, db_path)


if __name__ == "__main__":
    main()
