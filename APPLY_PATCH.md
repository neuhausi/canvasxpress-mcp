# canvasxpress-ctypes-fix — Apply Instructions

## What this patch does

Fixes two files (`build_index.py` and `src/server.py`) to work on servers where
Python was compiled **without** `--enable-loadable-sqlite-extensions`.

Instead of calling `db.enable_load_extension()` directly (which raises `AttributeError`
on restricted Python builds), the patch uses a `ctypes` wrapper that calls the SQLite C
API directly via `/usr/lib64/libsqlite3.so.0`.

The sqlite3* handle offset (16 bytes) was confirmed for CPython 3.12 on this server via
a diagnostic pointer scan.

---

## Files changed

- `build_index.py` — adds `ExtensionConnection` wrapper class and `connect()` helper
- `src/server.py` — adds `ctypes` imports, `_sqlite_connect()` helper, patches `_vector_retrieve()`

---

## How to apply after a git pull

```bash
cd /home/canvasxpress/canvasxpress-mcp

# 1. Stash any local changes
git stash

# 2. Pull latest from upstream
git pull

# 3. Apply the patch
git apply --whitespace=nowarn canvasxpress-ctypes-fix.patch

# 4. Verify it applied correctly
grep -n "_sqlite_connect\|ctypes" src/server.py | head -5

# 5. Rebuild the index
python build_index.py

# 6. Restart the server
python src/server.py
```

---

## If the patch fails after a pull (upstream changed same lines)

```bash
git apply --whitespace=nowarn --reject canvasxpress-ctypes-fix.patch
```

This applies what it can and writes `*.rej` files for any conflicts.
Inspect the `.rej` files and manually merge the ctypes changes into the affected functions.

The key changes to re-apply manually if needed:

### src/server.py — add after `import sqlite3`
```python
import ctypes
import ctypes.util
```

### src/server.py — add before `def _vector_retrieve(`
```python
def _sqlite_connect(db_path: str):
    """Connect to sqlite3 with extension loading, using ctypes fallback if needed."""
    try:
        db = sqlite3.connect(db_path)
        db.enable_load_extension(True)
        return db
    except AttributeError:
        pass
    lib = ctypes.CDLL('/usr/lib64/libsqlite3.so.0')
    lib.sqlite3_db_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.sqlite3_db_filename.restype = ctypes.c_char_p
    lib.sqlite3_enable_load_extension.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.sqlite3_enable_load_extension.restype = ctypes.c_int

    class _ExtConn:
        def __init__(self):
            self._db = sqlite3.connect(db_path)
            ptr = ctypes.c_void_p.from_address(id(self._db) + 16).value
            lib.sqlite3_enable_load_extension(ptr, 1)
            self._ptr = ptr
        def load_extension(self, path):
            lib.sqlite3_load_extension.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
            lib.sqlite3_load_extension.restype = ctypes.c_int
            errmsg = ctypes.c_char_p()
            rc = lib.sqlite3_load_extension(self._ptr, path.encode() if isinstance(path, str) else path, None, ctypes.byref(errmsg))
            if rc != 0:
                raise RuntimeError(f"load_extension failed: {errmsg.value}")
        def enable_load_extension(self, val): pass
        def __getattr__(self, name): return getattr(self._db, name)

    return _ExtConn()
```

### src/server.py — replace in `_vector_retrieve()`
```python
# BEFORE:
db = sqlite3.connect(str(DB_FILE))
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# AFTER:
db = _sqlite_connect(str(DB_FILE))
sqlite_vec.load(db)
try:
    db.enable_load_extension(False)
except AttributeError:
    pass
```

### build_index.py — see full ExtensionConnection class in the patched file

---

## Re-saving the patch after changes

After applying and confirming everything works, refresh the patch file:

```bash
cd /home/canvasxpress/canvasxpress-mcp
git diff > canvasxpress-ctypes-fix.patch
```

---

## Notes

- The offset `16` is specific to CPython 3.12 on this server. If Python is upgraded or
  recompiled, run the diagnostic again to confirm:
  ```bash
  python3 -c "
  import sqlite3, ctypes
  lib = ctypes.CDLL('/usr/lib64/libsqlite3.so.0')
  lib.sqlite3_db_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
  lib.sqlite3_db_filename.restype = ctypes.c_char_p
  db = sqlite3.connect('/tmp/test.db')
  base = id(db)
  for offset in range(0, 128, 8):
      try:
          candidate = ctypes.c_void_p.from_address(base + offset).value
          if candidate and candidate > 0x10000:
              result = lib.sqlite3_db_filename(candidate, b'main')
              if result: print(f'offset={offset} -> {result}'); break
      except: pass
  "
  ```
- The ideal long-term fix is to recompile Python with `--enable-loadable-sqlite-extensions`
  or ask the server admin to install a Python build that includes it.
