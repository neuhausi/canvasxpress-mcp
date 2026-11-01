# canvasxpress-upload: a local MCP server that uploads a file behind the scenes

A chat client never hands a file to a remote MCP server, and the one way a model can move a file without
typing its rows (a `curl` from its sandbox) is blocked by some clients as "outbound data transfer". This local
server runs on the user's machine, outside any sandbox: its one tool `upload_csv(path)` reads the file from disk,
POSTs it to `https://canvasxpress.org/mcp/upload` and returns a link. The model then calls
`suggest_canvasxpress_dashboard(url=<link>)` on the remote server. The rows never pass through the model.

```
you:    Create a dashboard from ~/Downloads/trial-patients.csv
model:  upload_csv(path="~/Downloads/trial-patients.csv")         -> {url: https://canvasxpress.org/mcp/file/…}
model:  suggest_canvasxpress_dashboard(url="https://…/mcp/file/…") -> dashboard, 0.4 s
```

Standard library only; no install. Needs only a `python3`.

## Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config) and restart Claude Desktop:

```json
"canvasxpress-upload": {
  "command": "python3",
  "args": ["/ABSOLUTE/PATH/TO/canvasxpress-mcp/tools/local-upload/cx_upload.py"]
}
```

Or pack this folder as a Desktop Extension (`npx @anthropic-ai/mcpb pack tools/local-upload`) and double-click the
`.mcpb`. Any other client that runs local stdio servers (Cursor, VS Code, Claude Code: `claude mcp add
canvasxpress-upload -- python3 .../cx_upload.py`) works the same way.

`CX_UPLOAD_URL` points the tool at another CanvasXpress MCP server (`http://host:8100/mcp/upload`).

**Consent, every time, no standing privileges.** Nothing is sent without the person saying so: each `upload_csv`
call first shows a native dialog over the chat ("Claude asks to send *file* (N KB) to *host* … Allow / Don't allow");
a refusal is returned to the model as an error that tells it not to retry and not to type the rows. `CX_UPLOAD_ASK=0`
turns the dialog off for unattended use. On platforms without the dialog (Linux, Windows) the upload proceeds.

**macOS protected folders (Downloads, Desktop, Documents, iCloud).** The process Claude Desktop starts cannot read
them on its own. Instead of asking for a Files-and-Folders grant in System Settings, the tool opens the standard
macOS file chooser pointed at the file; choosing it there lets this one read through (macOS counts a file the person
picked as consent for that file only). Cancelling returns an error to the model. A file anywhere else (for example
`~/cx-data/`) is read without the chooser, after the consent dialog.

## Test

```
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"upload_csv","arguments":{"path":"~/Downloads/x.csv"}}}' | python3 cx_upload.py
```
