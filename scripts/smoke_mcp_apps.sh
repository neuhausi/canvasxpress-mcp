#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MCP Apps wire-format smoke test for the CanvasXpress MCP server.
#
# Verifies (against a running server on $MCP_URL):
#   1. initialize handshake works
#   2. resources/list contains ui://canvasxpress/chart with the correct
#      MIME type and _meta.ui.csp.resourceDomains
#   3. resources/read returns the HTML body with _meta.ui.csp on the content
#      item AND at the result level
#   4. tools/list has _meta.ui.resourceUri on all 4 chart tools and
#      NOT on list_chart_types
#
# Usage:
#   ./scripts/smoke_mcp_apps.sh
#   MCP_URL=http://example:8100/mcp ./scripts/smoke_mcp_apps.sh
#
# Requires: curl, jq.
# Exit code: 0 on full pass, non-zero on first failure.
# ---------------------------------------------------------------------------

set -u
set -o pipefail

MCP_URL="${MCP_URL:-http://localhost:8100/mcp}"
PROTO_VERSION="2026-01-26"
RESOURCE_URI="ui://canvasxpress/chart"
EXPECTED_MIME="text/html;profile=mcp-app"
EXPECTED_CDN="https://www.canvasxpress.org"
CHART_TOOLS=(
    generate_canvasxpress_config
    modify_canvasxpress_config
    generate_km_config
    create_map_config
)
NON_CHART_TOOL="list_chart_types"

PASS=0
FAIL=0
FAILED_CHECKS=()

c_green=$'\033[32m'
c_red=$'\033[31m'
c_dim=$'\033[2m'
c_reset=$'\033[0m'

ok()   { PASS=$((PASS+1)); printf "  %s✓%s %s\n" "$c_green" "$c_reset" "$1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); printf "  %s✗%s %s\n" "$c_red" "$c_reset" "$1"; [ -n "${2:-}" ] && printf "    %s%s%s\n" "$c_dim" "$2" "$c_reset"; }

# Streamable-HTTP responses may be either application/json or text/event-stream.
# For SSE we extract the JSON payload from the first "data:" line.
# Argument 1: full HTTP body. Stdout: a single JSON object.
extract_json() {
    local body="$1"
    if printf '%s' "$body" | head -c 1 | grep -q '{'; then
        printf '%s' "$body"
    else
        # SSE: pick first data: line and strip the prefix
        printf '%s' "$body" | awk '/^data: /{sub(/^data: /,""); print; exit}'
    fi
}

# POST a JSON-RPC request. Echoes the parsed JSON body to stdout. Response
# headers are written to $HDR_FILE so the caller (in the parent shell) can
# inspect them after the command substitution returns.
HDR_FILE="$(mktemp)"
trap 'rm -f "$HDR_FILE"' EXIT
post_rpc() {
    local payload="$1"
    local extra_headers="${2:-}"
    local body
    body=$(curl -sS -X POST "$MCP_URL" \
        -D "$HDR_FILE" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        ${extra_headers:+-H "$extra_headers"} \
        --data "$payload")
    local rc=$?
    [ $rc -ne 0 ] && return $rc
    extract_json "$body"
}

# ---------------------------------------------------------------------------
echo
echo "MCP Apps smoke test → $MCP_URL"
echo

# --- 1. initialize ---------------------------------------------------------
echo "[1/4] initialize handshake"
init_payload=$(jq -n --arg v "$PROTO_VERSION" '{
  jsonrpc:"2.0", id:1, method:"initialize",
  params:{ protocolVersion:$v, capabilities:{}, clientInfo:{name:"smoke",version:"0"} }
}')
init_resp=$(post_rpc "$init_payload") || { fail "initialize POST failed"; exit 1; }

if echo "$init_resp" | jq -e '.result.protocolVersion' >/dev/null 2>&1; then
    ok "initialize returned a result"
else
    fail "initialize response missing .result.protocolVersion" "$init_resp"
    exit 1
fi

# Extract session ID if the server set one (FastMCP streamable-http does).
SESSION_ID=$(grep -i '^mcp-session-id:' "$HDR_FILE" | head -1 | sed -E 's/^[^:]*:[[:space:]]*//' | tr -d '\r\n')
SESS_HDR=""
if [ -n "$SESSION_ID" ]; then
    SESS_HDR="Mcp-Session-Id: $SESSION_ID"
    ok "session id obtained ($SESSION_ID)"
else
    fail "no Mcp-Session-Id header in initialize response — subsequent calls will fail" \
         "headers were: $(tr '\r' ' ' < "$HDR_FILE" | head -c 400)"
    exit 1
fi

# Send notifications/initialized so the server transitions to ready state.
post_rpc '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}' "$SESS_HDR" >/dev/null || true

# --- 2. resources/list ----------------------------------------------------
echo
echo "[2/4] resources/list"
list_resp=$(post_rpc '{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{}}' "$SESS_HDR") \
    || { fail "resources/list POST failed"; exit 1; }

res_entry=$(echo "$list_resp" | jq --arg u "$RESOURCE_URI" '.result.resources[] | select(.uri==$u)')
if [ -n "$res_entry" ]; then
    ok "resource $RESOURCE_URI is listed"
else
    fail "resource $RESOURCE_URI not in resources/list" "$list_resp"
fi

mime=$(echo "$res_entry" | jq -r '.mimeType // empty')
if [ "$mime" = "$EXPECTED_MIME" ]; then
    ok "list mimeType = $EXPECTED_MIME"
else
    fail "list mimeType mismatch" "got: $mime  expected: $EXPECTED_MIME"
fi

list_csp=$(echo "$res_entry" | jq -r '._meta.ui.csp.resourceDomains // empty | if type=="array" then join(",") else . end')
if echo ",$list_csp," | grep -q ",$EXPECTED_CDN,"; then
    ok "list _meta.ui.csp.resourceDomains includes $EXPECTED_CDN"
else
    fail "list _meta.ui.csp.resourceDomains missing CDN" "got: $list_csp"
fi

# --- 3. resources/read ----------------------------------------------------
echo
echo "[3/4] resources/read"
read_payload=$(jq -n --arg u "$RESOURCE_URI" '{
  jsonrpc:"2.0", id:3, method:"resources/read", params:{uri:$u}
}')
read_resp=$(post_rpc "$read_payload" "$SESS_HDR") \
    || { fail "resources/read POST failed"; exit 1; }

content=$(echo "$read_resp" | jq '.result.contents[0] // empty')
if [ -n "$content" ]; then
    ok "read returned at least one content item"
else
    fail "read response has no contents" "$read_resp"
    exit 1
fi

read_mime=$(echo "$content" | jq -r '.mimeType // empty')
if [ "$read_mime" = "$EXPECTED_MIME" ]; then
    ok "read content[0].mimeType = $EXPECTED_MIME"
else
    fail "read content[0].mimeType mismatch" "got: $read_mime"
fi

html_len=$(echo "$content" | jq -r '.text // "" | length')
if [ "$html_len" -gt 100 ]; then
    ok "read content[0].text non-empty ($html_len bytes)"
else
    fail "read content[0].text suspiciously short" "$html_len bytes"
fi

# Check for the cx-chart canvas + the CanvasXpress constructor in the HTML.
html=$(echo "$content" | jq -r '.text // ""')
if echo "$html" | grep -q 'id="cx-chart"'; then
    ok 'HTML contains <canvas id="cx-chart">'
else
    fail 'HTML missing cx-chart canvas'
fi
if echo "$html" | grep -q 'new CanvasXpress'; then
    ok "HTML contains 'new CanvasXpress' constructor"
else
    fail "HTML missing CanvasXpress constructor call"
fi

content_csp=$(echo "$content" | jq -r '._meta.ui.csp.resourceDomains // empty | if type=="array" then join(",") else . end')
if echo ",$content_csp," | grep -q ",$EXPECTED_CDN,"; then
    ok "read content[0]._meta.ui.csp.resourceDomains includes $EXPECTED_CDN"
else
    fail "read content[0]._meta.ui.csp missing CDN" "got: $content_csp"
fi

result_csp=$(echo "$read_resp" | jq -r '.result._meta.ui.csp.resourceDomains // empty | if type=="array" then join(",") else . end')
if echo ",$result_csp," | grep -q ",$EXPECTED_CDN,"; then
    ok "read result._meta.ui.csp.resourceDomains includes $EXPECTED_CDN"
else
    fail "read result._meta.ui.csp missing CDN" "got: $result_csp"
fi

# --- 4. tools/list --------------------------------------------------------
echo
echo "[4/4] tools/list — _meta.ui.resourceUri on chart tools"
tools_resp=$(post_rpc '{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{}}' "$SESS_HDR") \
    || { fail "tools/list POST failed"; exit 1; }

for t in "${CHART_TOOLS[@]}"; do
    ru=$(echo "$tools_resp" | jq -r --arg n "$t" '.result.tools[] | select(.name==$n) | ._meta.ui.resourceUri // empty')
    if [ "$ru" = "$RESOURCE_URI" ]; then
        ok "$t → _meta.ui.resourceUri = $RESOURCE_URI"
    else
        fail "$t missing or wrong _meta.ui.resourceUri" "got: $ru"
    fi
done

ru_neg=$(echo "$tools_resp" | jq -r --arg n "$NON_CHART_TOOL" '.result.tools[] | select(.name==$n) | ._meta.ui.resourceUri // empty')
if [ -z "$ru_neg" ]; then
    ok "$NON_CHART_TOOL has no _meta.ui.resourceUri (correct — out of scope)"
else
    fail "$NON_CHART_TOOL unexpectedly carries _meta.ui.resourceUri" "got: $ru_neg"
fi

# --- summary --------------------------------------------------------------
echo
TOTAL=$((PASS+FAIL))
if [ "$FAIL" -ne 0 ]; then
    printf "%s%d/%d checks failed%s\n" "$c_red" "$FAIL" "$TOTAL" "$c_reset"
    for f in "${FAILED_CHECKS[@]}"; do printf "  - %s\n" "$f"; done
    exit 1
fi

# --- 5. browser preview (optional) ----------------------------------------
# Call generate_canvasxpress_config with a small example, then synthesize a
# self-contained HTML file that iframes the MCP App resource HTML and posts
# the chart config the same way a host would (ui/notifications/tool-result).
# Open the file in a browser to actually SEE the chart render.
echo
echo "[5/5] browser preview"

call_payload=$(jq -n --arg n "generate_canvasxpress_config" '{
  jsonrpc:"2.0", id:5, method:"tools/call",
  params:{
    name:$n,
    arguments:{
      description:"Bar chart of expression by gene",
      headers:["Gene","Expression"],
      data:[
        ["Gene","Expression"],
        ["BRCA1",1.2],["TP53",2.5],["EGFR",0.8],["MYC",3.1],["KRAS",1.9]
      ],
      column_types:{"Gene":"string","Expression":"numeric"}
    }
  }
}')
call_resp=$(post_rpc "$call_payload" "$SESS_HDR") || { fail "tools/call POST failed"; exit 1; }

# The tool result content[0].text is a JSON string containing {config, data,
# valid, warnings, ...}. The MCP App iframe parses that wrapper and renders
# via `new CanvasXpress({renderTo, config, data})` (object-form constructor).
# Mirror that exactly here so the preview validates the same code path the
# iframe will exercise in a real host.
tool_payload=$(echo "$call_resp" | jq -c '.result.content[0].text | fromjson' 2>/dev/null)
if [ -z "$tool_payload" ] || [ "$tool_payload" = "null" ]; then
    fail "tools/call did not return a usable tool result payload" "$call_resp"
    exit 1
fi
has_config=$(echo "$tool_payload" | jq -r '.config.graphType // empty')
has_data=$(echo "$tool_payload" | jq -r 'if (.data|type)=="array" then "yes" else "no" end')
if [ -z "$has_config" ]; then
    fail "tool payload missing config.graphType" "$tool_payload"
    exit 1
fi
if [ "$has_data" != "yes" ]; then
    fail "tool payload missing data array (required for MCP App inline rendering)" "$tool_payload"
    exit 1
fi
ok "generate_canvasxpress_config returned config + data ($(echo "$tool_payload" | wc -c) bytes)"

# Encode full payload as a JSON literal for embedding in <script>.
payload_js=$(echo "$tool_payload" | jq -c .)

# Fetch the CanvasXpress library once and inline it into preview.html so the
# file works completely offline — required for environments where Chrome's
# enterprise policy blocks https:// subresources loaded from file:// pages
# (you'll see "blocked:origin" in the Network panel).
CXLIB_CACHE="$(cd "$(dirname "$0")" && pwd)/.canvasxpress.min.js"
if [ ! -s "$CXLIB_CACHE" ]; then
    echo "  fetching canvasXpress.min.js (one-time, cached at $CXLIB_CACHE)"
    if ! curl -fsSL https://www.canvasxpress.org/dist/canvasXpress.min.js -o "$CXLIB_CACHE"; then
        fail "could not download canvasXpress.min.js for inlining" \
             "preview.html will not work offline; check network from this host"
        exit 1
    fi
fi
cxlib_size=$(wc -c < "$CXLIB_CACHE")
ok "canvasXpress library available for inlining ($cxlib_size bytes)"

preview_file="$(cd "$(dirname "$0")" && pwd)/preview.html"
# Stream the file together in pieces to avoid loading the ~MB library into
# a shell variable.
{
    cat <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CanvasXpress MCP App preview</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; font-family: system-ui, sans-serif; }
    header { padding: 8px 12px; background: #222; color: #eee; font-size: 13px; }
    header code { background: #333; padding: 2px 6px; border-radius: 3px; }
    #cx-chart { width: 100vw; height: calc(100vh - 38px); display: block; }
    #cx-error { color: #b00; padding: 12px; font-family: monospace; white-space: pre-wrap; }
  </style>
  <script>
/* === Inlined canvasXpress.min.js (no network access required) === */
EOF
    cat "$CXLIB_CACHE"
    cat <<EOF
/* === end inlined library === */
  </script>
</head>
<body>
  <header>
    CanvasXpress MCP App preview — chart config generated by
    <code>generate_canvasxpress_config</code> via MCP <code>tools/call</code>.
    Library inlined; no network needed.
  </header>
  <canvas id="cx-chart"></canvas>
  <div id="cx-error" style="display:none"></div>
  <script>
    const TOOL_PAYLOAD = ${payload_js};
    function showError(msg) {
      const el = document.getElementById('cx-error');
      el.textContent = '[preview] ' + msg;
      el.style.display = 'block';
      console.error('[cx-preview]', msg);
    }
    window.addEventListener('DOMContentLoaded', function() {
      if (typeof CanvasXpress === 'undefined') {
        showError('CanvasXpress library not defined — inline script failed to evaluate.');
        return;
      }
      const config = TOOL_PAYLOAD && TOOL_PAYLOAD.config;
      const data   = TOOL_PAYLOAD && TOOL_PAYLOAD.data;
      if (!config || !config.graphType) { showError('payload.config.graphType missing'); return; }
      if (!data) { showError('payload.data missing — MCP Apps inline rendering requires data'); return; }
      try {
        new CanvasXpress({ renderTo: 'cx-chart', config: config, data: data });
      } catch (e) {
        showError('Render failed: ' + (e.message || String(e)));
      }
    });
  </script>
</body>
</html>
EOF
} > "$preview_file"

ok "wrote preview to $preview_file"
echo
printf "%sOpen in a browser to view the rendered chart:%s\n" "$c_green" "$c_reset"
printf "  file://%s\n" "$preview_file"
echo
TOTAL=$((PASS+FAIL))
printf "%s%d/%d checks passed%s\n" "$c_green" "$PASS" "$TOTAL" "$c_reset"
exit 0
