/**
 * CanvasXpress MCP Test Client — Node.js
 *
 * Usage:
 *   node test_client.mjs
 *   node test_client.mjs "Violin plot by cell type" "Gene,CellType,Expression"
 *   node test_client.mjs "Heatmap" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'
 *   node test_client.mjs "Heatmap" "Gene,Sample1,Treatment" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const MCP_URL = process.env.MCP_URL || "http://localhost:8100/mcp";

async function main() {
  const client = new Client({ name: "cx-test", version: "1.0.0" });
  const transport = new StreamableHTTPClientTransport(new URL(MCP_URL));
  await client.connect(transport);
  console.log(`Connected  : ${MCP_URL}\n`);

  const description = process.argv[2] || "Clustered heatmap with RdBu colors";

  // Parse remaining args flexibly — JSON object = column_types, JSON array = data, else = headers
  let headers = null, data = null, columnTypes = null;
  for (const arg of process.argv.slice(3)) {
    if (arg.startsWith("{")) columnTypes = JSON.parse(arg);
    else if (arg.startsWith("[")) data = JSON.parse(arg);
    else headers = arg.split(",");
  }

  // Default: sample data + types
  if (!headers && !data) {
    data = [
      ["Gene",  "Sample1", "Sample2", "Treatment"],
      ["BRCA1", 1.2,       3.4,       "Control"  ],
      ["TP53",  2.1,       0.9,       "Treated"  ],
      ["EGFR",  0.8,       2.3,       "Control"  ],
    ];
    columnTypes = { Gene: "string", Sample1: "numeric", Sample2: "numeric", Treatment: "factor" };
  }

  // Print summary
  console.log(`Description : ${description}`);
  if (data)    console.log(`Data        : ${data.length-1} rows × ${data[0].length} columns\nColumns     : ${data[0].join(", ")}`);
  if (headers) console.log(`Headers     : ${headers.join(", ")}`);
  if (columnTypes) console.log(`Types       : ${Object.entries(columnTypes).map(([k,v])=>`${k}=${v}`).join(", ")}`);
  console.log();

  // Build args
  const args = { description };
  if (data)        args.data         = data;
  else if (headers) args.headers     = headers;
  if (columnTypes) args.column_types = columnTypes;

  const result = await client.callTool({
    name: "generate_canvasxpress_config",
    arguments: args,
  });

  const response = JSON.parse(result.content[0].text);

  if (response.headers_used?.length)
    console.log(`Headers used : ${response.headers_used.join(", ")}`);
  if (response.types_used && Object.keys(response.types_used).length)
    console.log(`Types used   : ${Object.entries(response.types_used).map(([k,v])=>`${k}=${v}`).join(", ")}`);
  console.log();

  console.log("── Config ──────────────────────────────────");
  console.log(JSON.stringify(response.config, null, 2));

  console.log("\n── Validation ──────────────────────────────");
  if (response.valid) {
    console.log("✅ All column references are valid");
  } else {
    console.log("⚠️  Column reference warnings:");
    response.warnings.forEach(w => console.log(`   • ${w}`));
    if (Object.keys(response.invalid_refs).length)
      console.log("\n   Invalid refs:", JSON.stringify(response.invalid_refs, null, 2));
  }

  await client.close();
}

main().catch(err => { console.error("Error:", err.message); process.exit(1); });
