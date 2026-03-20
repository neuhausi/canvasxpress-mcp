/**
 * CanvasXpress MCP Test Client — Node.js
 *
 * Usage:
 *   # Run built-in examples (generate + modify)
 *   node test_client.mjs --examples
 *
 *   # Generate a config
 *   node test_client.mjs "Violin plot by cell type"
 *   node test_client.mjs "Heatmap" "Gene,Sample1,Treatment"
 *   node test_client.mjs "Scatter plot" "Gene,Expr,Treatment" '{"Gene":"string","Expr":"numeric","Treatment":"factor"}'
 *   node test_client.mjs "Heatmap" '[["Gene","S1","S2"],["BRCA1",1.2,3.4]]' '{"Gene":"string","S1":"numeric","S2":"numeric"}'
 *
 *   # Modify an existing config
 *   node test_client.mjs --modify '{"graphType":"Bar","xAxis":["Gene"]}' "add a title My Chart"
 *   node test_client.mjs --modify '{"graphType":"Heatmap","xAxis":["Gene"]}' "change colorScheme to Spectral"
 *
 * Requirements:
 *   npm install @modelcontextprotocol/sdk
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const MCP_URL = process.env.MCP_URL || "http://localhost:8100/mcp";
const SEP  = "─".repeat(50);
const SEP2 = "═".repeat(50);

// ---------------------------------------------------------------------------
// Built-in examples
// ---------------------------------------------------------------------------

const GENERATE_EXAMPLES = [
  {
    label: "Clustered heatmap",
    description: "Clustered heatmap with RdBu colors and dendrograms on both axes",
    data: [
      ["Gene",  "Control1", "Control2", "Drug1", "Drug2"],
      ["BRCA1", 2.1,        0.9,        3.8,     3.2   ],
      ["TP53",  1.2,        1.4,        0.3,     0.5   ],
      ["EGFR",  0.8,        0.6,        2.9,     3.1   ],
      ["MYC",   3.2,        2.8,        0.4,     0.6   ],
    ],
    columnTypes: { Gene: "string", Control1: "numeric", Control2: "numeric",
                   Drug1: "numeric", Drug2: "numeric" },
  },
  {
    label: "Volcano plot",
    description: "Volcano plot with log2 fold change on x-axis and -log10 p-value on y-axis",
    data: [
      ["Gene",  "log2FC", "negLog10P"],
      ["GeneA",  2.3,      4.1       ],
      ["GeneB", -1.8,      3.7       ],
      ["GeneC",  0.2,      0.4       ],
      ["GeneD",  3.1,      6.2       ],
    ],
    columnTypes: { Gene: "string", log2FC: "numeric", negLog10P: "numeric" },
  },
  {
    label: "Violin plot",
    description: "Violin plot of gene expression grouped by cell type with Tableau colors",
    headers: ["CellID", "Expression", "CellType"],
    columnTypes: { CellID: "string", Expression: "numeric", CellType: "factor" },
  },
  {
    label: "PCA scatter plot",
    description: "PCA scatter plot with PC1 vs PC2 colored by Treatment with regression ellipses",
    headers: ["Sample", "PC1", "PC2", "Treatment"],
    columnTypes: { Sample: "string", PC1: "numeric", PC2: "numeric", Treatment: "factor" },
  },
  {
    label: "Kaplan-Meier survival curve",
    description: "Kaplan-Meier survival curve for two treatment groups",
    headers: ["Patient", "Time", "Event", "Treatment"],
    columnTypes: { Patient: "string", Time: "numeric", Event: "numeric", Treatment: "factor" },
  },
  {
    label: "Stacked percent bar",
    description: "Stacked percent bar chart of market share by year and company",
    data: [
      ["Company", "Y2021", "Y2022", "Y2023"],
      ["Alpha",    35,      28,      31     ],
      ["Beta",     28,      33,      29     ],
      ["Gamma",    37,      39,      40     ],
    ],
    columnTypes: { Company: "string", Y2021: "numeric", Y2022: "numeric", Y2023: "numeric" },
  },
  {
    label: "Ridgeline density",
    description: "Ridgeline density plot of expression values by cell population",
    headers: ["Cell", "Value", "Population"],
    columnTypes: { Cell: "string", Value: "numeric", Population: "factor" },
  },
  {
    label: "Sankey flow diagram",
    description: "Sankey diagram showing patient flow from diagnosis through treatment to outcome",
    headers: ["Diagnosis", "Treatment", "Outcome"],
    columnTypes: { Diagnosis: "factor", Treatment: "factor", Outcome: "factor" },
  },
];

const MODIFY_EXAMPLES = [
  {
    label: "Add title and switch theme",
    startConfig: {
      graphType: "Heatmap", xAxis: ["Gene"],
      samplesClustered: true, variablesClustered: true, colorScheme: "RdBu",
    },
    instruction: "add a title Expression Heatmap and switch to dark theme",
  },
  {
    label: "Change color scheme and add title",
    startConfig: { graphType: "Bar", xAxis: ["Region"], graphOrientation: "horizontal" },
    instruction: "change the color scheme to Tableau and add a title Regional Sales",
  },
  {
    label: "Remove legend and set axis titles",
    startConfig: {
      graphType: "Scatter2D", xAxis: ["PC1"], yAxis: ["PC2"],
      colorBy: "Treatment", showLegend: true,
    },
    instruction: "remove the legend and set xAxisTitle to PC1 (32%) and yAxisTitle to PC2 (18%)",
  },
  {
    label: "Add grouping and jitter",
    startConfig: { graphType: "Boxplot", xAxis: ["Expression"] },
    instruction: "add groupingFactors for the CellType column and enable jitter on the data points",
  },
];

// ---------------------------------------------------------------------------
// MCP helpers
// ---------------------------------------------------------------------------

async function makeClient() {
  const client = new Client({ name: "cx-test", version: "1.0.0" });
  const transport = new StreamableHTTPClientTransport(new URL(MCP_URL));
  await client.connect(transport);
  return client;
}

async function callTool(toolName, args) {
  const client = await makeClient();
  console.log(`Connected  : ${MCP_URL}`);
  try {
    const result = await client.callTool({ name: toolName, arguments: args });
    return JSON.parse(result.content[0].text);
  } finally {
    await client.close();
  }
}

function parseExtraArgs(extraArgs) {
  let headers = null, data = null, columnTypes = null;
  for (const arg of extraArgs) {
    if (arg.startsWith("{"))      columnTypes = JSON.parse(arg);
    else if (arg.startsWith("[")) data        = JSON.parse(arg);
    else                          headers     = arg.split(",");
  }
  return { headers, data, columnTypes };
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

function printGenerateResult(response) {
  if (response.headers_used?.length)
    console.log(`Headers used : ${response.headers_used.join(", ")}`);
  if (response.types_used && Object.keys(response.types_used).length)
    console.log(`Types used   : ${Object.entries(response.types_used).map(([k,v])=>`${k}=${v}`).join(", ")}`);
  console.log();
  console.log(`── Config ${SEP}`);
  console.log(JSON.stringify(response.config, null, 2));
  console.log(`\n── Validation ${SEP}`);
  if (response.valid) {
    console.log("✅ All column references are valid");
  } else {
    console.log("⚠️  Column reference warnings:");
    response.warnings.forEach(w => console.log(`   • ${w}`));
    if (Object.keys(response.invalid_refs || {}).length)
      console.log("\n   Invalid refs:", JSON.stringify(response.invalid_refs, null, 2));
  }
}

function printModifyResult(original, response, instruction) {
  const ch = response.changes || {};
  console.log(`── Changes ${SEP}`);
  console.log(`   Instruction : ${instruction}`);
  console.log(`   Added       : ${ch.added?.length   ? ch.added.join(", ")   : "none"}`);
  console.log(`   Removed     : ${ch.removed?.length ? ch.removed.join(", ") : "none"}`);
  console.log(`   Changed     : ${ch.changed?.length ? ch.changed.join(", ") : "none"}`);
  console.log(`\n── Modified config ${SEP}`);
  console.log(JSON.stringify(response.config, null, 2));
  console.log(`\n── Validation ${SEP}`);
  if (response.valid) {
    console.log("✅ All column references are valid");
  } else {
    console.log("⚠️  Column reference warnings:");
    response.warnings.forEach(w => console.log(`   • ${w}`));
  }
}

// ---------------------------------------------------------------------------
// --examples mode
// ---------------------------------------------------------------------------

async function runExamples() {
  console.log(`\n${SEP2}`);
  console.log("  CanvasXpress MCP — Built-in Examples");
  console.log(`  Server : ${MCP_URL}`);
  console.log(SEP2);

  console.log(`\n${SEP}\n  GENERATE EXAMPLES\n${SEP}`);
  for (let i = 0; i < GENERATE_EXAMPLES.length; i++) {
    const ex = GENERATE_EXAMPLES[i];
    console.log(`\n[${i+1}/${GENERATE_EXAMPLES.length}] ${ex.label}`);
    console.log(`  Description : ${ex.description}`);
    if (ex.data)
      console.log(`  Data        : ${ex.data.length-1} rows × ${ex.data[0].length} columns  (${ex.data[0].join(", ")})`);
    else if (ex.headers)
      console.log(`  Headers     : ${ex.headers.join(", ")}`);
    console.log();
    try {
      const args = { description: ex.description };
      if (ex.data)        args.data         = ex.data;
      else if (ex.headers) args.headers     = ex.headers;
      if (ex.columnTypes) args.column_types = ex.columnTypes;
      const response = await callTool("generate_canvasxpress_config", args);
      printGenerateResult(response);
    } catch (e) {
      console.log(`  ❌ Error: ${e.message}`);
    }
    if (i < GENERATE_EXAMPLES.length - 1) console.log(`\n${SEP}`);
  }

  console.log(`\n\n${SEP}\n  MODIFY EXAMPLES\n${SEP}`);
  for (let i = 0; i < MODIFY_EXAMPLES.length; i++) {
    const ex = MODIFY_EXAMPLES[i];
    console.log(`\n[${i+1}/${MODIFY_EXAMPLES.length}] ${ex.label}`);
    console.log(`  Instruction  : ${ex.instruction}`);
    console.log(`  Start config : ${JSON.stringify(ex.startConfig)}`);
    console.log();
    try {
      const response = await callTool("modify_canvasxpress_config", {
        config: ex.startConfig,
        instruction: ex.instruction,
      });
      printModifyResult(ex.startConfig, response, ex.instruction);
    } catch (e) {
      console.log(`  ❌ Error: ${e.message}`);
    }
    if (i < MODIFY_EXAMPLES.length - 1) console.log(`\n${SEP}`);
  }

  console.log(`\n${SEP2}\n`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const args = process.argv.slice(2);

  if (args[0] === "--examples") {
    await runExamples();
    return;
  }

  if (args[0] === "--modify") {
    if (args.length < 3) {
      console.error("Usage: node test_client.mjs --modify '<config_json>' '<instruction>'");
      process.exit(1);
    }
    const originalConfig = JSON.parse(args[1]);
    const instruction    = args[2];
    const { headers, data, columnTypes } = parseExtraArgs(args.slice(3));

    console.log(`Tool        : modify_canvasxpress_config`);
    console.log(`Instruction : ${instruction}`);
    console.log(`Config keys : ${Object.keys(originalConfig).join(", ")}`);
    console.log();

    const toolArgs = { config: originalConfig, instruction };
    if (data)        toolArgs.data         = data;
    else if (headers) toolArgs.headers     = headers;
    if (columnTypes) toolArgs.column_types = columnTypes;

    const response = await callTool("modify_canvasxpress_config", toolArgs);
    printModifyResult(originalConfig, response, instruction);
    return;
  }

  // Default: generate
  const description = args[0] || "Clustered heatmap with RdBu colors";
  let { headers, data, columnTypes } = parseExtraArgs(args.slice(1));

  if (!headers && !data) {
    data = [
      ["Gene",  "Sample1", "Sample2", "Treatment"],
      ["BRCA1", 1.2,       3.4,       "Control"  ],
      ["TP53",  2.1,       0.9,       "Treated"  ],
      ["EGFR",  0.8,       2.3,       "Control"  ],
    ];
    columnTypes = { Gene: "string", Sample1: "numeric", Sample2: "numeric", Treatment: "factor" };
  }

  console.log(`Tool        : generate_canvasxpress_config`);
  console.log(`Description : ${description}`);
  if (data)    console.log(`Data        : ${data.length-1} rows × ${data[0].length} columns\nColumns     : ${data[0].join(", ")}`);
  if (headers) console.log(`Headers     : ${headers.join(", ")}`);
  if (columnTypes) console.log(`Types       : ${Object.entries(columnTypes).map(([k,v])=>`${k}=${v}`).join(", ")}`);
  console.log();

  const toolArgs = { description };
  if (data)         toolArgs.data         = data;
  else if (headers) toolArgs.headers      = headers;
  if (columnTypes)  toolArgs.column_types = columnTypes;

  const response = await callTool("generate_canvasxpress_config", toolArgs);
  printGenerateResult(response);
}

main().catch(err => { console.error("Error:", err.message); process.exit(1); });
