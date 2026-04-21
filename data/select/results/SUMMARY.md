# select_chart Test Results
Generated: 2026-04-20 22:58  |  Server: http://localhost:8100/select

## Overall

| Metric | Count | % |
|--------|------:|---|
| **Total tests** | 52 | 100% |
| Recommended (top pick) | 19 | 37% |
| Alternative (in top-N) | 13 | 25% |
| **PASS (rec + alt)** | **32** | **62%** |
| MISS | 20 | 38% |

## Recommended (top pick matched) — 19

| Expected | Recommended | Score | Intent |
|----------|-------------|------:|--------|
| correlation | Correlation | 1.0 | Show correlation matrix of gene expression across samples |
| dotplot | Dotplot | 0.85 | Show individual data points for small cohort by group |
| dumbbell | Dumbbell | 1.0 | Before vs after comparison across regions as dumbbell chart |
| gantt | Gantt | 1.0 | Show project schedule with task start and end dates by phase |
| heatmap | Heatmap | 1.0 | Show gene expression heatmap across samples |
| kaplan_meier | KaplanMeier | 1.0 | Survival analysis comparing drug vs placebo arm |
| line | Line | 0.85 | Show monthly sales trend by region over time |
| pareto | Pareto | 1.0 | Pareto chart of defect types by frequency |
| qq | QQ | 1.0 | QQ plot to assess normality of observed residuals |
| sankey | Sankey | 1.0 | Show patient flow through clinical trial stages |
| scatter2d | Scatter2D | 1.0 | Show PCA of samples colored by condition |
| scatter3d | Scatter3D | 1.0 | 3D scatter of samples across three principal components by c |
| scatter_bubble_2d | ScatterBubble2D | 1.0 | Bubble chart of GDP per capita vs life expectancy sized by p |
| spaghetti | Spaghetti | 1.0 | Individual patient lab trajectories over visits by treatment |
| splom | SPLOM | 1.0 | Pairwise scatter matrix of iris measurements by species |
| tornado | Tornado | 1.0 | Tornado sensitivity analysis of key business variables |
| venn | Venn | 1.0 | Show overlap between three gene sets |
| violin | Violin | 1.0 | Show gene expression distribution by cell type |
| volcano | Volcano | 1.0 | Volcano plot of differential expression log2FC vs significan |

## Alternative (expected chart in alternatives) — 13

| Expected | Recommended | Alt position | Top 3 Alts | Intent |
|----------|-------------|:------------:|------------|--------|
| bar | Dotplot | alt#1 | Bar, Heatmap, Treemap | Compare average values across four groups |
| boxplot | Violin | alt#2 | Bar, Boxplot, Heatmap | Compare distribution of values across treatment gr |
| bubble | ScatterBubble2D | alt#2 | Scatter2D, Bubble, Heatmap | Bubble chart with x, y position and bubble size as |
| chord | Sankey | alt#3 | Alluvial, Stacked, Chord | Show trade flow between global regions |
| density | Violin | alt#3 | Bar, Boxplot, Density | Compare smooth density distribution of biomarker b |
| histogram | Violin | alt#2 | Boxplot, Histogram, Heatmap | Show frequency distribution of continuous measurem |
| lollipop | Boxplot | alt#2 | Heatmap, Lollipop, Volcano | Rank genes by expression fold change as lollipop c |
| network | Sankey | alt#1 | Network, Alluvial, Stacked | Show network connections between nodes with edge w |
| stacked | Dumbbell | alt#1 | Stacked, DotLine, Dotplot | Show quarterly breakdown by region as stacked bar |
| stacked_percent | Stacked | alt#2 | SPLOM, StackedPercent, Dotplot | Show age group composition per product as 100% sta |
| sunburst | Treemap | alt#3 | Bar, Stacked, Sunburst | Show adverse event hierarchy by SOC, preferred ter |
| tree_bracket | Dotplot | alt#3 | Ridgeline, Upset, TreeBracket | Tournament bracket results by round |
| waterfall | Bar | alt#1 | Waterfall, Dotplot, Ridgeline | Show cumulative financial waterfall from sales to  |

## MISS (expected chart not found) — 20

| Expected | Recommended | Top 3 Alts | Intent |
|----------|-------------|------------|--------|
| alluvial | Upset | Treemap, TreeBracket, Network | Show patient response state transitions from basel |
| area | Dotplot | Treemap, Ridgeline, Bar | Show revenue area chart over months by region |
| bar_line | Line | Bar, Heatmap, Boxplot | Show monthly sales bars with market share trend li |
| binplot | Scatter2D | Heatmap, Boxplot, Treemap | Bin scatter of two numeric variables across two gr |
| bullet | DotLine | Dotplot, Scatter3D, Ridgeline | Bullet chart comparing actual performance against  |
| bump | Dotplot | Boxplot, Scatter2D, Ridgeline | Show ranking changes of drugs over years |
| cdf | Violin | Boxplot, Heatmap, Histogram | Cumulative distribution comparison between two gro |
| cleveland | Bar | SPLOM, Dotplot, Ridgeline | Cleveland dot plot comparing metric values in 2020 |
| contour | Violin | Boxplot, Correlation, Density | Show 2D density contour of two correlated biomarke |
| dot_line | Bar | Stacked, Heatmap, StackedPercent | Show mean response with SD at each timepoint by tr |
| hexplot | Scatter2D | Density, Boxplot, Ridgeline | Show density of two continuous variables with 500  |
| parallel_coordinates | Heatmap | Boxplot, Scatter2D, Scatter3D | Show multivariate profiles of three classes |
| radar | Bar | Dotplot, Scatter3D, Ridgeline | Compare team performance across five skill dimensi |
| ribbon | Gantt | Heatmap, Boxplot, Scatter2D | Ribbon plot of ranges per group |
| ridgeline | Violin | Bar, Boxplot, Histogram | Compare score distributions across five department |
| streamgraph | Stacked | Boxplot, Scatter2D, Line | Show composition of device usage over time as stre |
| time_series | Line | Bar, Density, Boxplot | Sensor readings over 48-hour period for two sensor |
| treemap | Heatmap | Bar, Stacked, StackedPercent | Show headcount hierarchically by department and su |
| upset | TagCloud | TreeBracket, Violin, Boxplot | Upset plot showing intersections of four gene sets |
| wordcloud | TagCloud | Dotplot, Treemap, Ridgeline | Show frequency of clinical research terms as word  |

## Notes

- Test data in `data/select/<name>.json` — one JSON file per chart type.
- Per-test API responses saved in `data/select/results/<name>.json`.
- Matching is case-insensitive and ignores punctuation/spaces.
- A test **passes** if the expected chart type appears as either the top recommendation or in any alternative.
- Script to regenerate test data: `gen_select_data.py`
