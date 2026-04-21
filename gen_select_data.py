"""
Generate test data files in data/select/ for every CanvasXpress chart type.
Each file contains {"intent": "...", "data": [[header...], [row...]...]}
Run from repo root: python3 gen_select_data.py
"""
import json, os, random, datetime

random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "data", "select")
os.makedirs(OUT, exist_ok=True)


def write(name, rows, intent):
    path = os.path.join(OUT, name + ".json")
    with open(path, "w") as f:
        json.dump({"intent": intent, "data": rows}, f, indent=2)
    print("wrote", path)


def cat_num(n, cats, mean=50, sd=10):
    rows = [["Sample", "Group", "Value"]]
    for i in range(n):
        g = cats[i % len(cats)]
        rows.append([f"S{i+1}", g, round(random.gauss(mean + (ord(g[0]) % 20), sd), 1)])
    return rows


# ── Bar ───────────────────────────────────────────────────────────────────────
write("bar", cat_num(40, ["TypeA", "TypeB", "TypeC", "TypeD"]),
      "Compare average values across four groups")

# ── Stacked ───────────────────────────────────────────────────────────────────
r = [["Category", "Q1", "Q2", "Q3", "Q4"]]
for c in ["North", "South", "East", "West"]:
    r.append([c] + [random.randint(20, 80) for _ in range(4)])
write("stacked", r, "Show quarterly breakdown by region as stacked bar")

# ── StackedPercent ────────────────────────────────────────────────────────────
r = [["Product", "Age18_34", "Age35_54", "Age55plus"]]
for p in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
    r.append([p] + [random.randint(15, 50) for _ in range(3)])
write("stacked_percent", r, "Show age group composition per product as 100% stacked bar")

# ── Boxplot ───────────────────────────────────────────────────────────────────
write("boxplot", cat_num(80, ["Control", "TreatmentA", "TreatmentB"]),
      "Compare distribution of values across treatment groups")

# ── Violin ────────────────────────────────────────────────────────────────────
write("violin", cat_num(100, ["CellTypeA", "CellTypeB", "CellTypeC", "CellTypeD"]),
      "Show gene expression distribution by cell type")

# ── Dotplot ───────────────────────────────────────────────────────────────────
write("dotplot", cat_num(20, ["GroupA", "GroupB"]),
      "Show individual data points for small cohort by group")

# ── Heatmap ───────────────────────────────────────────────────────────────────
r = [["Gene"] + [f"Sample{i+1}" for i in range(8)]]
for g in [f"Gene{i+1}" for i in range(12)]:
    r.append([g] + [round(random.gauss(0, 2), 2) for _ in range(8)])
write("heatmap", r, "Show gene expression heatmap across samples")

# ── Line ──────────────────────────────────────────────────────────────────────
r = [["Month", "Sales", "Region"]]
for g in ["North", "South"]:
    for m in range(1, 13):
        r.append([f"2024-{m:02d}", round(random.gauss(100, 15), 1), g])
write("line", r, "Show monthly sales trend by region over time")

# ── Area ──────────────────────────────────────────────────────────────────────
r = [["Month", "Revenue", "Region"]]
for g in ["East", "West", "Central"]:
    for m in range(1, 13):
        r.append([f"2024-{m:02d}", round(random.gauss(80, 12), 1), g])
write("area", r, "Show revenue area chart over months by region")

# ── Scatter2D ─────────────────────────────────────────────────────────────────
r = [["Sample", "PCA1", "PCA2", "Condition"]]
for cond in ["WT", "KO", "Treated"]:
    for i in range(30):
        r.append([f"{cond}_{i+1}",
                  round(random.gauss(ord(cond[0]) % 5 * 3, 2), 2),
                  round(random.gauss(ord(cond[0]) % 3 * 2, 2), 2),
                  cond])
write("scatter2d", r, "Show PCA of samples colored by condition")

# ── Scatter3D ─────────────────────────────────────────────────────────────────
r = [["Sample", "PC1", "PC2", "PC3", "Cluster"]]
for cl in ["A", "B", "C"]:
    for i in range(20):
        r.append([f"{cl}_{i+1}",
                  round(random.gauss(ord(cl) * 2, 3), 2),
                  round(random.gauss(ord(cl), 2), 2),
                  round(random.gauss(ord(cl) * 1.5, 2), 2),
                  cl])
write("scatter3d", r, "3D scatter of samples across three principal components by cluster")

# ── Volcano ───────────────────────────────────────────────────────────────────
r = [["Gene", "log2FC", "negLog10Pvalue"]]
for i in range(200):
    r.append([f"Gene{i+1}", round(random.gauss(0, 1.5), 3),
              round(random.uniform(0.5, 8), 3)])
write("volcano", r, "Volcano plot of differential expression log2FC vs significance")

# ── KaplanMeier ───────────────────────────────────────────────────────────────
r = [["Patient", "Time", "Event", "Arm"]]
for arm in ["Placebo", "Drug"]:
    for i in range(30):
        r.append([f"{arm}_{i+1}", round(random.uniform(1, 36), 1),
                  random.choice([0, 1]), arm])
write("kaplan_meier", r, "Survival analysis comparing drug vs placebo arm")

# ── Correlation ───────────────────────────────────────────────────────────────
r = [["Sample"] + [f"Gene{i+1}" for i in range(8)]]
for i in range(30):
    r.append([f"S{i+1}"] + [round(random.gauss(5, 2), 2) for _ in range(8)])
write("correlation", r, "Show correlation matrix of gene expression across samples")

# ── Sankey ────────────────────────────────────────────────────────────────────
r = [["Source", "Target", "Value"]]
for s, t, v in [("ScreenedIn", "Enrolled", 80), ("ScreenedIn", "ScreenedOut", 40),
                ("Enrolled", "Completed", 60), ("Enrolled", "DropOut", 20)]:
    r.append([s, t, v])
write("sankey", r, "Show patient flow through clinical trial stages")

# ── Waterfall ─────────────────────────────────────────────────────────────────
r = [["Category", "Value"]]
for cat, val in [("Sales", 120), ("COGS", -45), ("Gross", -20), ("OpEx", -30),
                 ("EBIT", 15), ("Tax", -10), ("Net", 0)]:
    r.append([cat, val])
write("waterfall", r, "Show cumulative financial waterfall from sales to net income")

# ── Histogram ─────────────────────────────────────────────────────────────────
r = [["Sample", "Value", "Group"]]
for g in ["Normal", "Skewed"]:
    for i in range(80):
        v = round(random.gauss(50, 10) if g == "Normal" else abs(random.gauss(20, 15)), 1)
        r.append([f"{g}_{i+1}", v, g])
write("histogram", r, "Show frequency distribution of continuous measurement")

# ── Density ───────────────────────────────────────────────────────────────────
r = [["Sample", "Biomarker", "Group"]]
for g in ["Control", "Case"]:
    mean = 45 if g == "Control" else 60
    for i in range(60):
        r.append([f"{g}_{i+1}", round(random.gauss(mean, 8), 2), g])
write("density", r, "Compare smooth density distribution of biomarker between two groups")

# ── Ridgeline (canonical) ─────────────────────────────────────────────────────
depts = ["Engineering", "Marketing", "Finance", "Operations", "Research"]
means_d = {"Engineering": 79, "Marketing": 59, "Finance": 90,
           "Operations": 47, "Research": 68}
r = [["Sample", "Score", "Department"]]
for d in depts:
    for i in range(30):
        r.append([f"{d[:3]}_{i+1}", round(random.gauss(means_d[d], 7), 1), d])
write("ridgeline", r, "Compare score distributions across five departments")

# ── Treemap ───────────────────────────────────────────────────────────────────
r = [["Department", "SubDept", "Headcount"]]
for dept, subs in [("Engineering", ["Backend", "Frontend", "DevOps", "QA"]),
                   ("Sales", ["Inside", "Field"]),
                   ("HR", ["Recruiting", "L&D"])]:
    for s in subs:
        r.append([dept, s, random.randint(10, 60)])
write("treemap", r, "Show headcount hierarchically by department and sub-department")

# ── Network ───────────────────────────────────────────────────────────────────
r = [["Source", "Target", "Weight"]]
nodes = ["A", "B", "C", "D", "E", "F"]
for i, s in enumerate(nodes):
    for t in nodes[i + 1:]:
        if random.random() > 0.4:
            r.append([s, t, round(random.uniform(0.1, 1.0), 2)])
write("network", r, "Show network connections between nodes with edge weights")

# ── Venn ──────────────────────────────────────────────────────────────────────
r = [["Set", "Member"]]
for m in [f"G{i}" for i in range(1, 21)]:
    r.append(["SetA", m])
for m in [f"G{i}" for i in range(11, 31)]:
    r.append(["SetB", m])
for m in [f"G{i}" for i in range(6, 16)]:
    r.append(["SetC", m])
write("venn", r, "Show overlap between three gene sets")

# ── BarLine ───────────────────────────────────────────────────────────────────
r = [["Month", "Sales", "MarketShare", "Group"]]
for g in ["Product1", "Product2"]:
    for m in range(1, 13):
        r.append([f"2024-{m:02d}", random.randint(50, 150),
                  round(random.uniform(10, 40), 1), g])
write("bar_line", r, "Show monthly sales bars with market share trend line overlay")

# ── Hexplot ───────────────────────────────────────────────────────────────────
r = [["Sample", "X", "Y"]]
for i in range(500):
    r.append([f"S{i+1}", round(random.gauss(0, 3), 2), round(random.gauss(0, 3), 2)])
write("hexplot", r, "Show density of two continuous variables with 500 data points")

# ── Contour ───────────────────────────────────────────────────────────────────
r = [["Sample", "Biomarker1", "Biomarker2"]]
for i in range(300):
    x = round(random.gauss(0, 2), 2)
    r.append([f"S{i+1}", x, round(random.gauss(x * 0.5, 1.5), 2)])
write("contour", r, "Show 2D density contour of two correlated biomarkers")

# ── Sunburst ──────────────────────────────────────────────────────────────────
r = [["Level1", "Level2", "Level3", "Value"]]
for soc in ["Blood", "Cardiac", "Nervous"]:
    for pt in [f"{soc}_PT1", f"{soc}_PT2"]:
        for sev in ["Mild", "Moderate", "Severe"]:
            r.append([soc, pt, sev, random.randint(5, 30)])
write("sunburst", r, "Show adverse event hierarchy by SOC, preferred term, and severity")

# ── Chord ─────────────────────────────────────────────────────────────────────
r = [["From", "To", "Value"]]
regions = ["US", "EU", "APAC", "LATAM"]
for i, a in enumerate(regions):
    for b in regions[i + 1:]:
        r.append([a, b, random.randint(10, 100)])
write("chord", r, "Show trade flow between global regions")

# ── ParallelCoordinates ───────────────────────────────────────────────────────
r = [["Sample", "F1", "F2", "F3", "F4", "F5", "Class"]]
for cls in ["Class1", "Class2", "Class3"]:
    for i in range(25):
        r.append([f"{cls}_{i+1}"] +
                 [round(random.gauss(ord(cls[-1]) * 10, 5), 1) for _ in range(5)] +
                 [cls])
write("parallel_coordinates", r, "Show multivariate profiles of three classes")

# ── SPLOM ─────────────────────────────────────────────────────────────────────
r = [["Sample", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]]
for sp in ["setosa", "versicolor", "virginica"]:
    base = {"setosa": [5.0, 3.4, 1.5, 0.2],
            "versicolor": [5.9, 2.8, 4.3, 1.3],
            "virginica": [6.6, 3.0, 5.6, 2.0]}[sp]
    for i in range(20):
        r.append([f"{sp}_{i+1}"] +
                 [round(b + random.gauss(0, 0.4), 1) for b in base] +
                 [sp])
write("splom", r, "Pairwise scatter matrix of iris measurements by species")

# ── Radar ─────────────────────────────────────────────────────────────────────
r = [["Team", "Speed", "Strength", "Agility", "Endurance", "Precision"]]
for team in ["Alpha", "Beta", "Gamma", "Delta"]:
    r.append([team] + [random.randint(40, 100) for _ in range(5)])
write("radar", r, "Compare team performance across five skill dimensions")

# ── Streamgraph ───────────────────────────────────────────────────────────────
r = [["Year", "Category", "Value"]]
for yr in range(2015, 2026):
    for cat in ["Mobile", "Desktop", "Tablet", "TV", "Other"]:
        r.append([str(yr), cat, random.randint(10, 80)])
write("streamgraph", r, "Show composition of device usage over time as streamgraph")

# ── Gantt ─────────────────────────────────────────────────────────────────────
r = [["Task", "Start", "End", "Phase"]]
for t in [("Design", "2024-01-01", "2024-02-28", "Phase1"),
           ("Development", "2024-02-01", "2024-05-31", "Phase1"),
           ("Testing", "2024-05-01", "2024-06-30", "Phase2"),
           ("Staging", "2024-06-15", "2024-07-15", "Phase2"),
           ("Launch", "2024-07-01", "2024-07-31", "Phase3")]:
    r.append(list(t))
write("gantt", r, "Show project schedule with task start and end dates by phase")

# ── Lollipop ──────────────────────────────────────────────────────────────────
r = [["Gene", "Expression", "Direction"]]
for i in range(20):
    fc = round(random.gauss(0, 2), 2)
    r.append([f"Gene{i+1}", fc, "Up" if fc > 0 else "Down"])
write("lollipop", r, "Rank genes by expression fold change as lollipop chart")

# ── TagCloud / WordCloud ──────────────────────────────────────────────────────
r = [["Term", "Frequency", "Category"]]
for w, f in [("cancer", 85), ("therapy", 72), ("survival", 68), ("treatment", 65),
             ("biomarker", 60), ("mutation", 55), ("response", 52), ("prognosis", 48),
             ("immunotherapy", 45), ("gene", 40), ("protein", 38), ("cell", 35),
             ("pathway", 32), ("clinical", 30), ("trial", 28)]:
    r.append([w, f, random.choice(["Clinical", "Molecular"])])
write("wordcloud", r, "Show frequency of clinical research terms as word cloud")

# ── DotLine ───────────────────────────────────────────────────────────────────
r = [["Timepoint", "Mean", "SD", "Group"]]
for g in ["Drug", "Placebo"]:
    for t in ["BL", "W4", "W8", "W12", "W24"]:
        r.append([t, round(random.gauss(50, 5), 1), round(random.uniform(2, 6), 1), g])
write("dot_line", r, "Show mean response with SD at each timepoint by treatment group")

# ── ScatterBubble2D ───────────────────────────────────────────────────────────
r = [["Country", "GDPperCapita", "LifeExpectancy", "Population", "Region"]]
for row in [("US", 65000, 78, 330, "Americas"), ("DE", 48000, 81, 83, "Europe"),
            ("JP", 42000, 84, 125, "Asia"), ("BR", 15000, 75, 215, "Americas"),
            ("IN", 7000, 70, 1400, "Asia"), ("CN", 18000, 77, 1400, "Asia"),
            ("FR", 44000, 82, 67, "Europe"), ("NG", 5000, 62, 220, "Africa"),
            ("ZA", 13000, 64, 60, "Africa"), ("AU", 55000, 83, 26, "Oceania"),
            ("MX", 19000, 75, 130, "Americas"), ("KR", 35000, 83, 52, "Asia")]:
    r.append(list(row))
write("scatter_bubble_2d", r,
      "Bubble chart of GDP per capita vs life expectancy sized by population")

# ── Alluvial ──────────────────────────────────────────────────────────────────
r = [["PatientID", "Baseline", "Week12", "Week24"]]
states = ["CR", "PR", "SD", "PD"]
for i in range(60):
    r.append([f"P{i+1}", random.choice(states), random.choice(states),
              random.choice(states)])
write("alluvial", r, "Show patient response state transitions from baseline to week 24")

# ── Binplot ───────────────────────────────────────────────────────────────────
r = [["Sample", "X", "Y", "Group"]]
for g in ["A", "B"]:
    for i in range(200):
        r.append([f"{g}_{i+1}",
                  round(random.gauss(ord(g) * 0.5, 2), 2),
                  round(random.gauss(ord(g) * 0.3, 2), 2),
                  g])
write("binplot", r, "Bin scatter of two numeric variables across two groups")

# ── Bubble ────────────────────────────────────────────────────────────────────
r = [["Sample", "X", "Y", "Size", "Group"]]
for g in ["Alpha", "Beta", "Gamma"]:
    for i in range(15):
        r.append([f"{g}_{i+1}",
                  round(random.uniform(1, 10), 1),
                  round(random.uniform(1, 10), 1),
                  round(random.uniform(5, 50), 1),
                  g])
write("bubble", r, "Bubble chart with x, y position and bubble size as third variable")

# ── Bullet ────────────────────────────────────────────────────────────────────
r = [["Metric", "Actual", "Target", "Min", "Max"]]
for metric in ["Revenue", "Margin", "NPS", "Retention", "Growth"]:
    t = random.randint(50, 80)
    r.append([metric, round(t * random.uniform(0.7, 1.2), 1), t, 0, 100])
write("bullet", r, "Bullet chart comparing actual performance against targets")

# ── Bump ──────────────────────────────────────────────────────────────────────
r = [["Drug", "Year", "Rank"]]
drugs = ["DrugA", "DrugB", "DrugC", "DrugD", "DrugE"]
for yr in range(2019, 2025):
    ranking = list(range(1, 6))
    random.shuffle(ranking)
    for drug, rank in zip(drugs, ranking):
        r.append([drug, str(yr), rank])
write("bump", r, "Show ranking changes of drugs over years")

# ── CDF ───────────────────────────────────────────────────────────────────────
r = [["Sample", "Value", "Group"]]
for g in ["GroupA", "GroupB"]:
    mean = 50 if g == "GroupA" else 65
    for i in range(100):
        r.append([f"{g}_{i+1}", round(random.gauss(mean, 10), 2), g])
write("cdf", r, "Cumulative distribution comparison between two groups")

# ── Cleveland ─────────────────────────────────────────────────────────────────
r = [["Country", "Value2020", "Value2024"]]
for country in ["US", "UK", "DE", "FR", "JP", "AU", "CA", "KR", "BR", "IN"]:
    v1 = random.randint(40, 80)
    r.append([country, v1, round(v1 * random.uniform(0.85, 1.2), 1)])
write("cleveland", r, "Cleveland dot plot comparing metric values in 2020 vs 2024 by country")

# ── Dumbbell ──────────────────────────────────────────────────────────────────
r = [["Region", "Before", "After"]]
for region in ["North", "South", "East", "West", "Central"]:
    b = random.randint(30, 60)
    r.append([region, b, round(b * random.uniform(0.9, 1.3), 1)])
write("dumbbell", r, "Before vs after comparison across regions as dumbbell chart")

# ── Pareto ────────────────────────────────────────────────────────────────────
r = [["DefectType", "Count"]]
for d in [("Wrong dose", 145), ("Mislabel", 98), ("Contamination", 72),
          ("Packaging", 55), ("Missing info", 40), ("Other A", 28),
          ("Other B", 20), ("Other C", 15), ("Other D", 12), ("Other E", 8)]:
    r.append(list(d))
write("pareto", r, "Pareto chart of defect types by frequency")

# ── QQ ────────────────────────────────────────────────────────────────────────
r = [["Sample", "Observed", "Group"]]
for g in ["Normal", "Heavy-tailed"]:
    for i in range(80):
        v = (random.gauss(0, 1) if g == "Normal"
             else random.gauss(0, 1) * random.gauss(0, 1))
        r.append([f"{g}_{i+1}", round(v, 3), g])
write("qq", r, "QQ plot to assess normality of observed residuals")

# ── Ribbon ────────────────────────────────────────────────────────────────────
r = [["Sample", "Start", "End", "Group"]]
for g in ["PathA", "PathB", "PathC"]:
    for i in range(10):
        s = round(random.uniform(0, 100), 1)
        r.append([f"{g}_{i+1}", s, round(s + random.uniform(5, 20), 1), g])
write("ribbon", r, "Ribbon plot of ranges per group")

# ── Spaghetti ─────────────────────────────────────────────────────────────────
r = [["PatientID", "Visit", "LabValue", "Treatment"]]
for arm in ["Drug", "Placebo"]:
    for pid in range(1, 16):
        base = random.gauss(50, 10)
        for visit in ["BL", "W4", "W8", "W12"]:
            r.append([f"{arm}_{pid}", visit,
                      round(base + random.gauss(-5 if arm == "Drug" else 0, 3), 1),
                      arm])
write("spaghetti", r, "Individual patient lab trajectories over visits by treatment arm")

# ── TimeSeries ────────────────────────────────────────────────────────────────
r = [["Datetime", "Value", "Sensor"]]
base_dt = datetime.datetime(2024, 1, 1)
for sensor in ["Sensor1", "Sensor2"]:
    for h in range(48):
        dt = base_dt + datetime.timedelta(hours=h)
        r.append([dt.strftime("%Y-%m-%d %H:%M"), round(random.gauss(20, 3), 1), sensor])
write("time_series", r, "Sensor readings over 48-hour period for two sensors")

# ── Tornado ───────────────────────────────────────────────────────────────────
r = [["Variable", "LowImpact", "HighImpact"]]
for v in [("Price sensitivity", -25, 40), ("Market size", -15, 30),
          ("Cost reduction", -10, 20), ("Competitor entry", -20, 5),
          ("Regulatory change", -30, 10), ("Technology adoption", -5, 35)]:
    r.append(list(v))
write("tornado", r, "Tornado sensitivity analysis of key business variables")

# ── TreeBracket ───────────────────────────────────────────────────────────────
r = [["Team", "Round", "Result"]]
teams = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
for t in teams:
    r.append([t, "Quarterfinal", random.choice(["Win", "Loss"])])
for t in ["Alpha", "Gamma", "Epsilon", "Theta"]:
    r.append([t, "Semifinal", random.choice(["Win", "Loss"])])
for t in ["Alpha", "Epsilon"]:
    r.append([t, "Final", random.choice(["Win", "Loss"])])
write("tree_bracket", r, "Tournament bracket results by round")

# ── Upset ─────────────────────────────────────────────────────────────────────
r = [["Member", "SetA", "SetB", "SetC", "SetD"]]
for i in range(50):
    r.append([f"G{i+1}"] + [random.choice([0, 1]) for _ in range(4)])
write("upset", r, "Upset plot showing intersections of four gene sets")


print("\nDone. Files in", OUT)
import glob
files = sorted(glob.glob(os.path.join(OUT, "*.json")))
print(f"{len(files)} files:")
for f in files:
    print(" ", os.path.basename(f))
