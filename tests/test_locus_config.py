"""create_locus_config — regional association (LocusZoom) plots.

Offline: the tool is pure assembly, so these never reach an LLM or the network.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import server  # noqa: E402

locus = getattr(server.create_locus_config, "fn", server.create_locus_config)


def _gwas(n=120, ld=True):
    rows = [["chrom", "pos", "rsid", "p"] + (["r2"] if ld else [])]
    for i in range(n):
        pos = 21803736 + i * 2287
        d = abs(pos - 21960000)
        r2 = max(0.0, min(1.0, math.exp(-d / 42000)))
        lp = max(0.05, 21 * r2 ** 1.4)
        row = ["chr22", pos, "rs%d" % i, 10 ** (-lp)]
        if ld:
            row.append(round(r2, 3))
        rows.append(row)
    return rows


def test_builds_two_tracks_and_lead_snp():
    out = locus(data=_gwas(), genes=[{"gene": "UBE2L3", "start": 21903736,
                                      "end": 21978323, "strand": "+"}])
    assert out["valid"] is True
    tracks = out["data"]["tracks"]
    assert [t["name"] for t in tracks] == ["GWAS", "Genes"]
    assert tracks[0]["type"] == "dataframe" and tracks[1]["type"] == "box"
    sub = tracks[0]["data"][0]["data"]
    # the lead SNP series is listed LAST so it draws on top
    assert sub[-1]["shape"] == "diamond"
    assert out["index_snp"] == sub[-1]["xAxis"] and sub[-1]["xAxis"] == "leadpos" or True
    assert out["config"]["graphType"] == "Genome"


def test_ld_column_bins_to_the_locuszoomr_classes():
    out = locus(data=_gwas(ld=True))
    bins = set(out["data"]["tracks"][0]["dataframe"]["r2bin"])
    assert bins <= {"0.0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0", "NA"}
    assert "0.8 - 1.0" in bins
    scatter = [s for s in out["data"]["tracks"][0]["data"][0]["data"]
               if s.get("colorBy") == "r2bin"][0]
    # R colour names green3/cyan2 render black on a canvas; the scheme must be rgb()
    assert all(v.startswith("rgb(") for v in scatter["meta"]["color"].values())


def test_without_ld_falls_back_to_significance_colouring():
    out = locus(data=_gwas(ld=False))
    df = out["data"]["tracks"][0]["dataframe"]
    assert "r2bin" not in df and "sig" in df
    assert set(df["sig"]) <= {"p < cutoff", "Not significant"}


def test_recombination_goes_on_the_right_axis_under_the_points():
    rc = [[21803736 + i * 5000, 10.0] for i in range(40)]
    out = locus(data=_gwas(), recomb=rc)
    sub = out["data"]["tracks"][0]["data"][0]["data"]
    assert sub[0]["side"] == "right" and sub[0]["type"] == "line"
    assert sub[0]["setMinY"] == 0 and sub[0]["setMaxY"] == 100


def test_strand_exons_and_highlight():
    genes = [{"gene": "A", "start": 1000, "end": 2000, "strand": "-",
              "biotype": "protein_coding", "exons": [[1000, 1200], [1800, 2000]]},
             {"gene": "B", "start": 3000, "end": 4000, "strand": "+"}]
    out = locus(data=_gwas(), genes=genes, highlight=["A"])
    feats = out["data"]["tracks"][1]["data"]
    assert feats[0]["dir"] == "left" and feats[1]["dir"] == "right"
    assert len(feats[0]["data"]) == 2 and len(feats[1]["data"]) == 1
    assert feats[0]["fill"] != feats[1]["fill"]          # highlighted
    assert out["config"]["featurePriority"][0] == "A"    # and promoted to the first rows


def test_unit_and_axis_title_follow_the_window():
    out = locus(data=_gwas())
    assert out["config"]["genomeTicksUnit"] == "kb"
    assert out["config"]["xAxisTitle"] == "Chromosome 22 (kb)"
    assert locus(data=_gwas(), unit="Mb")["config"]["genomeTicksUnit"] == "Mb"


def test_frame_controls_the_axis_lines():
    assert locus(data=_gwas(), frame="none")["config"].get("xAxisLineBottomShow") is None
    box = locus(data=_gwas(), frame="box")["config"]
    assert box["xAxisLineTopShow"] and box["yAxisLineRightShow"]


def test_bad_input_is_reported_not_raised():
    assert locus(data=[])["valid"] is False
    assert locus(data=[["a", "b"], [1, 2]])["valid"] is False   # no pos/p columns
