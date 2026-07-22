"""error_datasets: snapshot a topology's error parameters and stamp path sets.

Each DatasetID is one network snapshot: every varying parameter is drawn once
per node / edge / detector from the physically-cited RANGES below, written out
twice:
  snapshots/<name>_ds<k>.json      the full spec with sampled values substituted
                                   (schema-valid: materialize it to simulate)
  <name>_<kind>_datasets.csv       one row per (DatasetID, path) for each of
                                   prior/train/test, carrying every varied
                                   parameter: per-node and per-edge |-joined
                                   lists in path order, detector values as
                                   dataset-level columns
Within a DatasetID every row reads the same snapshot, across all three files."""

import copy
import json
import math
from pathlib import Path

from qetwork.seed import make_rng
from qetwork.topologies.topology_spec import TopologySpec

N_DATASETS = 32

# (lo, hi, scale); "log" = log-uniform for decade-spanning quantities
RANGES = {
    # -- node: memory coherence, ps
    "t1":             (1e9, 1e14, "log"),   # 1 ms - 100 s: NV registers >1 s (Bradley et al., PRX 9, 031045 (2019));
                                            # Eu:YSO 6 h (Zhong et al., Nature 517, 177 (2015)); ion ~1 h (Wang et al., Nat. Commun. 12, 233 (2021))
    "t2_frac":        (0.1, 1.0, "log"),    # t2 = frac * t1; pure dephasing keeps T2 <= T1 (<= 2*T1 theoretical bound)
    # -- node: gate errors
    "p_depol_1q":     (1e-5, 1e-3, "log"),  # 1q error 1e-6 (ion, Harty et al., PRL 113, 220501 (2014)) to ~1e-3 (SC, Krantz et al., APR 6, 021318 (2019))
    "p_depol_2q":     (1e-4, 1e-2, "log"),  # 2q 99.9% ion (Ballance et al., PRL 117, 060504 (2016)); 99-99.7% SC
    "p_depol_swap":   (3e-4, 3e-2, "log"),  # SWAP = 3 CNOTs -> ~3x 2q error
    "coh1_angle":     (0.0, 0.05, "lin"),   # residual 1q calibration error, ~1% of pi/2 (Krantz et al. 2019)
    "coh2_zz_angle":  (0.0, 0.05, "lin"),   # residual ZZ crosstalk (Krantz et al. 2019)
    # -- node: SFWM source
    "src_visibility": (0.90, 0.995, "lin"), # SFWM two-photon visibility (Takesue & Inoue, PRA 70, 031802(R) (2004);
                                            # Signorini & Pavesi, AVS Quantum Sci. 2, 041701 (2020))
    "src_phase":      (0.0, 2 * math.pi, "lin"),
    # -- detectors (source node only)
    "coupling":       (0.8, 1.0, "lin"),    # fiber-to-detector coupling, <~1 dB
    "mzi_phase_error":(0.0, 0.15, "lin"),   # Kylia MINT ER >= 18 dB -> residual phase misset <~0.15 rad
    "mzi_arm_loss":   (0.0, 1.0, "lin"),    # dB per arm; Kylia MINT IL <= 2.0 dB total
    "bs_reflectivity":(0.45, 0.55, "lin"),  # +/-5% splitting tolerance over band
    "bs_loss":        (0.0, 0.3, "lin"),    # excess loss
    "snspd_efficiency": (0.80, 0.98, "lin"),# 93% (Marsili et al., Nat. Photon. 7, 210 (2013)) to 98% (Reddy et al., Optica 7, 1649 (2020))
    "snspd_jitter":   (10.0, 100.0, "lin"), # ps FWHM (You, Nanophotonics 9, 2673 (2020))
    "snspd_dark":     (1e-13, 1e-9, "log"), # per ps = 0.1 - 1000 Hz (You 2020)
    "snspd_dead":     (10_000.0, 100_000.0, "lin"),  # ps, 10 - 100 ns (You 2020)
    # -- edges
    "attenuation":    (0.16, 0.25, "lin"),  # dB/km @1550: Corning SMF-28 Ultra typ <=0.18; ITU-T G.652 bound
    "insertion_loss_db": (0.1, 1.0, "lin"), # connector/splice IL 0.1-0.5 dB per mated pair (IEC 61300-3-34), 1-2 per link
}

_NODE_FIELDS = ["t1", "t2", "p_depol_1q", "p_depol_2q", "p_depol_swap",
                "coh1_angle", "coh2_zz_angle", "src_visibility", "src_phase"]
_EDGE_FIELDS = ["length", "attenuation", "insertion_loss_db"]
_DET_FIELDS = ["coupling_1", "coupling_2", "mzi_phase_error", "mzi_loss_short",
               "mzi_loss_long", "bs1_reflectivity", "bs1_loss", "bs2_reflectivity",
               "bs2_loss", "snspd1_efficiency", "snspd1_jitter", "snspd1_dark",
               "snspd1_dead", "snspd2_efficiency", "snspd2_jitter", "snspd2_dark",
               "snspd2_dead"]


def _draw(rng, key: str) -> float:
    lo, hi, scale = RANGES[key]
    if scale == "log":
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def _sample_node(rng) -> dict[str, float]:
    t1 = _draw(rng, "t1")
    return {"t1": t1, "t2": t1 * _draw(rng, "t2_frac"),
            "p_depol_1q": _draw(rng, "p_depol_1q"),
            "p_depol_2q": _draw(rng, "p_depol_2q"),
            "p_depol_swap": _draw(rng, "p_depol_swap"),
            "coh1_angle": _draw(rng, "coh1_angle"),
            "coh2_zz_angle": _draw(rng, "coh2_zz_angle"),
            "src_visibility": _draw(rng, "src_visibility"),
            "src_phase": _draw(rng, "src_phase")}


def _sample_detector(rng) -> dict[str, float]:
    out = {"coupling_1": _draw(rng, "coupling"), "coupling_2": _draw(rng, "coupling"),
           "mzi_phase_error": _draw(rng, "mzi_phase_error"),
           "mzi_loss_short": _draw(rng, "mzi_arm_loss"),
           "mzi_loss_long": _draw(rng, "mzi_arm_loss")}
    for bs in ("bs1", "bs2"):
        out[f"{bs}_reflectivity"] = _draw(rng, "bs_reflectivity")
        out[f"{bs}_loss"] = _draw(rng, "bs_loss")
    for sn in ("snspd1", "snspd2"):
        out[f"{sn}_efficiency"] = _draw(rng, "snspd_efficiency")
        out[f"{sn}_jitter"] = _draw(rng, "snspd_jitter")
        out[f"{sn}_dark"] = _draw(rng, "snspd_dark")
        out[f"{sn}_dead"] = round(_draw(rng, "snspd_dead"))   # SNSPD wants int ps
    return out


def _sample_edge(rng) -> dict[str, float]:
    return {"attenuation": _draw(rng, "attenuation"),
            "insertion_loss_db": _draw(rng, "insertion_loss_db")}


def sample_snapshot(base: dict, rng) -> tuple[dict, dict, dict, dict]:
    """Return (spec_dict, node_vals, edge_vals, det_vals) for one DatasetID.

    spec_dict is the base spec with every sampled value substituted, so it can
    be written out and materialized; the *_vals dicts back the CSV columns."""
    spec = copy.deepcopy(base)
    node_vals, det_vals = {}, {}
    for nid, node in spec["nodes"].items():
        v = _sample_node(rng)
        node_vals[nid] = v
        node["t1"], node["t2"] = v["t1"], v["t2"]
        node["gates"]["p_depol_1q"] = v["p_depol_1q"]
        node["gates"]["p_depol_2q"] = v["p_depol_2q"]
        node["gates"]["p_depol_swap"] = v["p_depol_swap"]
        node["gates"]["coherent_1q"]["angle"] = v["coh1_angle"]
        node["gates"]["coherent_2q"]["zz_angle"] = v["coh2_zz_angle"]
        node["source"]["visibility"] = v["src_visibility"]
        node["source"]["phase"] = v["src_phase"]
        for dname in sorted(node["detectors"]):
            det, d = node["detectors"][dname], _sample_detector(rng)
            det_vals[dname] = d
            det["coupling_1"], det["coupling_2"] = d["coupling_1"], d["coupling_2"]
            det["mzi"]["phase_error"] = d["mzi_phase_error"]
            det["mzi"]["loss_short"] = d["mzi_loss_short"]
            det["mzi"]["loss_long"] = d["mzi_loss_long"]
            for bs in ("bs1", "bs2"):
                det["mzi"][bs]["reflectivity"] = d[f"{bs}_reflectivity"]
                det["mzi"][bs]["loss"] = d[f"{bs}_loss"]
            for i, sn in (("1", "snspd1"), ("2", "snspd2")):
                det[f"snspd_{i}"]["efficiency"] = d[f"{sn}_efficiency"]
                det[f"snspd_{i}"]["jitter_fwhm"] = d[f"{sn}_jitter"]
                det[f"snspd_{i}"]["dark_count_rate"] = d[f"{sn}_dark"]
                det[f"snspd_{i}"]["dead_time"] = d[f"{sn}_dead"]
    edge_vals = {}
    for ename, e in spec["edges"].items():
        v = _sample_edge(rng)
        edge_vals[ename] = {"length": e["length"], **v}
        e["attenuation"] = v["attenuation"]
        e["insertion_loss_db"] = v["insertion_loss_db"]
    return spec, node_vals, edge_vals, det_vals


def _read_paths_csv(path) -> list[tuple[str, str]]:
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0] != "PathID,PathString":
        raise ValueError(f"{path} is not a PathID,PathString CSV")
    return [tuple(line.split(",", 1)) for line in lines[1:]]


def _g(x: float) -> str:
    return format(x, ".6g")


def generate_error_datasets(topology, *, n_datasets: int = N_DATASETS,
                            seed: int | None = None, paths_dir=None,
                            out_dir=None) -> dict[str, Path]:
    """Sample n_datasets snapshots; stamp prior/train/test path sets with them.

    Reads <name>_{prior,train,test}.csv from paths_dir (default: the topology
    file's directory), writes snapshots/<name>_ds<k>.json and
    <name>_<kind>_datasets.csv into out_dir (default: paths_dir)."""
    if n_datasets < 1:
        raise ValueError(f"n_datasets must be positive, got {n_datasets}")
    spec = TopologySpec.from_json(topology) if not isinstance(topology, TopologySpec) else topology
    base = spec.to_dict()
    if paths_dir is None:
        paths_dir = Path.cwd() if isinstance(topology, TopologySpec) \
            else Path(topology).resolve().parent
    paths_dir = Path(paths_dir)
    out_dir = paths_dir if out_dir is None else Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"out_dir {out_dir} is not an existing directory")
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)

    edge_by_pair = {frozenset((e["u"], e["v"])): ename
                    for ename, e in spec.edges.items()}
    det_names = sorted(dn for n in spec.nodes.values() for dn in n["detectors"])
    det_cols = [f"{dn}_{f}" for dn in det_names for f in _DET_FIELDS]
    header = ("DatasetID,PathID,PathString,"
              + ",".join(f"node_{f}" for f in _NODE_FIELDS) + ","
              + ",".join(f"edge_{f}" for f in _EDGE_FIELDS)
              + ("," + ",".join(det_cols) if det_cols else ""))

    rng = make_rng(seed)
    snapshots = []
    for d in range(n_datasets):
        snap, node_vals, edge_vals, det_vals = sample_snapshot(base, rng)
        snap["name"] = f"{spec.name}-ds{d}"
        TopologySpec(copy.deepcopy(snap))            # validate before writing
        with open(snap_dir / f"{spec.name}_ds{d}.json", "w") as f:
            json.dump(snap, f, indent=2, allow_nan=False)
        snapshots.append((node_vals, edge_vals, det_vals))

    outs = {}
    for kind in ("prior", "train", "test"):
        rows = _read_paths_csv(paths_dir / f"{spec.name}_{kind}.csv")
        out = out_dir / f"{spec.name}_{kind}_datasets.csv"
        with open(out, "w") as f:
            f.write(header + "\n")
            for d, (node_vals, edge_vals, det_vals) in enumerate(snapshots):
                det_part = "," + ",".join(
                    _g(det_vals[dn][fld]) for dn in det_names for fld in _DET_FIELDS
                ) if det_names else ""
                for pid, pstr in rows:
                    nodes = pstr.split("->")
                    hops = [edge_by_pair[frozenset(h)] for h in zip(nodes, nodes[1:])]
                    node_part = ",".join(
                        "|".join(_g(node_vals[n][fld]) for n in nodes)
                        for fld in _NODE_FIELDS)
                    edge_part = ",".join(
                        "|".join(_g(edge_vals[e][fld]) for e in hops)
                        for fld in _EDGE_FIELDS)
                    f.write(f"{d},{pid},{pstr},{node_part},{edge_part}{det_part}\n")
        outs[kind] = out
        print(f"{kind}: {len(rows)} paths x {n_datasets} datasets -> {out}")
    print(f"snapshots: {n_datasets} spec files in {snap_dir}")
    return outs


if __name__ == "__main__":
    here = Path(__file__).resolve().parent           # .../topologies/sim_path_gen
    generate_error_datasets(here.parent / "grid10x10.json")
