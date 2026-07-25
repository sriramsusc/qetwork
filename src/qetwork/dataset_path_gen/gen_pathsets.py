"""run: pipeline over one existing topology file — path sets -> error-stamped datasets.

    python run.py <topology.json> [--hop-range LO HI] [--n-datasets K]
                  [--split-seed S] [--error-seed S]

For any <stem>.json every output lands under a fresh <stem>/ next to this
script (OUT_ROOT, .../qetwork/dataset_path_gen/):
    raw_paths/       <name>_{prior,train,test}.csv
    error_datasets/  <name>_{prior,test}_datasets.csv, one
                     <name>_train_ds<k>_datasets.csv per DatasetID,
                     plus snapshots/<name>_ds<k>.json
The stage functions write <name>_-prefixed subdirectories; each one is renamed
to its fixed raw_paths/error_datasets name as soon as the stage finishes.
--hop-range is required for generator kinds missing from HOP_RANGE (only
"grid" is configured there)."""

import argparse
import shutil
import time
from pathlib import Path

from qetwork.dataset_path_gen.raw_paths import generate_datasets
from qetwork.dataset_path_gen.error_datasets import generate_error_datasets
from qetwork.topologies.topology_spec import TopologySpec

OUT_ROOT = Path(__file__).resolve().parent   # topology dirs are created here

N_DATASETS = 34
SPLIT_SEED = None              # train/test split; None -> DEFAULT_SEED
ERROR_SEED = None              # snapshot sampling; None -> DEFAULT_SEED


def _adopt(src: Path, dst: Path) -> Path:
    """Move a freshly written stage directory onto its fixed name; a stale
    directory from an earlier run of the same pipeline is replaced."""
    if dst.exists():
        shutil.rmtree(dst)
    return src.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate path sets and error datasets for one topology JSON.")
    ap.add_argument("topology", type=Path, help="existing topology JSON file")
    ap.add_argument("--hop-range", type=int, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="sampled hop counts, inclusive (required for kinds "
                         "not in HOP_RANGE)")
    ap.add_argument("--n-datasets", type=int, default=N_DATASETS,
                    help="error snapshots to sample")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--error-seed", type=int, default=ERROR_SEED)
    args = ap.parse_args()

    topo = args.topology.resolve()
    if topo.suffix != ".json" or not topo.is_file():
        raise SystemExit(f"{topo} is not an existing .json topology file")
    spec = TopologySpec.from_json(topo)          # fail fast before any mkdir
    run_dir = OUT_ROOT / topo.stem
    run_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    generate_datasets(topo, seed=args.split_seed, out_dir=run_dir,
                      hop_range=tuple(args.hop_range) if args.hop_range else None)
    raw_dir = _adopt(run_dir / f"{spec.name}_raw_paths", run_dir / "raw_paths")
    print(f"[1/2] path sets -> {raw_dir}  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    generate_error_datasets(topo, n_datasets=args.n_datasets,
                            seed=args.error_seed,
                            paths_dir=raw_dir, out_dir=run_dir)
    err_dir = _adopt(run_dir / f"{spec.name}_error_datasets",
                     run_dir / "error_datasets")
    print(f"[2/2] error datasets -> {err_dir}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
