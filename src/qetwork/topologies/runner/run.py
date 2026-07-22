"""run: full pipeline — topology -> path sets -> error-stamped datasets.

Stage 1 writes grid10x10.json into qetwork/topologies/, stage 2 writes
<name>_raw_paths/<name>_{prior,train,test}.csv next to it, stage 3 reads
those and writes <name>_error_datasets/ next to it: <name>_{prior,test}_datasets.csv,
one <name>_train_ds<k>_datasets.csv per DatasetID, plus snapshots/<name>_ds<k>.json."""

import time
from pathlib import Path

from qetwork.topologies.topology_generator import generate
from qetwork.topologies.sim_path_gen.path_gen_from_topo import generate_datasets
from qetwork.topologies.sim_path_gen.error_datasets import generate_error_datasets

TOPOLOGIES = Path(__file__).resolve().parent   # .../qetwork/topologies

ROWS, COLS = 10, 10
TOPO_SEED = 7                  # resolves lengths at spec time (roles are pinned below)
ROLES = ("n1", f"n{COLS}")     # top-left corner -> top-right corner
QLINK_RANGE = (10.0, 100.0)
SPLIT_SEED = None              # train/test split; None -> DEFAULT_SEED
N_DATASETS = 15
ERROR_SEED = None              # snapshot sampling; None -> DEFAULT_SEED

def main() -> None:
    topo = TOPOLOGIES / f"grid{ROWS}x{COLS}.json"

    t0 = time.perf_counter()
    generate("grid", rows=ROWS, cols=COLS, seed=TOPO_SEED,
             out_path=str(topo), qlink_length_range=QLINK_RANGE, roles=ROLES)
    print(f"[1/3] topology -> {topo}  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    generate_datasets(topo, seed=SPLIT_SEED)
    print(f"[2/3] path sets done  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    generate_error_datasets(topo, n_datasets=N_DATASETS, seed=ERROR_SEED)
    print(f"[3/3] error datasets done  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
