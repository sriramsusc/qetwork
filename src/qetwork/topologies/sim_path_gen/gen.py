from pathlib import Path
from qetwork.topologies import topology_generator
from qetwork.topologies.topology_generator import generate

topo = Path(topology_generator.__file__).resolve().parent   # .../qetwork/topologies

generate("grid", rows=10, cols=10, seed=7,
         out_path=str(topo / "grid10x10.json"),
         qlink_length_range=(10.0, 100.0))
