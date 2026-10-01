Clone the repo, make it working directory, install requirements(not much numpy and maybe matplot)

move the given CSV files into input_dir (input_dir should have only a bunch of csv files and a snapshot subdirectory, no other directopries)

Run the following:

uv run python -m qetwork.applications.run_nb --jobs 0 --purification --purification-rounds 2  --samples 40 --protocol seq

Once thats done, then

uv run python -m qetwork.applications.run_nb --jobs 0 --purification --purification-rounds 2  --samples 40 --protocol par

Once thats done, then

uv run python -m qetwork.applications.run_nb --jobs 0 --purification --purification-rounds 2  --samples 40 --protocol tad

## Petri net model

`qetwork.petri` models entanglement generation + swapping (`EntanglementDistribution`, all three
modes) as a timed coloured Petri net and cross-checks it against the simulator. Design, net
structure and diagram: [docs/petri_net.md](docs/petri_net.md).

uv run python -m qetwork.petri.validate                          # net vs simulator, synthetic chain

uv run python -m qetwork.petri.validate --csv src/qetwork/input_dir/tester.csv --row 0

uv run pytest tests/test_petri.py
