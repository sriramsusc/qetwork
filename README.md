Clone the repo, make it working directory, install requirements(not much numpy and maybe matplot)

move the given CSV files into input_dir

Run the following:

uv run python -m qetwork.applications.run_nb --jobs 0 --purification --purification-rounds 2  --samples 40 --protocol seq

Once thats done, then

uv run python -m qetwork.applications.run_nb --jobs 0 --purification --purification-rounds 2  --samples 40 --protocol par

Once thats done, then

uv run python -m qetwork.applications.run_nb --jobs 0 --purification --purification-rounds 2  --samples 40 --protocol tad
