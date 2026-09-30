# Gravity-Adaptive Hierarchy

- Use `uv` for the Python environment and dependencies; keep generated files under this project in `D:\jev`.
- `uv run python train.py` trains NYC, Chicago, and San Francisco and then evaluates all three downstream tasks for each city. `uv run python test.py` evaluates existing outputs.
- The three datasets are downloaded from the pinned HAFusion source when missing and are ignored by Git. Do not commit `data_*` or `priors/` arrays. Training uses the fixed configuration in `src/gah/config.py` and writes only to `outputs/gah/`.
- Preserve the current model and evaluator numerical paths. Verify script changes with the three-city evaluation and run full CUDA training before claiming newly reproduced training results.

