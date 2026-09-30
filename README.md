# Gravity-Adaptive Hierarchy

Train and test on NYC, Chicago, and San Francisco. Requires Python 3.10–3.12, `uv`, and an NVIDIA CUDA GPU. The three datasets are downloaded from the [HAFusion source archive](https://github.com/MiRuacle24/HAFusion/archive/73fbe911901ada71bb40129e19d86dedb9b9602c.zip) on first run; they are not stored in this repository.

```bash
uv run python train.py
```

The command trains all three cities and reports the three downstream tasks for each. Outputs are saved under `outputs/gah/`. To test existing outputs again, run `uv run python test.py`.

Reference: [HAFusion code and data](https://github.com/MiRuacle24/HAFusion) · [HAFusion paper](https://arxiv.org/abs/2312.04606). See [third-party provenance](THIRD_PARTY.md).

