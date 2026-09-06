# Installation Matrix

**Not yet published to PyPI.** Clone the repository and run the commands below
from its root. A bare package-name install is not currently available.

```bash
git clone https://github.com/LazzyXP/Multimodal-Recommender-System.git
cd Multimodal-Recommender-System
```

The project root `uv.toml` uses the Aliyun PyPI mirror by default. The package
follows the same separation used by AutoGluon: the recommender wheel is
platform-neutral, and the optional deep-learning runtime is selected at install time.
CUDA is not bundled into `multimodal-recommender` and should not be installed on a Mac.

## macOS

Apple Silicon and Intel Macs use the CPU/MPS PyTorch wheel. PyTorch selects MPS on
Apple Silicon when it is available:

```bash
python -m pip install ".[torch]"
```

To force CPU inference or training, pass `model_configs={"LightGCN": {"device": "cpu"}}`
to `fit()`. Use `device="mps"` to require Apple Metal acceleration.

## Linux or Windows CPU

For a CPU-only deployment, use the PyTorch CPU index as an additional index:

```bash
python -m pip install ".[torch]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Linux or Windows NVIDIA CUDA

The CUDA runtime is provided by the PyTorch wheel. The host still needs a compatible
NVIDIA driver; installing the full CUDA toolkit is optional for this package. The
package constrains torch below 2.7 because newer wheels may require CUDA 13. For a
CUDA 12.4 host, use the matching `cu124` channel:

```bash
uv pip install torch \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple
python -m pip install .
```

Then leave `device="auto"` (the default), or set
`model_configs={"LightGCN": {"device": "cuda"}}` explicitly. Do not use a CUDA
index on macOS; it cannot enable CUDA without NVIDIA hardware.

## Verification

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from mmrec import MultiModalRecommender; print(MultiModalRecommender().available_models())"
```

The graph models are listed in the catalog even when Torch is absent, but automatic
selection skips them until the optional runtime is installed. This keeps the core
CSV/Parquet conversion and NumPy models installable on every supported platform.
