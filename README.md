# IsaacLab Template Project

A lean template for building custom reinforcement‑learning environments and training scripts on top of NVIDIA IsaacLab. Dependency management is handled by **[uv](https://github.com/astral-sh/uv)** for reproducible lockfile installs, and a CUDA‑enabled Docker image is provided.

---

## Quick Start (local)

Sync dependencies:

```bash
uv sync
```

Install the project in editable mode:

```bash
uv pip install -e .
```

Launch a headless training run on the demo task:

```bash
uv run src/rsl_rl/train.py --task=Isaac-Cartpole-Custom-Direct-v0 --headless
```

---

## Docker Workflow

Build the image

```bash
sudo docker build -t isaac:test .
```

Open an interactive shell (optional)

```bash
sudo docker run -it --entrypoint bash --gpus all isaac:test
```

Test the training run

```bash
sudo docker run --rm --gpus all -v "$(pwd)":/src isaac:test src/rsl_rl/train.py --task=Isaac-Cartpole-Custom-Direct-v0 --headless
```

If you update a dependency (including IsaacLab) you need to run an upgrade before building the docker container.

```bash
uv lock --upgrade
sudo docker build -t isaac:test .
```


---

## Project Layout

```
.
├── Dockerfile               # GPU-enabled Isaac container
├── entrypoint.sh            # Container entry script
├── IsaacLab/                # Upstream IsaacLab source (submodule)
│   └── ...
├── pyproject.toml           # PEP 621 project metadata
├── src/
│   ├── custom_tasks/        # Your custom RL environments & task configs
│   ├── rlc.egg-info/
│   └── rsl_rl/              # RSL‑RL trainer
```
