# AGENTS.md

This file contains everything an AI coding agent needs to know about mjlab.

---

## Project Overview

mjlab is a GPU-accelerated reinforcement learning and robotics research framework.
It combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based
environment API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
a GPU-accelerated backend for [MuJoCo](https://github.com/google-deepmind/mujoco).

The framework provides composable building blocks for environment design: entities
(robots, objects, terrain), actuators, sensors, and a suite of managers that
define the RL problem (observations, actions, rewards, terminations, domain
randomization, commands, curriculum, metrics).

Key facts:

- **Language**: Python 3.10–3.13
- **License**: Apache-2.0
- **GPU requirement**: NVIDIA GPU for training. macOS is supported for evaluation only.
- **Package name**: `mjlab` (version 1.3.0)
- **Repository**: https://github.com/mujocolab/mjlab
- **Docs**: https://mujocolab.github.io/mjlab/

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Physics | MuJoCo >= 3.8.0, MuJoCo Warp >= 3.8.0 |
| GPU compute | NVIDIA Warp >= 1.12.0 |
| Deep learning | PyTorch >= 2.7.0 |
| RL algorithms | rsl-rl-lib == 5.2.0 |
| Package manager | uv (with `uv.lock` committed) |
| Lint / format | Ruff 0.14.14 |
| Type checking | `ty` (fast) and `pyright` (thorough) |
| Testing | pytest |
| Documentation | Sphinx with sphinx-book-theme, sphinx-multiversion |
| CLI parsing | tyro |
| Logging / viz | Weights & Biases, TensorBoard, viser, mjviser |

MuJoCo Warp is fetched from a specific git revision (not PyPI). See
`pyproject.toml` under `[tool.uv.sources]`.

---

## Architecture

mjlab has a two-layer architecture:

### Simulation Layer

1. **Scene** composes entity MJCF files into a single `MjSpec`.
2. The spec is compiled into an `MjModel` on the CPU.
3. `Simulation` uploads the model to the GPU via MuJoCo Warp, allocating a single
   `MjData` with `N` parallel worlds.
4. CUDA graphs are captured for `step`, `forward`, `reset`, and `sense` to
   eliminate CPU dispatch overhead.

Core simulation modules:

- `mjlab.scene` — scene composition and entity placement
- `mjlab.entity` — robots, objects, terrain entities
- `mjlab.actuator` — actuator implementations (PD, DC motor, learned MLP, XML, builtin)
- `mjlab.sensor` — sensors (contact, camera, raycast, terrain height, builtin)
- `mjlab.sim` — MuJoCo Warp bridge and simulation data (`Simulation`, `SimulationCfg`)
- `mjlab.terrains` — procedural terrain generators
- `mjlab.asset_zoo` — bundled robot assets (Unitree G1, RL_BOY, etc.)

### Manager Layer

On top of the simulation layer, environments are defined by composing small
*terms* (functions or classes) and registering them with managers.

The core environment class is `ManagerBasedRlEnv` (`mjlab.envs`). It is configured
via `ManagerBasedRlEnvCfg`, a dataclass that holds term dictionaries for each
manager.

The eight managers (in `mjlab.managers`):

1. **ObservationManager** — observation groups with noise, delay, history
2. **ActionManager** — routes policy output to entity actuators
3. **RewardManager** — weighted sum of reward terms, scaled by step dt
4. **TerminationManager** — stop conditions (terminal vs. timeout)
5. **EventManager** — domain randomization and resets at lifecycle points
6. **CommandManager** — goal signals (velocity targets, pose targets)
7. **CurriculumManager** — adaptive difficulty
8. **MetricsManager** — per-step values logged as episode averages

There is also a **RecorderManager** for logging observations/actions during rollouts.

### Tasks and Registry

Tasks (environments) are defined under `mjlab.tasks/` (e.g., `velocity/`,
`tracking/`, `manipulation/`, `cartpole/`). Each task registers itself via
`mjlab.tasks.registry.register_mjlab_task()` with a unique task ID.

Task packages can also be auto-discovered via the `mjlab.tasks` entry point group.

---

## Code Organization

```
src/mjlab/
  __init__.py              # Package init: configures Warp, mediapy, imports registered tasks
  actuator/                # Actuator implementations
  asset_zoo/               # Bundled robot MJCFs and assets
  entity/                  # Entity definitions, data, variants
  envs/
    manager_based_rl_env.py  # Core vectorized RL environment
    mdp/                     # Built-in MDP terms (actions, observations, rewards, terminations, events, curriculums, metrics)
  managers/                # All manager implementations
  rl/                      # RSL-RL integration (runner, config, vecenv wrapper, exporter)
  scene/                   # Scene composition and configuration
  sensor/                  # Sensor implementations
  sim/                     # Simulation bridge to MuJoCo Warp
  tasks/                   # Task definitions
    velocity/              # Velocity tracking for legged robots
    tracking/              # Motion imitation
    manipulation/          # Manipulation tasks
    cartpole/              # Cartpole baseline
    registry.py            # Central task registry
  terrains/                # Terrain generators and configuration
  utils/                   # Utilities (spaces, torch helpers, XML, logging, buffers, noise, etc.)
    lab_api/               # Forked from Isaac Lab (BSD-3-Clause, excluded from lint/type checks)
  viewer/                  # Visualization: native MuJoCo viewer, Viser, offscreen renderer
  scripts/                 # CLI entry points
    train.py               # `train` command
    play.py                # `play` command
    demo.py                # `demo` command
    list_envs.py           # `list-envs` command
    nan_viz.py             # `viz-nan` command
    export_scene.py        # `export-scene` command
```

---

## Build, Development, and Test Commands

**Always use `uv run`, not `python`.**

### Environment setup

```sh
# GPU (default)
uv sync --all-packages --extra cu128 --group dev

# CPU-only
uv sync --all-packages --extra cpu --group dev
```

### Common commands

```sh
make format         # Format and lint with Ruff
make type           # Type-check with ty and pyright
make check          # make format && make type (run before every commit)
make test-fast      # Run tests excluding slow ones
make test           # Run the full test suite
make test-cpu       # Force CPU backend
make test-cpu-fast  # Force CPU backend, skip slow tests
make docs           # Build Sphinx documentation
make docs-watch     # Build docs with auto-reload
make build          # Build wheel + sdist and smoke-test
make publish-test   # Publish to TestPyPI
make publish        # Publish to PyPI
make docker-build   # Build Docker image
```

### Manual equivalents

```sh
# Format / lint
uv run ruff format
uv run ruff check --fix

# Type check
uv run ty check        # Fast
uv run pyright         # More thorough, but slower

# Test
uv run pytest tests/                    # Full suite
uv run pytest tests/<test_file>.py      # Single file
uv run pytest -m "not slow"             # Skip slow tests
FORCE_CPU=1 uv run pytest               # CPU-only

# Docs
uv run --group docs sphinx-build -j auto docs docs/_build
uv run --group docs sphinx-autobuild -j auto docs docs/_build
```

### Pre-commit

```sh
uvx pre-commit install
```

---

## Code Style Guidelines

- **Line length limit**: 88 columns (code, comments, docstrings).
- **Indent**: 2 spaces (Ruff `indent-width = 2`).
- **Type checking target**: Python 3.10 syntax.
- **Avoid local imports** unless strictly necessary (e.g., circular imports).
- **First-party imports**: `src = ["src"]` in Ruff config; imports look like `from mjlab.xxx import ...`.
- **Ruff lint rules**: `E4`, `E7`, `E9`, `F`, `I`, `B`; ignore `B011`.
- **Excluded from lint/type**: `src/mjlab/utils/lab_api/` and `typings/`.

### Dataclass configs

Most configuration is done via `dataclass(kw_only=True)`. Config classes end in
`Cfg` (e.g., `SimulationCfg`, `SceneCfg`). The project uses `tyro` to turn these
into CLI arguments.

### Tyro flags (global)

The package sets three global tyro flags in `mjlab/__init__.py`:

- `AvoidSubcommands` — simpler CLI, no union type switching
- `FlagConversionOff` — use `--flag False` instead of `--no-flag`
- `UsePythonSyntaxForLiteralCollections` — e.g., `--tuple (1,2,3)`

---

## Testing Instructions

- Tests live in `tests/` and use **functions and fixtures**, not test classes.
- Favor **targeted, efficient tests** over exhaustive edge-case coverage.
- Prefer running **individual tests** rather than the full suite during iteration.
- Slow tests are marked with `@pytest.mark.slow` and skipped by `make test-fast`.
- `tests/conftest.py` contains shared fixtures, including `configure_test_environment`
  which sets `wp.config.quiet = True`.
- `tests/fixtures/` contains small MJCF files for unit tests.
- The `get_test_device()` helper prefers CUDA but can be overridden with `FORCE_CPU=1`.

---

## Runtime and Environment Variables

| Variable | Effect |
|----------|--------|
| `MJLAB_WARP_QUIET=1` | Suppress Warp kernel compilation progress output |
| `FORCE_CPU=1` | Force CPU backend for tests |
| `MUJOCO_GL=egl` | Headless rendering (set in Docker) |
| `CUDA_VISIBLE_DEVICES` | Controls GPU visibility for multi-GPU training |

---

## CLI Entry Points

The package installs these console scripts:

- `train <task-id> [options]` — Train an RL agent with RSL-RL
- `play <task-id> [options]` — Evaluate a trained policy
- `demo` — Run a quick demo
- `list-envs` — List all registered task IDs
- `viz-nan` — Visualize NaN occurrences
- `export-scene` — Export a scene

Examples:

```sh
uv run train Mjlab-Velocity-Flat --env.scene.num-envs 4096
uv run play Mjlab-Velocity-Flat --wandb-run-path your-org/mjlab/run-id
uv run play Mjlab-Your-Task-Id --agent zero
```

Multi-GPU training uses `--gpu-ids "[0, 1]"` and is orchestrated via `torchrunx`.

---

## CI / CD

GitHub Actions workflows (in `.github/workflows/`):

- `ci.yml` — Lint/format, tests (Python 3.10–3.13), pyright, ty check. Runs on
  CPU (`--extra cpu`). Caches Warp kernel compilation under `~/.cache/warp`.
- `docs.yml` — Builds Sphinx multi-version docs and deploys to GitHub Pages.
- `docker.yml` — Builds and publishes Docker image.
- `release.yml` — Triggered on version tags.

---

## Release Process

See `RELEASING.md` for full details. Summary:

1. Bump version in `pyproject.toml`.
2. Update `CITATION.cff`.
3. Update `docs/source/changelog.rst`.
4. `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`
5. `make build` to verify wheel + sdist.
6. `make publish-test` (optional).
7. `UV_PUBLISH_TOKEN=<token> make publish`.
8. Verify with `uvx --refresh --from mjlab demo`.

---

## Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in the title.
- PR body should be plain, concise prose. No section headers, checklists, or
  structured templates. Describe the problem, what the change does, and any
  non-obvious tradeoffs.
- PR and commit messages are rendered on GitHub — do not hard-wrap at 88 columns.
  Let each sentence flow on one line.
- When making user-facing changes, add an entry to `docs/source/changelog.rst`
  under the "Upcoming version (not yet released)" section using Added/Changed/Fixed.
  Reference issues with `:issue:\`123\``.

---

## Documentation

- Source files are in `docs/source/` (reStructuredText and Markdown via MyST).
- `docs/conf.py` configures Sphinx with autodoc, multi-version support, and custom
  docstring processing for dataclasses.
- Build with `make docs` or `make docs-watch`.
- The `architecture_overview.rst` page is the best starting point for understanding
  the system design.

---

## Security Considerations

- The project runs GPU kernels via NVIDIA Warp and MuJoCo Warp. These are compiled
  at runtime; do not run untrusted environment definitions on shared machines.
- `wp.config.enable_backward = False` is set globally in `mjlab/__init__.py`.
- Docker image runs as root and exposes port 8080; use standard container security
  practices in production.

---

## Key Files for Quick Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, tool configs (ruff, pyright, ty, pytest) |
| `uv.lock` | Locked dependency tree — commit changes |
| `Makefile` | Common dev commands |
| `src/mjlab/envs/manager_based_rl_env.py` | Core environment class |
| `src/mjlab/tasks/registry.py` | Task registration system |
| `docs/source/architecture_overview.rst` | Architecture documentation |
| `tests/conftest.py` | Shared test fixtures |
