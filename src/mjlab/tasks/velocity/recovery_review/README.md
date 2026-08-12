# LaFAN recovery review

This folder contains the complete recovery-review workflow. Runtime artifacts
are also kept together under `artifacts/recovery/review/` by default:

```text
artifacts/recovery/review/
├── lafan_manifest.json
├── review_queue.json
├── reviewer-01.json
├── metrics.validation.json
├── metrics.test.json
└── lafan_manifest.reviewed.json
```

## Resume the checked-in review

The `academic` branch contains the exact review queue, the current
`reviewer-01.json`, and the minimal 17-file LaFAN1 subset referenced by all
segment/validation/test items. From a fresh clone:

```bash
git clone --branch academic https://github.com/DearSean/mjlab-copy.git
cd mjlab-copy
uv sync --extra cpu --group dev
uv run python -m mjlab.tasks.velocity.recovery_review.viewer \
  --reviewer reviewer-01 \
  --mode segments
```

The default dataset root is `data/lafan1`. Review decisions are saved atomically
to `artifacts/recovery/review/reviewer-01.json`. Commit and push that file before
moving to another computer. The bundled BVH files retain their separate LaFAN1
CC BY-NC-ND 4.0 license; see `data/lafan1/README.md` and `LICENSE.txt`.

## 1. Prepare a new queue (full dataset only)

```bash
uv run python -m mjlab.tasks.velocity.recovery_review.prepare \
  --dataset-root /path/to/full/lafan1
```

Do not run this command merely to resume the checked-in queue. Rebuilding a new
queue requires all 77 original LaFAN1 files and changes the queue hash to which
the human review state is bound.

The high-recall detector accepts three useful terminal modes:

- `stationary`: recovery ends in a stable upright posture.
- `locomotion`: support transfer is followed directly by walking or running.
- `support_only`: the support phase is useful, but a complete recovery cannot
  yet be established automatically.

## 2. Review all proposed segments

```bash
uv run python -m mjlab.tasks.velocity.recovery_review.viewer \
  --reviewer reviewer-01 \
  --mode segments
```

Accept a segment when it contains a real low/fallen state, a purposeful support
transfer, and a safe terminal state. A locomotion terminal does not need to stop
or become fully erect. Mark support start, support completion, and locomotion
takeover when they are visible. Reject ordinary crouches, transient upright
passes, clipped attempts, and attempts that immediately collapse.

The Playback panel uses original BVH global frame numbers and provides
`Previous frame` and `Next frame` buttons for exact stepping. Use the adjacent
`Set ... = current global frame` buttons. The end-frame button writes the
current global frame plus one because intervals are half-open:
`[start_frame, end_frame)`. Accepted or rejected reviews are locked; click the
orange `Modify` button before changing them. Gray `AUTO` fields are automatic
defaults and green `SET` fields were explicitly selected by the reviewer.

## 3. Blind frame-label audit

Tune thresholds only with the validation partition:

```bash
uv run python -m mjlab.tasks.velocity.recovery_review.viewer \
  --reviewer reviewer-01 \
  --mode validation
```

Automatic labels are hidden in this mode. Mark a contact when the site is at the
floor without visible penetration and is stationary over adjacent frames or
clearly bears support. Use `uncertain` instead of forcing an ambiguous label.
The automatic label uses floor clearance, signed vertical velocity, and temporal
hysteresis. Foot contact uses a `0.035H` core band plus a speed-limited `0.06H`
ambiguity band and roughly 50 ms confirmation. Horizontal sliding or pivoting
never invalidates a site already inside the core band; the wider band rejects a
fast airborne swing. The 93D velocity channels retain motion information.

After thresholds are frozen, unlock the test partition explicitly:

```bash
uv run python -m mjlab.tasks.velocity.recovery_review.viewer \
  --reviewer reviewer-01 \
  --mode test \
  --allow-test
```

## 4. Evaluate and compile

```bash
uv run python -m mjlab.tasks.velocity.recovery_review.evaluate \
  --review-file artifacts/recovery/review/reviewer-01.json \
  --partition validation

uv run python -m mjlab.tasks.velocity.recovery_review.compile \
  --review-file artifacts/recovery/review/reviewer-01.json
```

Compilation refuses a mismatched queue/manifest and, by default, any pending
segment. It never changes the source BVH or automatic manifest. Only accepted
segments are written to `lafan_manifest.reviewed.json`.
