# Experiment Layout

Each scientific experiment lives in its own directory under `experiments/`.
Do not reuse a prior experiment directory for a materially different model,
augmentation policy, loss, data split, or benchmark protocol.

```text
experiments/<experiment-id>/
  README.md                 # hypothesis and acceptance criteria
  configs/                  # immutable arm configurations
  train.slurm               # experiment-specific scheduler entrypoint
  src/                      # optional experiment-only replacement modules
```

Shared implementations stay in `scripts/` (data loading, common models,
masking, benchmark orchestration). An experiment configuration names the shared
components it uses. If a hypothesis requires a new implementation, add it under
that experiment's `src/` directory and document the divergence; do not mutate a
shared component in a way that changes an already-defined experiment.

Run artifacts are isolated by experiment and arm:

```text
data/checkpoints/experiments/<experiment-id>/<arm>/
models/experiments/<experiment-id>/<arm>/
```

Every checkpoint records its resolved configuration. Benchmark outputs always
belong beside the exact checkpoint that produced them.

## Creating An Experiment

1. Pick a descriptive, versioned ID such as `span_masking_v1` or
   `larger_model_d512_v1`; never overwrite an existing ID.
2. Create `experiments/<id>/README.md` stating one hypothesis, the invariant
   protocol, the changed factor(s), metrics, and stop criteria.
3. Put one JSON config per arm in `configs/`. Hold split, validation, benchmark,
   seed policy, and all unrelated hyperparameters constant across comparison
   arms. The project-standard output heads are masked reconstruction, redshift,
   two-class DESI `SPECTYPE` classification (`GALAXY=0`, `QSO=1`), and four
   multi-label target-selection flags (`BGS`, `LRG`, `ELG`, `QSO`). Keep those
   objectives and their weights fixed when comparing methods unless changing
   one is the stated hypothesis. Select checkpoints by validation
   `z_sigma_nmad`, and compare the fixed DESI/SDSS reports using the same
   normalized-residual definitions and catastrophic thresholds.
4. Reuse `scripts/run_materialized_experiment.py` and shared scripts when their
   behavior is identical to the proposed experiment. If not, place replacement
   code in `experiments/<id>/src/` and name it in the experiment README/config.
5. Create an experiment-local `train.slurm` that writes only to
   `data/checkpoints/experiments/<id>/` and `models/experiments/<id>/`.
6. Smoke-test a tiny run, inspect the resolved config saved beside its
   checkpoint, then submit the full Slurm run.

## Running And Comparing

Prepare reusable OOD data once on CPU, then submit each experiment's local
Slurm script. Check queue state with `squeue -u "$USER"`, follow its log under
`logs/`, and read the arm's `history_summary.json` after every epoch. A fair
comparison uses each arm's best checkpoint according to the same fixed
validation metric, followed by the same fixed DESI and SDSS benchmark sets.

The reusable baseline is `control_full_spectrum_v1`. Its first comparison is
`crop_transform_v1`. Copy the appropriate experiment directory structure, not
its output directories, when starting a new experiment. A control remains a
separate experiment because later methods should compare against its fixed,
independently reproducible results rather than overwrite or reinterpret it.
