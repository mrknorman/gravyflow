#!/usr/bin/env python
"""
Train an MLy network (Skliris, Norman & Sutton 2024, arXiv:2009.14611) with the
run recipe, and optionally validate it.

    python -m gravyflow.src.model.examples.train_mly_2024 \\
        --output-dir RUN_DIR --model coincidence [--preset cpu] [--validate 4096]

The run directory must be outside the package. Running the same command again
resumes the run from its last epoch; a run directory holding a run of another
configuration is refused. The best model ends up in ``RUN_DIR/model.keras``.
With ``--validate N``, the model is then scored by `gf.Validator` on N noise
examples and N WNB injections drawn from the validator's own seeds, and
``validation.h5`` and ``validation.html`` are written to ``RUN_DIR/validation``
(or ``--validation-dir``), and resumed if they belong to this model and these
settings.

Presets:
    cpu: H1 and L1, 10 epochs of 300 steps, 32 validation batches;
    gpu: the paper's H1, L1 and V1, and its scale: 135,000 (model 1) or
         270,000 (model 2) training examples per epoch in batches of 16, for
         15 or 10 epochs.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from dataclasses import replace
from pathlib import Path

import keras

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import DEFAULT_NOISE_PROFILE
from gravyflow.src.model.examples.mly_2024 import MLyCoherence, MLyCoincidence, MLyConfig
from gravyflow.src.model.examples.recipe import (
    DetectionBatches, RunDirectory, TrainingRun, TrainingSchedule, materialise,
)
from gravyflow.src.validate import ValidationConfig, Validator

MODELS = {"coincidence": MLyCoincidence, "coherence": MLyCoherence}

# The paper's training sets (90% of 150,000 or 300,000 examples; 10% held out
# for validation) in batches of 16, and its epochs; early stopping after 20
# epochs without improvement.
PAPER_SCHEDULES = {
    "coincidence": TrainingSchedule(epochs=15, steps_per_epoch=8438, validation_batches=938, patience=20),
    "coherence": TrainingSchedule(epochs=10, steps_per_epoch=16875, validation_batches=1875, patience=20),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an MLy burst-detection network")
    parser.add_argument("--output-dir", required=True,
                        help="Run directory, outside the package; an existing run there is resumed")
    parser.add_argument("--model", choices=sorted(MODELS), default="coincidence")
    parser.add_argument("--preset", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--epochs", type=int, help="Override the preset's epochs")
    parser.add_argument("--steps-per-epoch", type=int, help="Override the preset's steps per epoch")
    parser.add_argument("--validation-batches", type=int, help="Override the preset's validation batches")
    parser.add_argument("--patience", type=int, help="Override the preset's early-stopping patience")
    parser.add_argument("--seed", type=int, default=gf.Defaults.seed,
                        help="Root seed; each split draws its own streams from it")
    parser.add_argument("--validate", type=int, default=0, metavar="N",
                        help="Validate the trained model on N noise and N injection examples (0: skip)")
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--validation-dir", help="Where to write the validation (default: RUN_DIR/validation)")
    parser.add_argument("--validation-seed", type=int, default=gf.Defaults.seed,
                        help="The validator's own root seed")
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    return parser.parse_args(argv)


def configure(args):
    """The data configuration and the training schedule of the run."""
    if args.preset == "cpu":
        config, schedule = MLyConfig.cpu(), TrainingSchedule.cpu()
    else:
        config, schedule = MLyConfig(), PAPER_SCHEDULES[args.model]
    overrides = {k: v for k, v in {
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "validation_batches": args.validation_batches, "patience": args.patience,
    }.items() if v is not None}
    return config, replace(schedule, **overrides)


def train(args, model_class, config, schedule) -> keras.Model:
    run = TrainingRun(
        directory=RunDirectory(args.output_dir),
        schedule=schedule,
        seed=args.seed,
        description={"model": model_class.__name__, "config": config, "noise": gf.NoiseType.COLORED,
                     "noise_profile": DEFAULT_NOISE_PROFILE},
    )
    print(f"{model_class.__name__} (arXiv:2009.14611), preset {args.preset}: "
          f"{[ifo.name for ifo in config.ifos]} at {config.sample_rate_hertz:g} Hz, "
          f"{schedule.epochs} epochs x {schedule.steps_per_epoch} steps of {config.batch_size}")
    print(f"Run directory: {run.directory.path}")
    validation = materialise(
        DetectionBatches(model_class.dataset(
            config=config, seed=run.seed, group="validate",
            steps_per_epoch=schedule.validation_batches)),
        schedule.validation_batches,
        run.directory.path / "validation",
    )
    model = run.train(
        build_model=lambda: model_class.compile_model(model_class.model(config), config),
        training_data=lambda seed: DetectionBatches(model_class.dataset(
            config=config, seed=seed, group="train", steps_per_epoch=schedule.steps_per_epoch)),
        validation_data=validation,
        verbose=args.verbose,
    )
    state = run.directory.state
    print(f"Best epoch {state['best_epoch']}: val_loss {state['best_val_loss']:.4f}; "
          f"model saved to {run.directory.path / 'model.keras'}")
    return model


def validate(args, model_class, config, model) -> Validator:
    """Score the model on the validator's own seeds; print the background, efficiencies and ROC."""
    output_dir = Path(args.validation_dir) if args.validation_dir else Path(args.output_dir) / "validation"
    validator = Validator.validate(
        model=model,
        name=f"mly_{args.model}_colored",
        dataset_args=model_class.validation_dataset_args(config),
        config=ValidationConfig(seed=args.validation_seed, num_examples=args.validate,
                                batch_size=args.validation_batch_size, snr_range=(0.0, 30.0),
                                snr_bin_width=2.0),
        checkpoint_file_path=output_dir / "validation.h5",
    )
    validator.plot(output_dir / "validation.html")
    background = validator.background()
    print(f"Background: {background.num_windows} windows, {background.livetime_seconds:.0f} s livetime")
    for far, result in validator.efficiency_curves([1e-1, 1e-2, 1e-3]).items():
        threshold, fit = result.threshold, result.fit
        if threshold.resolved:
            print(f"FAR {far:.0e} Hz: threshold {threshold.threshold:.3f}, achieved "
                  f"{threshold.achieved.rate_hz:.2e} Hz (1σ upper end {threshold.achieved.upper_hz:.2e}); "
                  f"efficiency by SNR bin {[round(e, 2) for e in result.binned.efficiency.tolist()]}; "
                  f"50% and 90% SNR {fit.snr_50:.2f}, {fit.snr_90:.2f}")
            if not result.quotable:  # the standard's rule: ten independent background events above it
                print(f"    not quoted: {threshold.independent_events} independent background events, fewer than 10")
        else:
            print(f"FAR {far:.0e} Hz: unresolved by {background.livetime_seconds:.0f} s of background")
    for key, roc in validator.roc_curves().items():
        print(f"ROC {key}: AUC {roc['roc_auc']:.4f}")
    return validator


def main(argv=None) -> keras.Model:
    args = parse_args(argv)
    model_class = MODELS[args.model]
    config, schedule = configure(args)
    model = train(args, model_class, config, schedule)
    if args.validate > 0:
        validate(args, model_class, config, model)
    return model


if __name__ == "__main__":
    main()
