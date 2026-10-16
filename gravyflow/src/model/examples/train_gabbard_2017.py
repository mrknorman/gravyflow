#!/usr/bin/env python
"""
Train the Gabbard et al. 2017 CNN (arXiv:1712.06041v2) with the run recipe.

    python -m gravyflow.src.model.examples.train_gabbard_2017 --output-dir RUN_DIR [--preset cpu]

The run directory must be outside the package. Running the same command again
resumes the run from its last epoch; a run directory holding a run of another
configuration is refused. The best model ends up in ``RUN_DIR/model.keras``.

Presets:
    cpu: 2048 Hz, 10 epochs of 300 steps, 32 validation batches (minutes on a laptop);
    gpu: the paper's 8192 Hz, up to 1000 epochs of 11,250 steps (360,000 examples).
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from dataclasses import replace

import keras

import gravyflow as gf
from gravyflow.src.model.examples.gabbard_2017 import Gabbard2017, Gabbard2017Config
from gravyflow.src.model.examples.recipe import (
    DetectionBatches, RunDirectory, TrainingRun, TrainingSchedule, materialise,
)

PRESETS = {
    "cpu": (Gabbard2017Config.cpu, TrainingSchedule.cpu),
    "gpu": (Gabbard2017Config, TrainingSchedule.gpu),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Gabbard et al. 2017 BBH detection CNN")
    parser.add_argument("--output-dir", required=True,
                        help="Run directory, outside the package; an existing run there is resumed")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="cpu")
    parser.add_argument("--epochs", type=int, help="Override the preset's epochs")
    parser.add_argument("--steps-per-epoch", type=int, help="Override the preset's steps per epoch")
    parser.add_argument("--validation-batches", type=int, help="Override the preset's validation batches")
    parser.add_argument("--patience", type=int, help="Override the preset's early-stopping patience")
    parser.add_argument("--seed", type=int, default=gf.Defaults.seed,
                        help="Root seed; each split derives its own")
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    return parser.parse_args(argv)


def build_run(args) -> tuple:
    config_factory, schedule_factory = PRESETS[args.preset]
    config = config_factory()
    overrides = {k: v for k, v in {
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "validation_batches": args.validation_batches, "patience": args.patience,
    }.items() if v is not None}
    schedule = replace(schedule_factory(), **overrides)
    run = TrainingRun(
        directory=RunDirectory(args.output_dir),
        schedule=schedule,
        seed=args.seed,
        description={"model": "Gabbard2017", "config": config, "noise": gf.NoiseType.COLORED},
    )
    return config, run


def main(argv=None) -> keras.Model:
    args = parse_args(argv)
    config, run = build_run(args)
    schedule = run.schedule
    print(f"Gabbard et al. 2017 (arXiv:1712.06041v2), preset {args.preset}: "
          f"{config.sample_rate_hertz:g} Hz, {schedule.epochs} epochs x {schedule.steps_per_epoch} steps")
    print(f"Run directory: {run.directory.path}")

    validation = materialise(
        DetectionBatches(Gabbard2017.dataset(
            config=config, seed=run.seeds["validate"], group="validate",
            steps_per_epoch=schedule.validation_batches)),
        schedule.validation_batches,
    )
    model = run.train(
        build_model=lambda: Gabbard2017.compile_model(Gabbard2017.model(config), config),
        training_data=lambda seed, first_epoch: DetectionBatches(Gabbard2017.dataset(
            config=config, seed=seed, group="train", steps_per_epoch=schedule.steps_per_epoch)),
        validation_data=validation,
        verbose=args.verbose,
    )
    state = run.directory.state
    print(f"Best epoch {state['best_epoch']}: val_loss {state['best_val_loss']:.4f}; "
          f"model saved to {run.directory.path / 'model.keras'}")
    return model


if __name__ == "__main__":
    main()
