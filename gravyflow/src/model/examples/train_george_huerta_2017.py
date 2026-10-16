#!/usr/bin/env python
"""
Train the George & Huerta 2017 Deep Filtering networks (arXiv:1701.00008v3)
with the run recipe.

Tasks:
- regression: the mass predictor;
- classification: the detector;
- pretrain: the paper's scheme, the predictor first, then its weights moved
  to the detector.

    python -m gravyflow.src.model.examples.train_george_huerta_2017 \\
        --output-dir RUN_DIR --model shallow --task pretrain --preset cpu

Each phase is a run of its own in ``RUN_DIR/regression`` or
``RUN_DIR/classification`` (outside the package), resumed if run again. With
the curriculum, training SNR falls from 80-100 to 5-15 over the first half of
the epochs; the second half trains at the 5-15 the model is validated on.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from dataclasses import replace
from pathlib import Path

import gravyflow as gf
from gravyflow.src.model.examples.george_huerta_2017 import (
    GeorgeHuerta2017Config, GeorgeHuerta2017Deep, GeorgeHuerta2017Shallow, curriculum_epochs,
)
from gravyflow.src.model.examples.recipe import (
    DetectionBatches, RegressionBatches, RunDirectory, TrainingRun, TrainingSchedule, materialise,
)

MODELS = {"shallow": GeorgeHuerta2017Shallow, "deep": GeorgeHuerta2017Deep}
SCHEDULES = {"cpu": TrainingSchedule.cpu, "gpu": TrainingSchedule.gpu}
MASSES = (gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train George & Huerta 2017 Deep Filtering")
    parser.add_argument("--output-dir", required=True,
                        help="Run directory, outside the package; existing runs there are resumed")
    parser.add_argument("--model", choices=sorted(MODELS), default="shallow")
    parser.add_argument("--task", choices=["regression", "classification", "pretrain"], default="pretrain")
    parser.add_argument("--preset", choices=sorted(SCHEDULES), default="cpu",
                        help="cpu: a short schedule, at 2048 Hz for the shallow network (the deep "
                             "one needs 8192 Hz); gpu: the paper's scale")
    parser.add_argument("--epochs", type=int, help="Override the preset's epochs (per phase)")
    parser.add_argument("--steps-per-epoch", type=int, help="Override the preset's steps per epoch")
    parser.add_argument("--validation-batches", type=int, help="Override the preset's validation batches")
    parser.add_argument("--no-curriculum", action="store_true", help="Train at the target SNR throughout")
    parser.add_argument("--seed", type=int, default=gf.Defaults.seed)
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    return parser.parse_args(argv)


def configure(args):
    """The model configuration and schedule, with the curriculum fitted to the run's length."""
    overrides = {k: v for k, v in {
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "validation_batches": args.validation_batches,
    }.items() if v is not None}
    schedule = replace(SCHEDULES[args.preset](), **overrides)
    shallow_on_cpu = args.preset == "cpu" and args.model == "shallow"
    config = GeorgeHuerta2017Config.cpu() if shallow_on_cpu else GeorgeHuerta2017Config()
    if args.no_curriculum:
        return replace(config, use_curriculum=False), schedule
    span = curriculum_epochs(schedule.epochs)
    config = replace(config, use_curriculum=True, snr_curriculum_num_epochs=span)
    # Early stopping counts only once the curriculum has reached the target SNR.
    return config, replace(schedule, monitor_from_epoch=span)


def train_phase(model_class, task, config, schedule, args, pretrained=None):
    regression = task == "regression"
    run = TrainingRun(
        directory=RunDirectory(Path(args.output_dir) / task),
        schedule=schedule,
        seed=args.seed,
        description={"model": model_class.__name__, "task": task, "config": config,
                     "transfer": pretrained is not None},
    )
    make = model_class.dataset_regression if regression else model_class.dataset
    wrap = (lambda ds: RegressionBatches(ds, MASSES)) if regression else DetectionBatches
    validation = materialise(
        wrap(make(config=config, seed=run.seed, group="validate",
                  steps_per_epoch=schedule.validation_batches)),
        schedule.validation_batches,
    )

    def build():
        model = model_class.model(config, task=task)
        if pretrained is not None:
            print(f"Transferred {model_class.transfer_weights(pretrained, model)} layers from the predictor")
        return model_class.compile_model(model, config, task=task)

    print(f"{model_class.__name__} {task}: {schedule.epochs} epochs x {schedule.steps_per_epoch} steps "
          f"in {run.directory.path}")
    return run.train(
        build_model=build,
        training_data=lambda seed: wrap(make(
            config=config, seed=seed, group="train", steps_per_epoch=schedule.steps_per_epoch,
            curriculum=config.use_curriculum)),
        validation_data=validation,
        verbose=args.verbose,
    )


def main(argv=None):
    args = parse_args(argv)
    model_class = MODELS[args.model]
    config, schedule = configure(args)
    if args.task in ("regression", "pretrain"):
        predictor = train_phase(model_class, "regression", config, schedule, args)
        if args.task == "regression":
            return predictor
        return train_phase(model_class, "classification", config, schedule, args, pretrained=predictor)
    return train_phase(model_class, "classification", config, schedule, args)


if __name__ == "__main__":
    main()
