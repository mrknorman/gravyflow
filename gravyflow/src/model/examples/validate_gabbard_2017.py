#!/usr/bin/env python
"""
Validate a Gabbard et al. 2017 model trained by ``train_gabbard_2017``.

    python -m gravyflow.src.model.examples.validate_gabbard_2017 --run-dir RUN_DIR

Scores a noise background and an injection set drawn from the validator's own
seeds (never the training seed), and writes ``validation.h5`` and the report
``validation.html`` to ``RUN_DIR/validation`` (or ``--output-dir``). Running it
again resumes from ``validation.h5`` if that file belongs to this model and these
settings, and refuses it otherwise.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from pathlib import Path

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.model.examples.recipe import RunDirectory
from gravyflow.src.validate import ValidationConfig, Validator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate a trained Gabbard et al. 2017 model")
    parser.add_argument("--run-dir", required=True, help="The training run's directory")
    parser.add_argument("--output-dir", help="Where to write results (default: RUN_DIR/validation)")
    parser.add_argument("--noise-type", choices=["colored", "real"], default="colored",
                        help="Background noise; real noise needs network access")
    parser.add_argument("--num-examples", type=int, default=100_000,
                        help="Examples in the noise background and in the injection set")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=gf.Defaults.seed, help="The validator's own root seed")
    parser.add_argument("--real-events", action="store_true",
                        help="Also score real GW events (needs network access)")
    parser.add_argument("--mode", choices=["stream", "window"], default="stream",
                        help="stream (the standard: the model slid along continuous strain) or window "
                             "(each example one window; kept for compatibility and reproduction)")
    return parser.parse_args(argv)


def build_validation(args, run: dict) -> tuple:
    """
    The validator's data, exactly as `main` scores it: the dataset arguments,
    from the training run's recorded configuration, and the validation settings.
    """
    data = run["description"]["config"]
    noise_type = gf.NoiseType.REAL if args.noise_type == "real" else gf.NoiseType.COLORED
    dataset_args = validation_dataset_args(
        noise_type=noise_type,
        config=ValidationDatasetConfig(
            sample_rate_hertz=data["sample_rate_hertz"],
            onsource_duration_seconds=data["onsource_duration_seconds"],
            offsource_duration_seconds=data["offsource_duration_seconds"],
        ),
    )
    config = ValidationConfig(
        seed=args.seed,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        score_real_events=args.real_events,
        mode=args.mode,
    )
    return dataset_args, config


def main(argv=None) -> Validator:
    args = parse_args(argv)
    run_dir = RunDirectory(args.run_dir)
    run = run_dir.run
    if run is None:
        raise FileNotFoundError(f"No training run in {run_dir.path}.")
    model = run_dir.load_model("model", run["config_sha256"])

    dataset_args, config = build_validation(args, run)
    noise_type = dataset_args["noise_obtainer"].noise_type
    output_dir = Path(args.output_dir) if args.output_dir else run_dir.path / "validation"
    name = f"gabbard_2017_{noise_type.name.lower()}"
    print(f"Validating {run_dir.path / 'model.keras'} on {noise_type.name} noise; results in {output_dir}")
    validator = Validator.validate(
        model=model,
        name=name,
        dataset_args=dataset_args,
        config=config,
        checkpoint_file_path=output_dir / "validation.h5",
    )
    validator.plot(output_dir / "validation.html")

    background = validator.background()
    print(f"Background: {validator.far_scores.size} windows scored, {background.livetime_seconds:.0f} s livetime")
    for far, result in validator.efficiency_curves([1e-1, 1e-2, 1e-3]).items():
        threshold, fit = result.threshold, result.fit
        if threshold.resolved:
            print(f"FAR {far:.0e} Hz: threshold {threshold.threshold:.3f}, achieved "
                  f"{threshold.achieved.rate_hz:.2e} Hz (1σ upper end {threshold.achieved.upper_hz:.2e}); "
                  f"efficiency by SNR bin {[round(e, 2) for e in result.binned.efficiency.tolist()]}; "
                  f"50% and 90% SNR {fit.snr_50:.2f}, {fit.snr_90:.2f}")
            if not result.quotable:  # quoted only with ten independent background events above it
                print(f"    not quoted: {threshold.independent_events} independent background events, fewer than 10")
        else:
            print(f"FAR {far:.0e} Hz: unresolved by {background.livetime_seconds:.0f} s of background")
    for key, roc in validator.roc_curves().items():
        print(f"ROC {key}: AUC {roc['roc_auc']:.4f}")
    return validator


if __name__ == "__main__":
    main()
