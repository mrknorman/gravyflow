#!/usr/bin/env python
"""
Validate the matched-filter baseline on the example models' common validation data.

    python -m gravyflow.src.model.examples.validate_matched_filter_baseline --output-dir DIR

Scores a noise background and an injection set drawn from the validator's own
seeds, exactly as `validate_gabbard_2017` does for a trained network, so the
two reports compare like with like. Writes ``validation.h5`` and
``validation.html`` to DIR; running it again resumes from ``validation.h5`` if
that file belongs to this model and these settings.

The matched filter costs tens of milliseconds per example on a CPU with the
default bank (about a thousand templates; the cost grows with their number), so
the defaults score fewer examples than the network's script.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from pathlib import Path

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig
from gravyflow.src.model.examples.matched_filter_baseline import MatchedFilterBaseline
from gravyflow.src.validate import ValidationConfig, Validator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate the matched-filter baseline")
    parser.add_argument("--output-dir", required=True, help="Where to write results")
    parser.add_argument("--num-examples", type=int, default=10_000,
                        help="Examples in the noise background and in the injection set")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=gf.Defaults.seed, help="The validator's own root seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip the HTML report")
    return parser.parse_args(argv)


def main(argv=None) -> Validator:
    args = parse_args(argv)
    baseline = MatchedFilterBaseline.config
    dataset_args = MatchedFilterBaseline.validation_dataset_args(
        config=ValidationDatasetConfig(
            sample_rate_hertz=baseline.sample_rate_hertz,
            onsource_duration_seconds=baseline.onsource_duration_seconds,
            offsource_duration_seconds=baseline.offsource_duration_seconds,
            mass_range_msun=baseline.mass_range_msun,
        ),
        baseline=baseline,
    )
    config = ValidationConfig(seed=args.seed, num_examples=args.num_examples, batch_size=args.batch_size)
    output_dir = Path(args.output_dir)
    # The search covers the merger times the validation data's placement produces.
    model = MatchedFilterBaseline.model(dataset_args=dataset_args)
    layer = model.get_layer("matched_filter")
    print(f"Validating the matched-filter baseline ({layer.num_templates} templates, merger times "
          f"{layer.search_window_seconds[0]:.3f}-{layer.search_window_seconds[1]:.3f} s) on coloured noise; "
          f"results in {output_dir}")
    validator = Validator.validate(
        model=model,
        name="matched_filter_baseline",
        dataset_args=dataset_args,
        config=config,
        checkpoint_file_path=output_dir / "validation.h5",
    )
    if not args.no_plot:
        validator.plot(output_dir / "validation.html")

    to_snr = lambda score: score * baseline.temperature + baseline.snr_threshold
    background = validator.background()
    print(f"Background: {background.num_windows} windows, {background.livetime_seconds:.0f} s livetime")
    for far, result in validator.efficiency_curves([1e-1, 1e-2, 1e-3]).items():
        threshold, fit = result.threshold, result.fit
        if threshold.resolved:
            print(f"FAR {far:.0e} Hz: network SNR threshold {to_snr(threshold.threshold):.3f}, achieved "
                  f"{threshold.achieved.rate_hz:.2e} Hz (1σ upper end {threshold.achieved.upper_hz:.2e}); "
                  f"efficiency by SNR bin {[round(e, 2) for e in result.binned.efficiency.tolist()]}; "
                  f"50% and 90% SNR {fit.snr_50:.2f}, {fit.snr_90:.2f}")
        else:
            print(f"FAR {far:.0e} Hz: unresolved by {background.livetime_seconds:.0f} s of background")
    for key, roc in validator.roc_curves().items():
        print(f"ROC {key}: AUC {roc['roc_auc']:.4f}")
    return validator


if __name__ == "__main__":
    main()
