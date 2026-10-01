from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking.trace_commitment import train_trace_commitment


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train trace-supervised token commitment scorers.")
    parser.add_argument("--fit-csv", required=True, help="Trace CSV for fitting scaler and base logistic models.")
    parser.add_argument(
        "--calibration-csv",
        required=True,
        help="Trace CSV for separate sigmoid calibration and offline metrics.",
    )
    parser.add_argument("--fit-manifest", default=None, help="Optional immutable fit example manifest.")
    parser.add_argument(
        "--calibration-manifest",
        default=None,
        help="Optional immutable calibration example manifest.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for models, metadata, and reports.")
    parser.add_argument("--seed", type=int, default=23)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifacts = train_trace_commitment(
        fit_csv=args.fit_csv,
        calibration_csv=args.calibration_csv,
        fit_manifest=args.fit_manifest,
        calibration_manifest=args.calibration_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"trace_model={artifacts.trace_model_path}")
    print(f"confidence_model={artifacts.confidence_model_path}")
    print(f"metadata={artifacts.metadata_path}")
    print(f"metrics={artifacts.metrics_path}")
    print(f"provenance={artifacts.provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
