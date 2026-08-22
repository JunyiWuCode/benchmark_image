from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_image.official_suite import (
    OFFICIAL_PROTOCOLS,
    build_records,
    expected_image_count,
    select_coverage_smoke,
    verify_sources,
    write_records,
)
from benchmark_image.official_layouts import (
    audit_raw_images,
    collect_generation_manifests,
    materialize_official_layouts,
    read_records,
)


def _source_map(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"--source must be NAME=PATH, got {value!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Source-locked Z-Image Base English benchmark protocols"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocols = subparsers.add_parser("protocols")
    protocols.add_argument("--json", action="store_true")
    for command in ("preflight", "export-records"):
        child = subparsers.add_parser(command)
        child.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
        child.add_argument("--benchmark", action="append", choices=tuple(OFFICIAL_PROTOCOLS))
        if command == "export-records":
            child.add_argument("--profile", choices=("official_report", "training_monitor"), default="official_report")
            child.add_argument("--fallback-resolution", type=int, default=1024)
            child.add_argument(
                "--hps-resolution",
                type=int,
                default=0,
                help="Override HPSv3 generation with a square size; 0 keeps official aspect-1024.",
            )
            child.add_argument("--smoke-max-prompts-per-benchmark", type=int)
            child.add_argument("--coverage-smoke", action="store_true")
            child.add_argument("--output", type=Path, required=True)
    for command in ("audit-images", "collect-generation", "materialize-layouts"):
        child = subparsers.add_parser(command)
        child.add_argument("--records", type=Path, required=True)
        child.add_argument("--image-root", type=Path, required=True)
        if command == "materialize-layouts":
            child.add_argument("--layout-root", type=Path, required=True)
            child.add_argument("--model-name", default="Z-Image-Base-cfg4-nfe50")
        if command == "collect-generation":
            child.add_argument(
                "--existence-only",
                action="store_true",
                help="Check that every generated file exists without reopening every image.",
            )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "protocols":
        payload = {
            name: {
                "prompts": spec.prompts,
                "samples_per_prompt": spec.samples_per_prompt,
                "images": spec.images,
                "resolution_policy": spec.resolution_policy,
                "primary_metric": spec.primary_metric,
            }
            for name, spec in OFFICIAL_PROTOCOLS.items()
        }
        payload["_total"] = {"images": expected_image_count()}
        print(json.dumps(payload, indent=2) if args.json else "\n".join(f"{key}: {value}" for key, value in payload.items()))
        return
    if args.command in {"audit-images", "collect-generation", "materialize-layouts"}:
        records = read_records(args.records)
        if args.command == "collect-generation":
            print(json.dumps(
                collect_generation_manifests(
                    records,
                    args.image_root,
                    verify_images=not args.existence_only,
                ),
                indent=2,
            ))
            return
        audit = audit_raw_images(records, args.image_root)
        if not audit["complete"]:
            print(json.dumps(audit, indent=2))
            raise RuntimeError(f"Image audit failed for {audit['failure_count']} records")
        if args.command == "audit-images":
            print(json.dumps(audit, indent=2))
        else:
            print(json.dumps(materialize_official_layouts(records, args.image_root, args.layout_root, model_name=args.model_name), indent=2))
        return
    sources = _source_map(args.source)
    if args.command == "preflight":
        print(json.dumps(verify_sources(sources), indent=2))
        return
    records = build_records(
        sources,
        args.benchmark,
        profile=args.profile,
        smoke_max_prompts_per_benchmark=args.smoke_max_prompts_per_benchmark,
        fallback_resolution=args.fallback_resolution,
        hps_resolution=args.hps_resolution or None,
    )
    if args.coverage_smoke:
        if args.profile != "official_report" or args.smoke_max_prompts_per_benchmark is not None:
            raise ValueError("--coverage-smoke requires unsliced --profile official_report")
        records = select_coverage_smoke(records)
    summary = write_records(args.output, records)
    summary.update(
        {
            "benchmarks": args.benchmark or list(OFFICIAL_PROTOCOLS),
            "profile": args.profile,
            "smoke_max_prompts_per_benchmark": args.smoke_max_prompts_per_benchmark,
            "coverage_smoke": args.coverage_smoke,
            "fallback_resolution": args.fallback_resolution,
            "hps_resolution": args.hps_resolution,
            "reportable": args.profile == "official_report" and args.smoke_max_prompts_per_benchmark is None and not args.coverage_smoke,
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
