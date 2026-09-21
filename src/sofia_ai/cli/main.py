"""Sofia Engine command-line interface.

Commands are read-only by default. There is **no actuation subcommand**: Sofia's
CLI inspects, analyses, replays and benchmarks. Any command that would act on
equipment must go through :class:`sofia_ai.decision.policy.PolicyEngine` in
application code, never through this CLI.

Commands::

    sofia info        version, capabilities, environment
    sofia doctor      dependency and health checks
    sofia inspect     inspect a telemetry file's structure and quality
    sofia analyze     run the analysis pipeline over a signal file
    sofia replay      replay telemetry through the pipeline
    sofia benchmark   run the benchmark suite
    sofia models      list registered model backends
    sofia devices     describe configured devices
    sofia config      validate and dump configuration
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from typing import Any

from .. import CAPABILITIES, __version__
from ..core.errors import SofiaError

__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "build_parser", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sofia",
        description=(
            "Sofia Engine — open-source edge intelligence for industrial telemetry, "
            "machine health, embedded ML and safe technical automation."
        ),
        epilog=(
            "Sofia is not a certified SIS/PLC safety system. "
            "This CLI performs no machine actuation."
        ),
    )
    parser.add_argument("--version", action="version",
                        version=f"sofia {__version__} (Apache-2.0)")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    info = subparsers.add_parser("info", help="version, capabilities and environment")
    info.add_argument("--json", action="store_true", help="emit JSON")

    doctor = subparsers.add_parser("doctor", help="environment and dependency checks")
    doctor.add_argument("--json", action="store_true", help="emit JSON")

    inspect_cmd = subparsers.add_parser("inspect", help="inspect a telemetry file")
    inspect_cmd.add_argument("path", help="CSV or JSONL telemetry file")
    inspect_cmd.add_argument("--format", choices=("csv", "jsonl"), default=None,
                             help="force a parser (default: infer from extension)")
    inspect_cmd.add_argument("--device", default=None, help="default device id for CSV")
    inspect_cmd.add_argument("--unit", default=None, help="default unit for CSV")
    inspect_cmd.add_argument("--limit", type=int, default=20,
                             help="sample rows to display (default 20)")
    inspect_cmd.add_argument("--json", action="store_true", help="emit JSON")

    analyze = subparsers.add_parser("analyze", help="analyze a signal file")
    analyze.add_argument("path", help="CSV or JSONL file")
    analyze.add_argument("--channel", default=None, help="channel to analyze")
    analyze.add_argument("--sample-rate", type=float, default=1000.0,
                         help="sample rate in Hz (default 1000)")
    analyze.add_argument("--window", type=int, default=1024, help="window length")
    analyze.add_argument("--hop", type=int, default=None, help="window hop")
    analyze.add_argument("--detector", default="mad",
                         help="detector backend: mad|zscore|ewma|iqr|cusum|threshold")
    analyze.add_argument("--feature", default="rms", help="feature to monitor")
    analyze.add_argument("--threshold", type=float, default=3.0,
                         help="detector threshold")
    analyze.add_argument("--shaft-hz", type=float, default=None,
                         help="shaft frequency for order-energy features")
    analyze.add_argument("--json", action="store_true", help="emit JSON")

    replay = subparsers.add_parser("replay", help="replay telemetry through the pipeline")
    replay.add_argument("path", help="CSV or JSONL file")
    replay.add_argument("--window", type=int, default=512, help="window length")
    replay.add_argument("--hop", type=int, default=256, help="window hop")
    replay.add_argument("--sample-rate", type=float, default=1000.0,
                        help="sample rate in Hz")
    replay.add_argument("--detector", default="mad", help="detector backend")
    replay.add_argument("--feature", default="rms", help="feature to monitor")
    replay.add_argument("--ticks", type=int, default=64, help="maximum runtime ticks")
    replay.add_argument("--json", action="store_true", help="emit JSON")

    bench = subparsers.add_parser("benchmark", help="run the benchmark suite")
    bench.add_argument("--suite", default="all",
                       help="all|ingest|features|inference|pipeline|quantum")
    bench.add_argument("--iterations", type=int, default=50,
                       help="iterations per measurement (default 50)")
    bench.add_argument("--json", action="store_true", help="emit JSON")
    bench.add_argument("--output", default=None, help="write results to this JSON file")

    models = subparsers.add_parser("models", help="list registered model backends")
    models.add_argument("--json", action="store_true", help="emit JSON")

    devices = subparsers.add_parser("devices", help="describe configured devices")
    devices.add_argument("--config", default=None, help="Sofia config JSON path")
    devices.add_argument("--json", action="store_true", help="emit JSON")

    config = subparsers.add_parser("config", help="validate and dump configuration")
    config.add_argument("--config", default=None, help="Sofia config JSON path")
    config.add_argument("--json", action="store_true", help="emit JSON")

    ask = subparsers.add_parser("ask", help="consult Sofia AI Copilot with NVIDIA or OpenRouter")
    ask.add_argument("question", help="engineering or diagnostic question")
    ask.add_argument("--device", default="unknown-device", help="target device id")
    ask.add_argument("--provider", choices=("auto", "nvidia", "openrouter", "offline"),
                     default="auto", help="AI provider (default: auto)")
    ask.add_argument("--model", default=None, help="custom model name")
    ask.add_argument("--json", action="store_true", help="emit JSON")

    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any], as_json: bool, title: str | None = None) -> int:
    if as_json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
        return EXIT_OK
    if title:
        print(title)
        print("-" * len(title))
    _print_tree(payload)
    return EXIT_OK


def _print_tree(payload: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)) and value:
                print(f"{prefix}{key}:")
                _print_tree(value, indent + 2)
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            if isinstance(item, (dict, list)):
                _print_tree(item, indent + 2)
            else:
                print(f"{prefix}- {item}")
    else:
        print(f"{prefix}{payload}")


def cmd_info(args: argparse.Namespace) -> int:
    """Report version, capabilities and environment."""
    payload = {
        "sofia_version": __version__,
        "license": CAPABILITIES["license"],
        "organization": CAPABILITIES["organization"],
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "capabilities": CAPABILITIES,
    }
    return _emit(payload, args.json, "Sofia Engine")


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check optional dependencies and report which capabilities are available."""
    checks = _dependency_checks()
    payload = {
        "sofia_version": __version__,
        "checks": checks,
        "offline_ready": True,
        "notes": [
            "Core capability requires only numpy; extras are optional.",
            "MQTT/Modbus/serial adapters are unit-tested with injected fakes only.",
            "Microcontroller targets are NOT VERIFIED in this repository.",
        ],
    }
    return _emit(payload, args.json, "Sofia doctor")


def _dependency_checks() -> list[dict[str, Any]]:
    def probe(module: str, extra: str, purpose: str) -> dict[str, Any]:
        try:
            __import__(module)
            return {"name": module, "extra": extra, "purpose": purpose,
                    "available": True, "version": _module_version(module)}
        except ImportError:
            return {"name": module, "extra": extra, "purpose": purpose,
                    "available": False, "version": None}

    return [
        probe("numpy", "core", "numerical core (required)"),
        probe("paho.mqtt.client", "mqtt", "MQTT telemetry adapter"),
        probe("pymodbus.client", "modbus", "Modbus TCP/RTU adapter"),
        probe("serial", "serial", "serial line adapter"),
        probe("onnxruntime", "onnx", "ONNX Runtime inference backend"),
        probe("torch", "torch", "PyTorch backend and experimental RL"),
        probe("fastapi", "api", "optional HTTP API"),
        probe("cryptography", "security", "optional manifest signature verification"),
        {
            "name": "NVIDIA_API_KEY",
            "extra": "copilot",
            "purpose": "NVIDIA NIM AI Copilot backend",
            "available": bool(
                os.environ.get("NVIDIA_API_KEY") or os.environ.get("SOFIA_NVIDIA_API_KEY")
            ),
            "version": "configured"
            if (os.environ.get("NVIDIA_API_KEY") or os.environ.get("SOFIA_NVIDIA_API_KEY"))
            else None,
        },
        {
            "name": "OPENROUTER_API_KEY",
            "extra": "copilot",
            "purpose": "OpenRouter AI Copilot backend",
            "available": bool(
                os.environ.get("OPENROUTER_API_KEY") or os.environ.get("SOFIA_OPENROUTER_API_KEY")
            ),
            "version": "configured"
            if (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("SOFIA_OPENROUTER_API_KEY"))
            else None,
        },
    ]


def _module_version(module: str) -> str | None:
    try:
        imported = __import__(module, fromlist=["__version__"])
        return str(getattr(imported, "__version__", "unknown"))
    except (ImportError, AttributeError):  # pragma: no cover
        return None


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a telemetry file: counts, channels, units, quality, anomalies."""
    from ..core.quality import DataQuality
    from ..core.time import ClockMonitor
    from ..telemetry.file_sources import CsvSource, JsonlSource

    path = str(args.path)
    fmt = args.format or ("jsonl" if path.lower().endswith(".jsonl") else "csv")
    source = (
        JsonlSource(path, strict=False)
        if fmt == "jsonl"
        else CsvSource(path, device_id=args.device, unit=args.unit, strict=False)
    )
    with source:
        samples = []
        while True:
            batch = source.read(max_records=10_000)
            if not batch:
                break
            samples.extend(batch)

    monitor = ClockMonitor()
    anomalies = 0
    for sample in samples:
        if monitor.observe(sample.timestamp) is not None:
            anomalies += 1

    channels = sorted({s.channel for s in samples})
    devices = sorted({s.device_id for s in samples})
    units = sorted({s.unit for s in samples})
    quality_counts = {q.value: 0 for q in DataQuality}
    for sample in samples:
        quality_counts[sample.quality.value] += 1

    values = [s.value for s in samples]
    payload: dict[str, Any] = {
        "path": path,
        "format": fmt,
        "samples": len(samples),
        "devices": devices,
        "channels": channels,
        "units": units,
        "quality": quality_counts,
        "clock_anomalies": anomalies,
        "time_span_s": (max(s.timestamp for s in samples) - min(s.timestamp for s in samples))
        if samples else 0.0,
        "value_stats": _basic_stats(values),
        "source_stats": source.stats.as_dict(),
        "preview": [s.to_dict() for s in samples[: max(0, int(args.limit))]],
    }
    return _emit(payload, args.json, f"Inspect {path}")


def _basic_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a signal file: features, spectrum, detector, health event."""
    from ..features.extractor import extract_from_array
    from ..inference import build_manifest_for, create_backend
    from ..signal.windowing import windows_from_samples
    from ..telemetry.file_sources import CsvSource, JsonlSource

    path = str(args.path)
    fmt = "jsonl" if path.lower().endswith(".jsonl") else "csv"
    source = (
        JsonlSource(path, strict=False) if fmt == "jsonl"
        else CsvSource(path, device_id="cli-device", strict=False)
    )
    with source:
        samples = []
        while True:
            batch = source.read(max_records=100_000)
            if not batch:
                break
            samples.extend(batch)

    if not samples:
        return _emit({"error": "no samples found", "path": path}, args.json, "Analyze")

    channel = args.channel or samples[0].channel
    selected = [s for s in samples if s.channel == channel] or samples
    hop = args.hop or max(1, args.window // 2)
    windows = windows_from_samples(
        selected,
        length=args.window,
        hop=hop,
        sample_rate=args.sample_rate,
        unit=selected[0].unit,
        channel=channel,
        device_id=selected[0].device_id,
    )
    if not windows:
        return _emit(
            {"error": "insufficient samples for one window",
             "samples": len(selected), "window": args.window},
            args.json, "Analyze",
        )

    backend = create_backend(args.detector)
    if hasattr(backend, "config"):
        backend.config = type(backend.config)(
            feature=args.feature, threshold=args.threshold,
            window_size=32, warmup=4, min_samples=4,
        )
    backend.load(build_manifest_for(backend, model_id=f"cli-{args.detector}"))

    results = []
    first_features: dict[str, float] = {}
    for window in windows:
        features = extract_from_array(window.values, window.sample_rate,
                                      channel=channel, shaft_hz=args.shaft_hz)
        if not first_features:
            first_features = features.as_dict()
        results.append(backend.infer(features))

    anomalous = [r for r in results if r.is_anomalous]
    payload: dict[str, Any] = {
        "path": path,
        "channel": channel,
        "unit": selected[0].unit,
        "sample_rate": args.sample_rate,
        "windows": len(windows),
        "features": len(windows[0].values) if results else 0,
        "detector": args.detector,
        "feature": args.feature,
        "anomalous_windows": len(anomalous),
        "anomaly_ratio": len(anomalous) / len(results) if results else 0.0,
        "mean_confidence": (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        ),
        "mean_inference_ms": (
            1000.0 * sum(r.latency_s for r in results) / len(results) if results else 0.0
        ),
        "first_window_features": first_features,
        "note": "latency measured on this machine only; see docs/benchmarks.md",
    }
    return _emit(payload, args.json, f"Analyze {path}")


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a telemetry file through the edge runtime."""
    from ..edge import EdgeRuntime, RuntimeConfig
    from ..inference import build_manifest_for, create_backend
    from ..observability import MetricsRegistry
    from ..telemetry.file_sources import CsvSource, JsonlSource

    path = str(args.path)
    fmt = "jsonl" if path.lower().endswith(".jsonl") else "csv"
    source = (
        JsonlSource(path, strict=False) if fmt == "jsonl"
        else CsvSource(path, device_id="cli-device", strict=False)
    )
    backend = create_backend(args.detector)
    if hasattr(backend, "config"):
        backend.config = type(backend.config)(
            feature=args.feature, window_size=32, warmup=4, min_samples=4
        )
    backend.load(build_manifest_for(backend, model_id=f"cli-{args.detector}"))
    metrics = MetricsRegistry()
    runtime = EdgeRuntime(
        source=source,
        backend=backend,
        config=RuntimeConfig(window_length=args.window, window_hop=args.hop,
                             sample_rate=args.sample_rate, buffer_capacity=65536,
                             max_windows_per_tick=64),
        metrics=metrics,
    )
    runtime.start()
    events = runtime.run(max_ticks=args.ticks, batch_size=8192)
    snapshot = runtime.health_snapshot()
    runtime.stop()

    payload = {
        "path": path,
        "events": [e.to_dict() for e in events[:20]],
        "event_count": len(events),
        "stats": snapshot["stats"],
        "counters": metrics.snapshot()["counters"],
        "histograms": metrics.snapshot()["histograms"],
    }
    return _emit(payload, args.json, f"Replay {path}")


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the benchmark suite."""
    from ..benchmarking.core import format_report
    from ..benchmarking.suites import run_suite

    payload = run_suite(suite=args.suite, iterations=args.iterations)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, default=str)
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
        return EXIT_OK
    print(format_report(payload))
    return EXIT_OK


def cmd_models(args: argparse.Namespace) -> int:
    """List registered model backends."""
    from ..inference import model_registry

    entries = []
    for name in model_registry.names():
        try:
            backend = model_registry.create(name)
            entries.append({"name": name, "metadata": backend.metadata().to_dict()})
        except Exception as exc:
            entries.append({"name": name, "error": type(exc).__name__})
    return _emit({"backends": entries, "count": len(entries)}, args.json, "Model backends")


def cmd_devices(args: argparse.Namespace) -> int:
    """Describe configured devices."""
    from ..config import load_config

    config = load_config(args.config)
    payload = {
        "device_id": config.device_id,
        "offline": config.offline,
        "device_allowlist": list(config.device_allowlist),
        "devices": [
            {
                "device_id": d.device_id,
                "tags": dict(d.tags),
                "channels": [
                    {"name": c.name, "unit": c.unit, "minimum": c.minimum,
                     "maximum": c.maximum, "sample_rate": c.sample_rate}
                    for c in d.channels
                ],
            }
            for d in config.devices
        ],
    }
    return _emit(payload, args.json, "Devices")


def cmd_config(args: argparse.Namespace) -> int:
    """Validate and dump configuration."""
    from ..config import load_config

    config = load_config(args.config)
    payload = {"valid": True, "version": config.version, "config": config.to_dict()}
    return _emit(payload, args.json, "Configuration")


def cmd_ask(args: argparse.Namespace) -> int:
    """Consult Sofia AI Copilot."""
    from ..copilot.base import CopilotRequest
    from ..copilot.providers import AIEngine

    engine = AIEngine(provider=args.provider, model=args.model, allow_provider_text=True)
    req = CopilotRequest(
        question=args.question,
        device_id=args.device,
        audience="engineer",
    )
    res = engine.ask(req)
    if args.json:
        return _emit(res.to_dict(), True)
    print(res.text)
    return EXIT_OK


_COMMANDS = {
    "info": cmd_info,
    "doctor": cmd_doctor,
    "inspect": cmd_inspect,
    "analyze": cmd_analyze,
    "replay": cmd_replay,
    "benchmark": cmd_benchmark,
    "models": cmd_models,
    "devices": cmd_devices,
    "config": cmd_config,
    "ask": cmd_ask,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    handler = _COMMANDS[str(args.command)]
    try:
        return int(handler(args))
    except SofiaError as exc:
        print(f"sofia: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"sofia: file not found: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        return 130
