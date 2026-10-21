"""Run one MLU590 case in no-save and save modes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from case_metadata import CASES


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent
VALIDATOR = PACKAGE_ROOT / "lib" / "validate_run.py"
PYTHON = os.environ.get("BENCHMARK_PYTHON", sys.executable)


def option_value(args: list[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def selected_mode(args: list[str]) -> tuple[str, list[str]]:
    try:
        index = args.index("--benchmark_mode")
    except ValueError as error:
        raise ValueError("--benchmark_mode is required") from error
    if index + 1 >= len(args):
        raise ValueError("--benchmark_mode needs a value")
    mode = args[index + 1]
    if mode not in {"all", "no_save", "save", "check"}:
        raise ValueError("Usage: --benchmark_mode [all|no_save|save|check]")
    return mode, args[:index] + args[index + 2:]


def get_case(config_path: Path) -> dict:
    relative = config_path.resolve().relative_to(PACKAGE_ROOT).as_posix()
    try:
        return CASES[relative]
    except KeyError as error:
        raise ValueError(f"No MLU590 case metadata for {relative}") from error


def visible_devices(cards: int) -> list[str]:
    devices = os.environ.get("MLU_VISIBLE_DEVICES", "0").split(",")
    if len(devices) != cards or any(not item.isdigit() for item in devices):
        raise ValueError(f"This case requires {cards} MLU card IDs; got {','.join(devices)}")
    if len(set(devices)) != len(devices):
        raise ValueError(f"MLU_VISIBLE_DEVICES contains duplicates: {','.join(devices)}")
    return devices


def required_paths(case: dict, args: list[str], config_path: Path) -> list[Path]:
    paths = [
        Path(PYTHON),
        SOURCE_ROOT,
        Path(option_value(args, "--model_path") or case["model_path"]),
        config_path,
        PACKAGE_ROOT / case["entry_dir"] / case["entry_file"],
        config_path.parent / case["script"],
        VALIDATOR,
    ]
    for flag in ("--image_path", "--audio_path"):
        value = option_value(args, flag)
        if value:
            paths.append(Path(value))
    if case["case_kind"] == "paint":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("hy_repo", "mesh_path", "image_path",
                    "dino_ckpt_path", "realesrgan_ckpt_path"):
            value = config.get(key)
            if value:
                paths.append(Path(value))
    return paths


def check_case(case: dict, args: list[str], config_path: Path) -> None:
    visible_devices(case["cards"])
    for path in required_paths(case, args, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    json.loads(config_path.read_text(encoding="utf-8"))
    print(
        f"Checked: {case['case_id']} | cards={case['cards']} | "
        f"devices={os.environ['MLU_VISIBLE_DEVICES']} | config={config_path}"
    )


def set_environment(case: dict) -> dict[str, str]:
    environment = os.environ.copy()
    environment["LIGHTX2V_SOURCE"] = str(SOURCE_ROOT)
    environment.update({
        "PLATFORM": "cambricon_mlu",
        "DTYPE": case["dtype"],
        "SENSITIVE_LAYER_DTYPE": "None",
        "PROFILING_DEBUG_LEVEL": str(case["profiling_level"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    })
    if case["dtype"] == "BF16":
        environment.setdefault("PYTORCH_MLU_ALLOC_CONF", "expandable_segments:True")
    else:
        environment.setdefault("AI_DEVICE", "mlu")
    return environment


def record_cnmon(destination) -> None:
    if not shutil.which("cnmon"):
        destination.write("cnmon unavailable\n")
        return
    try:
        subprocess.run(["cnmon"], stdout=destination, stderr=subprocess.STDOUT,
                       timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        destination.write(f"cnmon unavailable: {error}\n")


def record_environment(path: Path, case: dict, environment: dict[str, str],
                       sources: list[tuple[str, Path]], model_path: str) -> None:
    source = Path(environment["LIGHTX2V_SOURCE"])
    marker = source / "SOURCE_COMMIT"
    commit = marker.read_text(encoding="utf-8").strip() if marker.is_file() else "unknown"
    if (source / ".git").exists():
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
    with path.open("w", encoding="utf-8") as record:
        record.write(f"recorded_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        record.write(f"hostname={socket.gethostname()}\n")
        record.write(f"kernel={os.uname().sysname} {os.uname().release}\n")
        record.write(f"case_id={case['case_id']}\n")
        record.write(f"card_count={case['cards']}\n")
        record.write(f"parallel_strategy={case['parallel_strategy']}\n")
        record.write(f"mlu_visible_devices={environment['MLU_VISIBLE_DEVICES']}\n")
        record.write(f"master_addr={environment.get('MASTER_ADDR', '127.0.0.1')}\n")
        record.write(f"master_port={environment.get('MASTER_PORT', '29587')}\n")
        record.write(f"python_bin={PYTHON}\npython_version={sys.version.split()[0]}\n")
        record.write(f"source_root={source}\nsource_commit={commit}\n")
        record.write(f"model_path={model_path}\ndtype={case['dtype']}\n")
        record.write(f"warmup_mode={case['warmup_mode']}\ncompile_kind={case['compile_kind']}\n")
        record.write(f"platform={environment['PLATFORM']}\n")
        record.write(f"profiling_debug_level={environment['PROFILING_DEBUG_LEVEL']}\n")
        for label, item in sources:
            record.write(f"{label}_sha256={hashlib.sha256(item.read_bytes()).hexdigest()}\n")
        for name in ("torch", "torch_mlu", "torch-mlu"):
            try:
                record.write(f"{name}={importlib.metadata.version(name)}\n")
            except importlib.metadata.PackageNotFoundError:
                pass
        record.write("\n[cnmon_before_run]\n")
        record_cnmon(record)


def mode_command(case: dict, cli_args: list[str], mode: str,
                 mode_dir: Path) -> list[str]:
    entry = mode_dir / case["entry_file"]
    config = mode_dir / "config.json"
    if case["case_kind"] == "paint":
        return [PYTHON, str(entry), "--config_json", str(config),
                "--model_path", option_value(cli_args, "--model_path") or case["model_path"],
                "--mode", mode, "--output_dir", str(mode_dir / "outputs")]

    args = cli_args.copy()
    config_index = args.index("--config_json") + 1
    args[config_index] = str(config)
    if mode == "no_save":
        args.append("--return_result_tensor")
    else:
        args.extend(("--save_result_path", str(mode_dir / "outputs" /
                                                f"request_1.{case['result_extension']}")))
    if case["entry_file"] == "infer_two_requests.py":
        command = [PYTHON, "-m", "infer_two_requests", *args]
    else:
        command = [PYTHON, str(entry), *args]
    if case["cards"] > 1:
        command = [
            PYTHON, "-m", "torch.distributed.run", "--nnodes=1",
            f"--nproc_per_node={case['cards']}",
            f"--master_addr={os.environ.get('MASTER_ADDR', '127.0.0.1')}",
            f"--master_port={os.environ.get('MASTER_PORT', '29587')}",
            f"--log-dir={mode_dir / 'rank_logs'}", "--redirects=3", *command[1:],
        ]
    return command


def run_mode(case: dict, cli_args: list[str], mode: str, run_root: Path,
             environment: dict[str, str], config_path: Path, script_path: Path,
             entry_source: Path) -> int:
    mode_dir = run_root / mode
    mode_dir.mkdir(parents=True)
    (mode_dir / "outputs").mkdir()
    shutil.copytree(entry_source, mode_dir, dirs_exist_ok=True)
    shutil.copy2(config_path, mode_dir / "config.json")
    shutil.copy2(script_path, mode_dir / script_path.name)
    runtime_dir = mode_dir / "_runtime"
    runtime_dir.mkdir()
    runtime_files = sorted(Path(__file__).parent.glob("*.py"))
    for source in runtime_files:
        shutil.copy2(source, runtime_dir / source.name)
    (mode_dir / "benchmark.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_environment = environment.copy()
    run_environment["PYTHONPATH"] = ":".join(
        [str(mode_dir), str(runtime_dir), run_environment.get("PYTHONPATH", "")]
    ).rstrip(":")
    sources = [("script", script_path), ("entry", entry_source / case["entry_file"]),
               ("config", config_path),
               ("benchmark_runner", Path(__file__)),
               ("case_metadata", Path(__file__).with_name("case_metadata.py")),
               ("validator", VALIDATOR)]
    sources.extend((f"runtime_{source.stem}", source) for source in runtime_files
                   if source.name not in {"benchmark_run.py", "case_metadata.py", "validate_run.py"})
    record_environment(mode_dir / "environment.txt", case, run_environment, sources,
                       option_value(cli_args, "--model_path") or case["model_path"])
    command = mode_command(case, cli_args, mode, mode_dir)
    started = datetime.now(timezone.utc).isoformat()
    with (mode_dir / "run.log").open("w", encoding="utf-8") as log:
        log.write(f"Case: {case['case_id']}\nMode: {mode}\n")
        log.write(f"MLU_VISIBLE_DEVICES: {run_environment['MLU_VISIBLE_DEVICES']}\n")
        log.write(f"Started at (UTC): {started}\nCommand: {shlex.join(command)}\n")
        log.flush()
        result = subprocess.run(command, cwd=mode_dir, env=run_environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
        ended = datetime.now(timezone.utc).isoformat()
        log.write(f"Ended at (UTC): {ended}\nExit code: {result.returncode}\n")

    validation = [
        PYTHON, str(VALIDATOR), "--run-dir", str(mode_dir),
        "--case-id", case["case_id"], "--mode", mode,
        "--card-count", str(case["cards"]), "--parallel-strategy", case["parallel_strategy"],
        "--warmup-mode", case["warmup_mode"], "--compile-kind", case["compile_kind"],
        "--case-kind", case["case_kind"], "--result-extension", case["result_extension"],
        "--expected-width", str(case["expected_width"]),
        "--expected-height", str(case["expected_height"]),
        "--expected-frames", str(case["expected_frames"]),
        "--exit-code", str(result.returncode), "--", *command,
    ]
    with (mode_dir / "validation.log").open("w", encoding="utf-8") as log:
        validation_result = subprocess.run(validation, cwd=mode_dir, env=run_environment,
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
    print((mode_dir / "validation.log").read_text(encoding="utf-8"), end="", flush=True)
    with (mode_dir / "device_after.txt").open("w", encoding="utf-8") as record:
        record_cnmon(record)
    (mode_dir / "status.txt").write_text(
        f"started_at_utc={started}\nended_at_utc={ended}\n"
        f"inference_exit_code={result.returncode}\n"
        f"validation_exit_code={validation_result.returncode}\n",
        encoding="utf-8",
    )
    return validation_result.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    mode, cli_args = selected_mode(args)
    config_value = option_value(cli_args, "--config_json")
    if not config_value:
        raise ValueError("--config_json is required")
    config_path = Path(config_value).resolve()
    case = get_case(config_path)
    os.environ.setdefault("MLU_VISIBLE_DEVICES", "0")
    os.environ["LIGHTX2V_SOURCE"] = str(SOURCE_ROOT)
    check_case(case, cli_args, config_path)
    if mode == "check":
        return 0

    script_path = config_path.parent / case["script"]
    entry_source = PACKAGE_ROOT / case["entry_dir"]
    environment = set_environment(case)
    run_root = Path(os.environ.get("BENCHMARK_OUTPUT_DIR") or (
        PACKAGE_ROOT / "runs" / Path(case["entry_dir"]).name /
        f"{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    )).resolve()
    modes = ("no_save", "save") if mode == "all" else (mode,)
    for selected in modes:
        status = run_mode(case, cli_args, selected, run_root, environment,
                          config_path, script_path, entry_source)
        if status:
            return status
    print(f"Logs and results: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
