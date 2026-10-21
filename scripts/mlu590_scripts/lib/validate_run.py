#!/usr/bin/env python3
"""Check a completed MLU590 run and record its native second Pipeline time."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys


PIPELINE = re.compile(r"RUN pipeline cost ([0-9.]+) seconds")
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".webm", ".glb", ".obj"}
STAGES = ("after_warmup", "after_request_1", "after_request_2")


def one_line(log: str, marker: str) -> str:
    lines = [line for line in log.splitlines() if marker in line]
    if len(lines) != 1:
        raise ValueError(f"Expected one {marker!r} line, found {len(lines)}")
    return lines[0]


def pipeline_within(log: str, start: str, end: str) -> float:
    if log.count(start) != 1 or log.count(end) != 1:
        raise ValueError(f"Missing or repeated request markers: {start}, {end}")
    section = log.split(start, 1)[1].split(end, 1)[0]
    times = PIPELINE.findall(section)
    if len(times) != 1:
        raise ValueError(f"Expected one native Pipeline between {start} and {end}")
    return float(times[0])


def compile_evidence(log: str, kind: str) -> dict:
    if kind == "feature":
        rows = []
        for stage in STAGES:
            line = one_line(log, f"[FeatureCompile] evidence stage={stage} ")
            match = re.search(r"attempts=(\d+) completed=(\d+) graphs=(\d+) mlu_dtypes=(\[[^\]]*\])", line)
            if not match:
                raise ValueError(f"Incomplete feature compiler evidence at {stage}")
            attempts, completed, graphs = map(int, match.groups()[:3])
            dtypes = ast.literal_eval(match.group(4))
            rows.append((attempts, completed, graphs, dtypes))
        if any(a != c or g < 1 or "torch.bfloat16" not in d for a, c, g, d in rows):
            raise ValueError("Feature MLU graph did not complete in BF16")
        counts = [row[0] for row in rows]
        if not 0 < counts[0] < counts[1] < counts[2]:
            raise ValueError("Feature compile calls did not increase across requests")
        return {"kind": kind, "calls": counts, "graphs": rows[-1][2]}

    if kind == "qwen_gelu":
        calls = []
        graphs = []
        for stage in STAGES:
            line = one_line(log, f"[GELU-only] evidence stage={stage} ")
            states = ast.literal_eval(line.split("states=", 1)[1])
            calls.append(sum(state["graph_executions"] for state in states))
            graphs.append(sum(state["backend_graphs"] for state in states))
        if not 0 < calls[0] < calls[1] < calls[2] or min(graphs) < 1:
            raise ValueError("Qwen GELU Inductor graph did not execute in all stages")
        return {"kind": kind, "calls": calls, "graphs": graphs[-1]}

    if kind == "wan_gelu":
        calls = []
        graphs = []
        for stage in STAGES:
            line = one_line(log, f"compile evidence stage={stage} config_use_compile=True ")
            states_text = line.split("transformer_states=(name,class,use_compile,full_block_cache,gelu_cache,gelu_attempts,gelu_completed,inductor_graphs): ", 1)[1]
            states = ast.literal_eval(states_text.split(" graph_counters=", 1)[0])
            if not states or any(state[5] != state[6] for state in states):
                raise ValueError("Wan GELU compiled calls were incomplete")
            calls.append(sum(state[5] for state in states))
            graphs.append(sum(state[7] for state in states))
        if not 0 < calls[0] < calls[1] < calls[2] or min(graphs) < 1:
            raise ValueError("Wan GELU compiled graph did not execute in all stages")
        return {"kind": kind, "calls": calls, "graphs": graphs[-1]}

    if kind == "native_blocks":
        caches = []
        for stage in STAGES:
            line = one_line(log, f"compile evidence stage={stage} config_use_compile=True ")
            states = ast.literal_eval(line.split("transformer_states=", 1)[1])
            caches.append(sum(state[3] or 0 for state in states))
        if min(caches) < 1:
            raise ValueError("Native torch.compile block cache is empty")
        return {"kind": kind, "compiled_blocks": caches[-1]}

    if kind == "h3d":
        calls = []
        graphs = []
        for stage in STAGES:
            line = one_line(log, f"[H3DCompile] evidence stage={stage} ")
            match = re.search(
                r"attempts=(\d+) completed=(\d+) graphs=(\d+) graph_executions=(\d+) mlu_dtypes=(\[[^\]]*\])",
                line,
            )
            if not match:
                raise ValueError(f"Incomplete Hunyuan3D compiler evidence at {stage}")
            attempts, completed, graph_count, executions = map(int, match.groups()[:4])
            dtypes = ast.literal_eval(match.group(5))
            if attempts != completed or graph_count < 1 or "torch.float16" not in dtypes:
                raise ValueError("Hunyuan3D FP16 MLU graph did not complete")
            calls.append(executions)
            graphs.append(graph_count)
        if not 0 < calls[0] < calls[1] < calls[2]:
            raise ValueError("Hunyuan3D compiled graph calls did not increase")
        return {"kind": kind, "calls": calls, "graphs": graphs[-1]}

    raise ValueError(f"Unknown compile evidence type: {kind}")


def parse_rank_log(log: str, warmup_mode: str, compile_kind: str) -> dict:
    if warmup_mode == "manual":
        warmup = pipeline_within(log, "[Benchmark] warmup begin", "[Benchmark] warmup end")
    elif warmup_mode == "native":
        times = re.findall(r"\bWarmup cost ([0-9.]+) seconds", log)
        if len(times) != 1:
            raise ValueError("Expected one native Warmup cost")
        warmup = float(times[0])
    else:
        raise ValueError(f"Unknown warmup mode: {warmup_mode}")

    requests = [
        pipeline_within(log, f"[Benchmark] request {number} begin", f"[Benchmark] request {number} end")
        for number in (1, 2)
    ]
    one_line(log, "[Benchmark] between-request distributed_barrier() returned")
    return {
        "warmup_pipeline_s": warmup,
        "pipeline_s": requests,
        "compile": compile_evidence(log, compile_kind),
    }


def rank_logs(run_dir: Path, card_count: int) -> dict[str, Path]:
    if card_count == 1:
        return {"0": run_dir / "run.log"}
    found = {}
    for rank in range(card_count):
        matches = list((run_dir / "rank_logs").rglob(f"{rank}/stderr.log"))
        if len(matches) != 1:
            raise ValueError(f"Expected one native stderr.log for rank {rank}; found {len(matches)}")
        found[str(rank)] = matches[0]
    return found


def check_media(path: Path, args: argparse.Namespace) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Saved result missing or empty: {path}")
    details = {"path": str(path), "bytes": path.stat().st_size}
    if args.result_extension == "png":
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            details.update(width=image.width, height=image.height, frames=1)
    elif args.result_extension == "mp4":
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            details.update(width=stream.width, height=stream.height)
            details["frames"] = sum(1 for _ in container.decode(video=0))
    elif args.result_extension == "glb":
        import trimesh

        scene = trimesh.load(str(path), force="scene")
        if not scene.geometry or not any(len(mesh.faces) for mesh in scene.geometry.values()):
            raise ValueError(f"GLB contains no mesh faces: {path}")
        details["meshes"] = len(scene.geometry)
    else:
        raise ValueError(f"Unsupported result extension: {args.result_extension}")

    if args.result_extension != "glb":
        actual = details["width"], details["height"], details["frames"]
        expected = args.expected_width, args.expected_height, args.expected_frames
        if actual != expected:
            raise ValueError(f"Result dimensions/frames {actual} differ from {expected}: {path}")
    return details


def check_outputs(run_dir: Path, args: argparse.Namespace) -> list[dict]:
    if args.mode == "no_save":
        unexpected = [str(path) for path in run_dir.rglob("*")
                      if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES]
        if unexpected:
            raise ValueError(f"No-save mode wrote media files: {unexpected}")
        return []

    outputs = []
    for number in (1, 2):
        if args.case_kind == "paint":
            path = run_dir / "outputs" / f"request_{number}" / "textured.glb"
        else:
            path = run_dir / "outputs" / f"request_{number}.{args.result_extension}"
        outputs.append(check_media(path, args))
    return outputs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--record-file", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=("no_save", "save"), required=True)
    parser.add_argument("--card-count", type=int, required=True)
    parser.add_argument("--parallel-strategy", required=True)
    parser.add_argument("--warmup-mode", choices=("manual", "native"), required=True)
    parser.add_argument("--compile-kind", required=True)
    parser.add_argument("--case-kind", required=True)
    parser.add_argument("--result-extension", required=True)
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    record_file = args.record_file or run_dir / "run.json"
    record = {
        "case_id": args.case_id,
        "mode": args.mode,
        "card_count": args.card_count,
        "parallel_strategy": args.parallel_strategy,
        "status": "running",
        "command": args.command[1:] if args.command[:1] == ["--"] else args.command,
        "exit_code": args.exit_code,
        "config_sha256": digest(run_dir / "config.json"),
        "log_path": str(run_dir / "run.log"),
    }

    try:
        if args.exit_code != 0:
            raise RuntimeError(f"Inference process exited with code {args.exit_code}")
        logs = rank_logs(run_dir, args.card_count)
        record["rank_log_paths"] = {rank: str(path) for rank, path in logs.items()}
        record["pipeline_by_rank"] = {
            rank: parse_rank_log(path.read_text(encoding="utf-8", errors="replace"),
                                 args.warmup_mode, args.compile_kind)
            for rank, path in logs.items()
        }
        record["results"] = check_outputs(run_dir, args)
        record["second_pipeline_s"] = max(
            item["pipeline_s"][1] for item in record["pipeline_by_rank"].values()
        )
        record["warmup_actual"] = True
        record["compile_actual"] = True
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed_run" if args.exit_code else "failed_validation"
        record["error"] = str(error)

    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if record["status"] != "complete":
        print(f"{args.case_id} {args.mode}: {record['status']}: {record['error']}", file=sys.stderr)
        return 1
    print(f"{args.case_id} {args.mode}: second native Pipeline = {record['second_pipeline_s']:.6f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
