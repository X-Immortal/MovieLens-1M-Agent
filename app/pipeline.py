#!/usr/bin/env python3
"""Orchestrate the real Hadoop Streaming quality and cleaning jobs."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import uuid
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/ml-1m"))
RUNTIME_DIR = Path(os.environ.get("RUNTIME_DIR", "/runtime"))
TABLES = ("users", "movies", "ratings")
TABLE_FILES = {table: DATA_DIR / f"{table}.dat" for table in TABLES}
T1_EPOCH = 1028504458
T2_EPOCH = 1037479524
T1_ISO = "2002-08-04T23:40:58Z"
T2_ISO = "2002-11-16T20:45:24Z"
RULE_VERSION = "default-v1"
SCORING_RULE_FINGERPRINT = hashlib.sha256(
    b"accurate=valid/total;complete=complete/total;unique=unique_records/total;"
    b"up_to_date=ratings.up_to_date/ratings.total;consistent=consistent/total;"
    b"tables=users,movies,ratings;T1=1028504458;T2=1037479524"
).hexdigest()
SCORING_METHOD = {
    "Accurate": "语义约束满足率；评分为 1-5、ID 为正整数、时间戳在 MovieLens 历史窗口内。",
    "Complete": "必需字段非空且字段数量正确的行比例。",
    "Unique": "业务键去重后的行数 / 总行数；ratings 业务键为 user、movie、timestamp。",
    "Up-to-date": "ratings 时间戳落在 2000-01-01 至 2004-11-09 历史窗口内的比例；T1/T2 仅用于区间计数。",
    "Consistent": "解析、类别取值和 ratings 跨表引用均满足规则的行比例。",
    "limitations": "格式和引用一致不代表用户自填属性现实真实；历史数据的时效性只能按固定窗口判断。",
}
ACTIVE_STATUS_PATH: Path | None = None


class PipelineError(RuntimeError):
    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


def _write_status(status_path: Path, status: str, **extra) -> None:
    payload = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with status_path.with_name("status_history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _streaming_jar() -> str:
    candidates = sorted(glob.glob("/opt/hadoop*/share/hadoop/tools/lib/hadoop-streaming-*.jar"))
    if not candidates:
        candidates = sorted(glob.glob("/opt/hadoop*/share/hadoop/tools/lib/*streaming*.jar"))
    if not candidates:
        raise PipelineError("容器内找不到 Hadoop Streaming jar")
    return candidates[-1]


def _run(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        process = subprocess.run(cmd, cwd="/app", stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
        if ACTIVE_STATUS_PATH:
            task_dir = ACTIVE_STATUS_PATH.parent
            task_id = task_dir.name
            _write_status(ACTIVE_STATUS_PATH, "failed", task_id=task_id, error=tail)
            failure_report = {
                "status": "failed", "task_id": task_id, "failure_code": "hadoop_job_failed",
                "error": tail, "execution": {"task_dir": str(task_dir), "logs": str(log_file.parent)},
            }
            (task_dir / "report.json").write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")
            raise PipelineError(f"Hadoop 命令失败（退出码 {process.returncode}）：\n{tail}", failure_report)
        raise PipelineError(f"Hadoop 命令失败（退出码 {process.returncode}）：\n{tail}")


def _streaming_job(
    name: str,
    input_path: Path,
    output_path: Path,
    mapper: str,
    reducer: str,
    log_dir: Path,
) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hadoop", "jar", _streaming_jar(),
        "-D", "fs.defaultFS=file:///",
        "-D", "mapreduce.framework.name=local",
        "-D", "mapreduce.job.reduces=1",
        "-D", "mapreduce.map.maxattempts=1",
        "-D", "mapreduce.reduce.maxattempts=1",
        "-input", f"file://{input_path}",
        "-output", f"file://{output_path}",
        "-mapper", mapper,
        "-reducer", reducer,
    ]
    _run(cmd, log_dir / f"{name}.log")


def _part_lines(output_path: Path):
    for part in sorted(output_path.glob("part-*")):
        with part.open(encoding="utf-8", errors="replace") as handle:
            yield from handle


def _quality_job(table: str, input_path: Path, output_path: Path, side_users: Path | None, side_movies: Path | None, log_dir: Path) -> dict:
    args = [
        "python3", "/app/hadoop_jobs/quality_mapper.py", "--table", table,
    ]
    if side_users:
        args += ["--users-file", str(side_users)]
    if side_movies:
        args += ["--movies-file", str(side_movies)]
    mapper = " ".join(args)
    reducer = f"python3 /app/hadoop_jobs/quality_reducer.py --table {table}"
    _streaming_job(f"quality_{table}_{output_path.parent.name}", input_path, output_path, mapper, reducer, log_dir)
    for line in _part_lines(output_path):
        key, _, value = line.rstrip("\r\n").partition("\t")
        if key == "summary":
            return json.loads(value)
    raise PipelineError(f"Hadoop 质量作业没有产生 summary：{table}")


def _clean_job(table: str, input_path: Path, output_path: Path, side_users: Path | None, side_movies: Path | None, log_dir: Path) -> dict:
    args = ["python3", "/app/hadoop_jobs/clean_mapper.py", "--table", table]
    if side_users:
        args += ["--users-file", str(side_users)]
    if side_movies:
        args += ["--movies-file", str(side_movies)]
    _streaming_job(
        f"clean_{table}", input_path, output_path,
        " ".join(args), "python3 /app/hadoop_jobs/clean_reducer.py", log_dir,
    )
    clean_file = output_path.parent / f"{table}.dat"
    reject_file = output_path.parent / f"{table}.rejected.jsonl"
    duplicate_file = output_path.parent / f"{table}.duplicates.dat"
    conflict_file = output_path.parent / f"{table}.conflicts.jsonl"
    counts = {"cleaned": 0, "repaired": 0, "rejected": 0, "duplicates": 0, "conflicts": 0}
    reject_reasons = Counter()
    reject_examples = defaultdict(list)
    duplicate_examples = []
    conflict_examples = []
    with clean_file.open("w", encoding="utf-8", newline="\n") as clean, \
         reject_file.open("w", encoding="utf-8", newline="\n") as rejects, \
         duplicate_file.open("w", encoding="utf-8", newline="\n") as duplicates, \
         conflict_file.open("w", encoding="utf-8", newline="\n") as conflicts:
        for line in _part_lines(output_path):
            prefix, _, value = line.rstrip("\r\n").partition("\t")
            if prefix == "DATA":
                clean.write(value + "\n")
                counts["cleaned"] += 1
            elif prefix == "REPAIRED":
                counts["repaired"] += int(value or 0)
            elif prefix == "REJECT":
                rejects.write(value + "\n")
                counts["rejected"] += 1
                try:
                    payload = json.loads(value)
                    reason = payload.get("reason", "unknown")
                    reject_reasons[reason] += 1
                    if len(reject_examples[reason]) < 3:
                        reject_examples[reason].append(payload.get("raw", ""))
                except json.JSONDecodeError:
                    reject_reasons["unparseable_reject_payload"] += 1
            elif prefix == "DUP":
                duplicates.write(value + "\n")
                counts["duplicates"] += 1
                if len(duplicate_examples) < 3:
                    duplicate_examples.append(value)
            elif prefix == "CONFLICT":
                conflicts.write(value + "\n")
                counts["conflicts"] += 1
                if len(conflict_examples) < 3:
                    conflict_examples.append(value)
    return {
        "table": table,
        **counts,
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
        "reject_examples": dict(sorted(reject_examples.items())),
        "duplicate_examples": duplicate_examples,
        "conflict_examples": conflict_examples,
        "clean_file": str(clean_file),
        "reject_file": str(reject_file),
        "conflict_file": str(conflict_file),
    }


def _score_hadoop(before: dict[str, dict], after: dict[str, dict], task_dir: Path, log_dir: Path) -> dict:
    input_path = task_dir / "score_input.tsv"
    with input_path.open("w", encoding="utf-8", newline="\n") as handle:
        for phase, summaries in (("before", before), ("after", after)):
            for item in summaries.values():
                handle.write(phase + "\t" + json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_path = task_dir / "quality_scores"
    _streaming_job("quality_scores", input_path, output_path,
                   "python3 /app/hadoop_jobs/score_mapper.py",
                   "python3 /app/hadoop_jobs/score_reducer.py", log_dir)
    for line in _part_lines(output_path):
        key, _, value = line.rstrip("\r\n").partition("\t")
        if key == "scores":
            return json.loads(value)
    raise PipelineError("Hadoop 质量评分作业没有产生 scores")


def _fixed_boundaries() -> dict:
    return {"T1": T1_ISO, "T2": T2_ISO, "T1_epoch": T1_EPOCH, "T2_epoch": T2_EPOCH,
            "basis": "Iteration 1 注册的固定时间边界；后续迭代必须复用"}


def _request_config(request: str) -> tuple[str, bool, str]:
    text = (request or "").strip()
    lowered = text.lower()
    if any(word in lowered for word in ("自定义", "非默认", "custom", "rule=")):
        return "", False, "未登记该自定义清洗规则；当前仅支持 default-v1（默认规则）"
    supported = (not text or "默认" in text or "default-v1" in lowered or "default" in lowered)
    if supported:
        return RULE_VERSION, True, "已匹配已登记的 default-v1 规则配置"
    return "", False, "未登记该自定义清洗规则；当前仅支持 default-v1（默认规则）"


def run_pipeline(request: str = "请使用默认规则清洗 MovieLens 1M，并评估清洗前后的五维质量") -> dict:
    global ACTIVE_STATUS_PATH
    if not all(path.exists() for path in TABLE_FILES.values()):
        missing = [str(path) for path in TABLE_FILES.values() if not path.exists()]
        raise PipelineError("缺少数据文件：" + ", ".join(missing))
    task_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    task_dir = RUNTIME_DIR / "runs" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    status_path = task_dir / "status.json"
    ACTIVE_STATUS_PATH = status_path
    _write_status(status_path, "running", task_id=task_id, request=request)
    rule_version, supported, resolution = _request_config(request)
    if not supported:
        report = {
            "status": "failed", "task_id": task_id, "request": request,
            "failure_code": "unsupported_rule", "error": resolution,
            "data_version": "ml-1m-local-raw-v1", "rule_version": None,
            "execution": {"task_dir": str(task_dir)},
        }
        (task_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_status(status_path, "failed", task_id=task_id, error=resolution, failure_code="unsupported_rule")
        ACTIVE_STATUS_PATH = None
        return report
    before_dir = task_dir / "quality_before"
    clean_dir = task_dir / "clean"
    after_dir = task_dir / "quality_after"
    log_dir = task_dir / "logs"
    for path in (before_dir, clean_dir, after_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_users, raw_movies = TABLE_FILES["users"], TABLE_FILES["movies"]
    _write_status(status_path, "running", task_id=task_id, stage="Hadoop 正在评价原始数据")
    before = {
        table: _quality_job(table, TABLE_FILES[table], before_dir / table, raw_users, raw_movies, log_dir)
        for table in TABLES
    }
    _write_status(status_path, "running", task_id=task_id, stage="Hadoop 正在清洗用户、电影和评分")
    users_clean = _clean_job("users", TABLE_FILES["users"], clean_dir / "users_job", None, None, log_dir)
    movies_clean = _clean_job("movies", TABLE_FILES["movies"], clean_dir / "movies_job", None, None, log_dir)
    ratings_clean = _clean_job(
        "ratings", TABLE_FILES["ratings"], clean_dir / "ratings_job",
        Path(users_clean["clean_file"]), Path(movies_clean["clean_file"]), log_dir,
    )
    cleaned_files = {table: Path(item["clean_file"]) for table, item in {
        "users": users_clean, "movies": movies_clean, "ratings": ratings_clean,
    }.items()}
    _write_status(status_path, "running", task_id=task_id, stage="Hadoop 正在评价清洗后数据")
    after = {
        table: _quality_job(
            table, cleaned_files[table], after_dir / table,
            cleaned_files["users"], cleaned_files["movies"], log_dir,
        ) for table in TABLES
    }
    cleaning = {"users": users_clean, "movies": movies_clean, "ratings": ratings_clean}
    _write_status(status_path, "running", task_id=task_id, stage="Hadoop 正在汇总五维质量评分")
    score = _score_hadoop(before, after, task_dir, log_dir)
    report = {
        "status": "completed",
        "task_id": task_id,
        "request": request,
        "data_version": "ml-1m-local-raw-v1",
        "cleaned_data_version": f"ml-1m-clean-v1-{task_id}",
        "rule_version": rule_version,
        "scoring_rule_fingerprint": SCORING_RULE_FINGERPRINT,
        "rule_resolution": resolution,
        "boundaries": _fixed_boundaries(),
        "execution": {
            "engine": "Hadoop Streaming",
            "framework": "local",
            "hadoop_streaming_jar": _streaming_jar(),
            "task_dir": str(task_dir),
            "logs": str(log_dir),
        },
        "quality": {
            "before": score["before"],
            "after": score["after"],
            "delta": score["delta"],
            "method": SCORING_METHOD,
            "summaries": {"before": before, "after": after},
        },
        "cleaning": cleaning,
        "time_split": {
            "source": "Hadoop quality summaries",
            "before": {k: before["ratings"][k] for k in ("train", "validation", "test")},
            "after": {k: after["ratings"][k] for k in ("train", "validation", "test")},
            "definition": "train <= T1; validation T1 < timestamp <= T2; test timestamp > T2 and within historical window",
        },
        "notes": [
            "评分和清洗文件均由本任务启动的 Hadoop Streaming mapper/reducer 产生。",
            "rejected 文件保留原始行及规则原因，不能把隔离行当作已修复。",
        ],
    }
    report_path = task_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUNTIME_DIR / "latest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(status_path, "completed", task_id=task_id, report=str(report_path))
    ACTIVE_STATUS_PATH = None
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default="请使用默认规则清洗 MovieLens 1M，并评估清洗前后的五维质量")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.request), ensure_ascii=False, indent=2))
