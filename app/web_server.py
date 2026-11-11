#!/usr/bin/env python3
"""Small HTTP interface for the existing MovieLens agent and its artifacts."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from agent import MovieLensAgent
from ml_experiment import RUNTIME


WEB = Path(__file__).parent / "web"
SESSIONS = RUNTIME / "sessions"
SESSIONS.mkdir(parents=True, exist_ok=True)
POOL = ThreadPoolExecutor(max_workers=3)
LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(session: dict) -> None:
    path = SESSIONS / f"{session['id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def get_session(session_id: str) -> dict | None:
    if not SESSION_ID.fullmatch(session_id):
        return None
    path = SESSIONS / f"{session_id}.json"
    return read_json(path) if path.is_file() else None


def list_sessions() -> list[dict]:
    sessions = []
    for path in SESSIONS.glob("*.json"):
        try:
            session = read_json(path)
            sessions.append({key: session.get(key) for key in ("id", "title", "created_at")}
                            | {"message_count": len(session.get("messages", []))})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(sessions, key=lambda item: item.get("created_at") or "", reverse=True)


def current_reports() -> tuple[dict | None, dict | None, Path | None]:
    quality = None
    experiment = None
    directory = None
    try:
        quality = read_json(RUNTIME / "latest_report.json")
    except (OSError, ValueError):
        pass
    try:
        pointer = read_json(RUNTIME / "latest_experiment.json")
        candidate = Path(pointer["report"])
        if not candidate.is_file() and str(candidate).startswith("/runtime/"):
            candidate = RUNTIME / str(candidate).removeprefix("/runtime/")
        if candidate.resolve().is_relative_to(RUNTIME.resolve()):
            experiment = read_json(candidate)
            directory = candidate.parent
    except (OSError, ValueError, KeyError):
        pass
    return quality, experiment, directory


def artifacts() -> list[dict]:
    quality, experiment, directory = current_reports()
    rows: list[tuple[str, str, Path]] = []
    if quality:
        rows.append(("hadoop-report", "Hadoop 清洗报告", RUNTIME / "latest_report.json"))
        for name, label in (("users", "用户清洗数据"), ("movies", "电影清洗数据"), ("ratings", "评分清洗数据")):
            filename = quality.get("cleaning", {}).get(name, {}).get("clean_file")
            if filename:
                rows.append((f"clean-{name}", label, Path(filename)))
    if experiment and directory:
        rows.append(("experiment-report", "模型实验报告", directory / "report.json"))
        for filename, label in (("user_profiles.csv", "用户画像"),
                                ("kmeans_assignments.csv", "K-Means 用户分群"),
                                ("agglomerative_assignments.csv", "层次聚类用户分群"),
                                ("pca_coordinates.csv", "PCA 二维坐标"),
                                ("truncated_svd_coordinates.csv", "SVD 二维坐标")):
            rows.append((filename, label, directory / filename))
    result = []
    for artifact_id, label, path in rows:
        if not path.is_file() and str(path).startswith("/runtime/"):
            path = RUNTIME / str(path).removeprefix("/runtime/")
        if path.is_file() and path.resolve().is_relative_to(RUNTIME.resolve()):
            result.append({"id": artifact_id, "label": label, "filename": path.name,
                           "size": path.stat().st_size, "path": path})
    return result


def bootstrap() -> dict:
    quality, experiment, _ = current_reports()
    model_ready = bool(quality and experiment and
                       quality.get("cleaned_data_version") == experiment.get("cleaned_data_version") and
                       quality.get("rule_version") == experiment.get("rule_version") and
                       quality.get("boundaries") == experiment.get("boundaries"))
    return {"sessions": list_sessions(), "quality": {
                "task_id": quality.get("task_id"), "data_version": quality.get("cleaned_data_version"),
                "boundaries": {key: quality.get("boundaries", {}).get(key) for key in ("T1", "T2")},
                "split_counts": quality.get("time_split", {}).get("after")
            } if quality else None,
            "experiment": {"experiment_id": experiment.get("experiment_id"),
                           "components": experiment.get("components")
                           } if experiment else None,
            "model_ready": model_ready,
            "deepseek_configured": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
            "artifacts": [{key: row[key] for key in ("id", "label", "filename", "size")}
                          for row in artifacts()]}


def current_stage(job: dict) -> str:
    root = RUNTIME / ("runs" if job.get("is_clean") else "experiments")
    if not job.get("is_clean") and job.get("kind") != "train":
        return job["stage"]
    candidates = []
    for path in root.glob("*/status.json"):
        try:
            if path.stat().st_mtime >= job["started_at"] - 1:
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return job["stage"]
    try:
        return read_json(max(candidates, key=lambda path: path.stat().st_mtime)).get("stage") or job["stage"]
    except (OSError, ValueError):
        return job["stage"]


def do_agent(job_id: str, session_id: str, prompt: str, previous_task: str | None, previous_answer: dict | None) -> None:
    try:
        answer = MovieLensAgent().run(prompt, previous_task, previous_answer)
        with LOCK:
            session = get_session(session_id)
            if session is None:
                raise FileNotFoundError("会话文件不存在。")
            session["messages"].append({"role": "assistant", "answer": answer})
            save_session(session)
            JOBS[job_id].update(state="completed", answer=answer)
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(state="failed", error=str(exc))


def do_train(job_id: str) -> None:
    try:
        process = subprocess.run(["python3", "/app/ml_experiment.py", "train"],
                                 text=True, capture_output=True, check=False)
        with LOCK:
            JOBS[job_id].update(state="completed" if process.returncode == 0 else "failed",
                                error=process.stderr[-3000:] if process.returncode else None,
                                output=process.stdout[-3000:])
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(state="failed", error=str(exc))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        print("web:", format % args, flush=True)

    def respond(self, status: int, body: bytes, mime: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def json(self, status: int, payload: dict | list) -> None:
        self.respond(status, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                     "application/json; charset=utf-8")

    def file(self, path: Path, download: bool = False) -> None:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/health", "/_stcore/health"):
            self.respond(200, b"ok", "text/plain; charset=utf-8")
        elif path == "/api/bootstrap":
            self.json(200, bootstrap())
        elif path.startswith("/api/sessions/"):
            session = get_session(path.rsplit("/", 1)[-1])
            self.json(200, session) if session else self.json(404, {"error": "会话不存在。"})
        elif path.startswith("/api/jobs/"):
            with LOCK:
                job = JOBS.get(path.rsplit("/", 1)[-1])
                if job:
                    response = dict(job)
                    if job["state"] == "running":
                        response["stage"] = current_stage(job)
                    self.json(200, response)
                else:
                    self.json(404, {"error": "任务不存在。"})
        elif path == "/api/artifacts":
            self.json(200, [{key: row[key] for key in ("id", "label", "filename", "size")}
                            for row in artifacts()])
        elif path.startswith("/api/artifacts/"):
            selected = next((row for row in artifacts() if row["id"] == path.rsplit("/", 1)[-1]), None)
            self.file(selected["path"], True) if selected else self.json(404, {"error": "文件不存在。"})
        elif path == "/":
            self.file(WEB / "index.html")
        elif path in ("/styles.css", "/app.js"):
            self.file(WEB / path[1:])
        else:
            self.json(404, {"error": "页面不存在。"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16384:
                self.json(413, {"error": "请求内容过长。"})
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是对象。")
        except (ValueError, json.JSONDecodeError):
            self.json(400, {"error": "请求内容不是有效 JSON。"})
            return
        if path == "/api/sessions":
            session = {"id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                       "title": "新对话", "messages": []}
            with LOCK:
                save_session(session)
            self.json(201, session)
        elif path.startswith("/api/sessions/") and path.endswith("/messages"):
            session_id = path.split("/")[3]
            prompt = payload.get("content")
            if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 4000:
                self.json(400, {"error": "请输入 1 至 4000 字的分析问题。"})
                return
            with LOCK:
                session = get_session(session_id)
                if session is None:
                    self.json(404, {"error": "会话不存在。"})
                    return
                if any(job["state"] == "running" and job.get("session_id") == session_id
                       for job in JOBS.values()):
                    self.json(409, {"error": "此会话仍有分析任务在运行。"})
                    return
                prior = next((item["answer"] for item in reversed(session["messages"])
                              if item.get("role") == "assistant"), None)
                session["messages"].append({"role": "user", "content": prompt.strip()})
                if len(session["messages"]) == 1:
                    session["title"] = prompt.strip()[:28]
                save_session(session)
                job_id = uuid.uuid4().hex
                JOBS[job_id] = {"id": job_id, "state": "running", "kind": "agent",
                                "session_id": session_id, "stage": "正在调用 Agent 与分析工具",
                                "started_at": time.time(),
                                "is_clean": any(word in prompt for word in
                                                ("重新清洗", "执行清洗", "运行清洗", "运行 Hadoop", "开始清洗"))}
                POOL.submit(do_agent, job_id, session_id, prompt.strip(),
                            prior.get("task") if prior else None, prior)
            self.json(202, {"job_id": job_id})
        elif path == "/api/train":
            with LOCK:
                if any(job["state"] == "running" and job["kind"] == "train" for job in JOBS.values()):
                    self.json(409, {"error": "模型训练已在运行。"})
                    return
                job_id = uuid.uuid4().hex
                JOBS[job_id] = {"id": job_id, "state": "running", "kind": "train",
                                "stage": "正在训练六种分析模型", "started_at": time.time()}
                POOL.submit(do_train, job_id)
            self.json(202, {"job_id": job_id})
        else:
            self.json(404, {"error": "接口不存在。"})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8501), Handler).serve_forever()
