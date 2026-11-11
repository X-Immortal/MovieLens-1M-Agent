#!/usr/bin/env python3
"""Small Agent tool layer: Hadoop execution first, DeepSeek explanation second."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from pipeline import PipelineError, run_pipeline


class DeepSeekClient:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            return "未检测到 DEEPSEEK_API_KEY；Hadoop 结果已经生成，Agent 解释暂不可用。"
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return f"DeepSeek 调用失败（{type(exc).__name__}）；页面仍展示 Hadoop 产生的原始结果。"


class MovieLensAgent:
    def __init__(self):
        self.llm = DeepSeekClient()

    def run(self, request: str) -> dict:
        try:
            report = run_pipeline(request)
        except PipelineError as exc:
            if exc.report:
                return exc.report
            raise
        if report.get("status") != "completed":
            return report
        report["agent_explanation"] = self.explain(report, request)
        path = report["execution"]["task_dir"] + "/report.json"
        # Persist the final report with the explanation as the artifact shown by the UI.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        latest_path = os.path.join(os.environ.get("RUNTIME_DIR", "/runtime"), "latest_report.json")
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report

    def explain(self, report: dict, request: str) -> str:
        facts = json.dumps({
            "task_id": report["task_id"],
            "boundaries": report["boundaries"],
            "quality": report["quality"],
            "cleaning": report["cleaning"],
            "notes": report["notes"],
        }, ensure_ascii=False, indent=2)
        return self.llm.complete(
            "你是 MovieLens 数据治理 Agent。只能根据给定 Hadoop 事实回答，不得编造分数、记录或处理结果。明确区分修复、去重和隔离，并指出五维指标的局限。用中文给出简洁的演示说明。",
            f"用户请求：{request}\nHadoop 事实：\n{facts}",
        )

    def follow_up(self, report: dict, question: str) -> str:
        facts = json.dumps({
            "task_id": report["task_id"],
            "quality": report["quality"],
            "cleaning": report["cleaning"],
            "boundaries": report["boundaries"],
        }, ensure_ascii=False, indent=2)
        return self.llm.complete(
            "你是数据治理结果追问助手。只引用给定事实；如果事实中没有答案，直接说无法从本次 Hadoop 结果确认。",
            f"问题：{question}\n本次任务事实：\n{facts}",
        )


def run_request(request: str) -> dict:
    return MovieLensAgent().run(request)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?", default="请使用默认规则清洗 MovieLens 1M，并评估清洗前后的五维质量")
    args = parser.parse_args()
    try:
        print(json.dumps(run_request(args.request), ensure_ascii=False, indent=2))
    except PipelineError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
