#!/usr/bin/env python3
"""Route natural-language requests to Hadoop and six persisted ML tools."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from ml_tools import TOOLS, ToolError, experiment, list_experiments
from ml_experiment import source_report
from pipeline import PipelineError, run_pipeline


class DeepSeekClient:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    def complete(self, system: str, user: str) -> str | None:
        if not self.api_key:
            return None
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        request = urllib.request.Request(url, data=body, method="POST",
                                         headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode())
            return payload["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
            return None


def _number(text: str, kind: str) -> int | None:
    pattern = (r"(?:UserID|用户(?:编号|ID)?)\s*(?:为|是|=|:|：|#)?\s*(\d+)" if kind == "user"
               else r"(?:MovieID|电影(?:编号|ID)?)\s*(?:为|是|=|:|：|#)?\s*(\d+)")
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def route(request: str, previous_task: str | None = None, llm: DeepSeekClient | None = None) -> dict:
    text = request.strip()
    lowered = text.lower()
    user_id, movie_id = _number(text, "user"), _number(text, "movie")
    user_list_match = re.search(
        r"(?:UserID|用户(?:编号|ID)?)\s*(?:为|是|=|:|：)?\s*((?:\d+\s*[,，、]\s*)+\d+)",
        text, re.IGNORECASE)
    user_ids = [int(value) for value in re.findall(r"\d+", user_list_match.group(1))] if user_list_match else None
    component_match = re.search(r"(?:目标|降到|压缩到|降至)\s*(\d+)\s*维", text)
    components = int(component_match.group(1)) if component_match else None
    cluster_match = re.search(r"(?:k\s*=\s*|分成\s*|划分为\s*)(\d+)\s*(?:群|类)?", text, re.IGNORECASE)
    cluster_count = int(cluster_match.group(1)) if cluster_match else None
    retention_match = re.search(
        r"(?:信息保留(?:率|阈值)?|解释方差(?:比例)?)\s*(?:至少|达到|不低于|>=|≥|为|到|超过)?\s*(\d+(?:\.\d+)?)\s*(%|％)?",
        text)
    retention_threshold = None
    if retention_match:
        retention_threshold = float(retention_match.group(1))
        if retention_match.group(2) or retention_threshold > 1:
            retention_threshold /= 100
        if not 0 <= retention_threshold <= 1:
            raise ToolError("信息保留阈值必须介于 0% 和 100% 之间。")
    names = []
    if "逻辑回归" in text or "logistic" in lowered:
        names.append("logistic_regression")
    if "决策树" in text or "decision tree" in lowered:
        names.append("decision_tree")
    if "k-means" in lowered or "kmeans" in lowered or "k 均值" in lowered:
        names.append("kmeans")
    if "层次聚类" in text or "agglomerative" in lowered:
        names.append("agglomerative")
    if re.search(r"\bpca\b", lowered) or "主成分" in text:
        names.append("pca")
    if "svd" in lowered or "奇异值" in text:
        names.append("truncated_svd")
    classification_names = [name for name in names if name in ("logistic_regression", "decision_tree")]
    clustering_names = [name for name in names if name in ("kmeans", "agglomerative")]
    reduction_names = [name for name in names if name in ("pca", "truncated_svd")]
    reduction_hint = any(word in text for word in ("降维", "二维", "坐标", "信息保留", "解释方差", "压缩"))
    comparison_tools = []
    if any(word in text for word in ("重新清洗", "执行清洗", "运行清洗", "运行 Hadoop", "开始清洗")):
        task = "clean"
    elif any(word in text for word in ("历史实验", "实验记录", "历史模型", "模型版本", "版本信息")):
        task = "history"
    elif (reduction_names or reduction_hint) and not classification_names:
        task = "reduction"
        comparison_tools = clustering_names or ["kmeans", "agglomerative"]
        names = reduction_names
    elif names:
        families = {"classification" if name in ("logistic_regression", "decision_tree") else
                    "clustering" if name in ("kmeans", "agglomerative") else "reduction" for name in names}
        if len(families) != 1:
            raise ToolError("一次请求请聚焦分类、聚类或降维中的一种任务。")
        task = families.pop()
    elif any(word in text for word in ("分类", "高分", "预测", "评分是否", "概率")) or (user_id is not None and movie_id is not None):
        task = "classification"
    elif any(word in text for word in ("聚类", "群体", "画像", "分群")):
        task = "clustering"
    elif any(word in text for word in ("降维", "二维", "坐标", "信息保留", "压缩")):
        task = "reduction"
    elif any(word in text for word in ("清洗", "质量", "五维", "数据版本", "报告")):
        task = "quality"
    elif previous_task and any(word in text for word in ("为什么", "依据", "局限", "再说", "解释", "指标", "特征", "参数", "阈值")):
        task = previous_task
    else:
        task = "unknown"
    routing_source = "rules"
    if task == "unknown" and llm and llm.api_key:
        response = llm.complete(
            '只返回 JSON：{"task":"classification|clustering|reduction|quality|unknown","user_id":整数或null,"movie_id":整数或null}。不得输出模型结果。',
            text)
        try:
            parsed = json.loads(response or "")
            if parsed.get("task") in ("classification", "clustering", "reduction", "quality"):
                task = parsed["task"]
                routing_source = "deepseek"
                user_id = user_id if user_id is not None else parsed.get("user_id")
                movie_id = movie_id if movie_id is not None else parsed.get("movie_id")
        except (TypeError, ValueError):
            pass
    defaults = {"classification": ["logistic_regression", "decision_tree"],
                "clustering": ["kmeans", "agglomerative"],
                "reduction": ["pca", "truncated_svd"]}
    if task == "classification" and (("UserID" in text and user_id is None) or ("MovieID" in text and movie_id is None)):
        raise ToolError("UserID 与 MovieID 必须填写正整数。")
    return {"task": task, "tools": names or defaults.get(task, []), "user_id": user_id,
            "user_ids": user_ids,
            "movie_id": movie_id, "components": components, "cluster_count": cluster_count,
            "retention_threshold": retention_threshold, "comparison_tools": comparison_tools,
            "routing_source": routing_source}


class MovieLensAgent:
    def __init__(self):
        self.llm = DeepSeekClient()

    def run(self, request: str, previous_task: str | None = None, previous_answer: dict | None = None) -> dict:
        try:
            if previous_answer and previous_answer.get("status") == "completed" and any(
                word in request for word in ("为什么", "依据", "局限", "解释", "再说")
            ) and not any(word in request for word in ("UserID", "MovieID", "用户", "电影", "聚类", "降维", "分类", "清洗")):
                answer = dict(previous_answer)
                answer["summary"] = "继续解释上一条已保存的工具结果。"
                answer["explanation"] = self.explain(previous_answer, request)
                return answer
            decision = route(request, previous_task, self.llm)
            task = decision["task"]
            if task == "unknown":
                raise ToolError("未识别任务。可以询问清洗质量、分类预测、用户聚类或降维结果。")
            if task == "clean":
                custom_rule = any(word in request.lower() for word in ("自定义", "非默认", "custom", "rule="))
                pipeline_request = request if custom_rule else "请使用默认规则清洗 MovieLens 1M，并评估清洗前后的五维质量"
                report = run_pipeline(pipeline_request)
                return {"status": report["status"], "task": "quality", "tools": ["hadoop_streaming"],
                        "results": [report], "summary": "Hadoop 清洗与评分已完成。" if report["status"] == "completed" else report.get("error", "Hadoop 任务失败。"),
                        "routing_source": decision["routing_source"]}
            if task == "quality":
                report = source_report()
                return {"status": "completed", "task": "quality", "tools": ["quality_report"],
                        "results": [report], "summary": "已读取真实 Hadoop 清洗报告。",
                        "data_version": report["cleaned_data_version"],
                        "routing_source": decision["routing_source"]}
            if task == "history":
                history = list_experiments()
                return {"status": "completed" if history else "failed", "task": "history",
                        "tools": ["experiment_history"], "results": history,
                        "summary": f"已读取 {len(history)} 次已完成实验。" if history else "尚无已完成实验。",
                        "routing_source": decision["routing_source"]}
            manifest, _ = experiment()
            if task == "classification" and (
                (decision["user_id"] is None) != (decision["movie_id"] is None) or
                ("预测" in request and (decision["user_id"] is None or decision["movie_id"] is None))
            ):
                raise ToolError("分类预测需要同时提供正整数 UserID 和 MovieID。")
            if task == "classification" and (decision["user_id"] is None or decision["movie_id"] is None):
                results = [{"tool": name, "feature_columns": manifest["classifiers"]["feature_columns"],
                            **manifest["classifiers"]["models"][name]} for name in decision["tools"]]
                summary = "已读取分类模型的验证集与测试集指标。"
            else:
                results = []
                for name in decision["tools"]:
                    fn = TOOLS[name]
                    if task == "classification":
                        result = fn(decision["user_id"], decision["movie_id"])
                    elif task == "clustering":
                        result = fn(None if decision["user_ids"] else decision["user_id"],
                                    decision["cluster_count"], decision["user_ids"])
                    else:
                        result = fn(decision["components"], decision["user_ids"] or
                                    ([decision["user_id"]] if decision["user_id"] is not None else None),
                                    decision["retention_threshold"], decision["comparison_tools"],
                                    decision["cluster_count"])
                    results.append(result)
                summary = f"已调用 {', '.join(decision['tools'])}，结果来自实验 {manifest['experiment_id']}。"
            answer = {"status": "completed", "task": task, "tools": decision["tools"],
                      "results": results, "summary": summary,
                      "experiment_id": manifest["experiment_id"], "data_version": manifest["cleaned_data_version"],
                      "routing_source": decision["routing_source"]}
            if any(word in request for word in ("为什么", "解释", "局限")):
                answer["explanation"] = self.explain(answer, request)
            return answer
        except (ToolError, PipelineError, FileNotFoundError, ValueError, KeyError) as exc:
            return {"status": "failed", "task": "unknown", "tools": [], "results": [], "error": str(exc)}

    def explain(self, answer: dict, question: str) -> str:
        task = answer.get("task")
        results = answer.get("results", [])
        if not results:
            return "本次没有可解释的工具结果。"
        if task == "quality":
            report = results[0]
            changes = "、".join(
                f"{name} {report['quality']['before'][name]:.2f}→{report['quality']['after'][name]:.2f}"
                for name in report["quality"]["before"])
            rejected = sum(report["cleaning"][table]["rejected"] for table in ("users", "movies", "ratings"))
            return f"Hadoop 同规则评价得到：{changes}。另有 {rejected} 条记录被隔离；隔离不等于修复。"
        if task == "classification":
            rows = []
            for item in results:
                metric = item.get("test_metrics", item.get("test", {}))
                probability = item.get("positive_probability")
                prediction = f"高分概率 {probability:.4f}；" if probability is not None else ""
                auc = f"{metric['roc_auc']:.4f}" if metric.get("roc_auc") is not None else "不适用"
                rows.append(f"{item['tool']}：{prediction}测试 F1 {metric['f1']:.4f}、ROC-AUC {auc}")
            return "；".join(rows) + "。测试指标来自 T2 之后，单次预测不能证明模型泛化能力。"
        if task == "clustering":
            return "；".join(
                f"{item['tool']} 将训练期 {item['users']} 名用户分为 {item['n_clusters']} 群，抽样轮廓系数 {item['silhouette']:.5f}"
                for item in results) + "。群体偏好是训练期统计描述，不能直接推断新用户。"
        if task == "reduction":
            return "；".join(
                f"{item['tool']} 将 {item['original_dimensions']} 维压缩为 {item['components']} 维，解释方差比例 {item['explained_variance_ratio_sum']:.5f}，重构均方误差 {item['reconstruction_mse']:.5f}"
                for item in results) + "。原空间与降维空间的轮廓系数使用不同距离空间。"
        if task == "history":
            return f"已登记 {len(results)} 次已完成实验；各实验的数据版本和参数以对应目录中的报告为准。"
        return "当前结果已保存在对应任务报告中。"


def run_request(request: str) -> dict:
    return MovieLensAgent().run(request)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?", default="请对照逻辑回归和决策树的分类测试指标")
    args = parser.parse_args()
    print(json.dumps(run_request(args.request), ensure_ascii=False, indent=2, default=str))
