"""Saved-model interfaces and validated MovieLens tool definitions."""
from __future__ import annotations

import json
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from jsonschema import Draft202012Validator, ValidationError
from ml_experiment import (
    RUNTIME,
    artifact_path,
    feature_frame,
    read_clean,
    source_report,
)
from pipeline import run_pipeline


class ToolError(ValueError):
    pass


def result_context(manifest: dict) -> dict:
    return {"experiment_id": manifest["experiment_id"], "data_version": manifest["cleaned_data_version"],
            "rule_version": manifest["rule_version"], "boundaries": manifest["boundaries"],
            "split_counts": manifest["classifiers"]["split_counts"]}


def experiment() -> tuple[dict, Path]:
    pointer = RUNTIME / "latest_experiment.json"
    if not pointer.is_file():
        raise ToolError("尚未训练迭代二模型。请先运行实验训练。")
    reference = json.loads(pointer.read_text(encoding="utf-8"))
    path = Path(reference["report"])
    if not path.exists() and str(path).startswith("/runtime/"):
        path = RUNTIME / str(path).removeprefix("/runtime/")
    if not path.is_file():
        raise ToolError("实验索引指向的报告不存在。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ToolError("迭代二实验尚未完成。")
    source = source_report()
    if (manifest["cleaned_data_version"] != source["cleaned_data_version"] or
            manifest["rule_version"] != source["rule_version"] or
            manifest["boundaries"] != source["boundaries"]):
        raise ToolError("模型与当前清洗数据版本、规则版本或 T1/T2 不一致；请重新训练或恢复对应数据报告。")
    return manifest, path.parent


@lru_cache(maxsize=2)
def _reference_tables(version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, users, movies = read_clean(source_report())
    return users, movies


@lru_cache(maxsize=2)
def _pair_stats(directory: str) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / "pair_snapshot_t1.csv").set_index(["UserID", "MovieID"])


def _classify(algorithm: str, user_id: int, movie_id: int) -> dict:
    manifest, directory = experiment()
    if not isinstance(user_id, int) or user_id <= 0 or not isinstance(movie_id, int) or movie_id <= 0:
        raise ToolError("UserID 和 MovieID 必须为正整数。")
    users, movies = _reference_tables(manifest["cleaned_data_version"])
    user = users.loc[users.UserID == user_id]
    movie = movies.loc[movies.MovieID == movie_id]
    if user.empty:
        raise ToolError(f"未知 UserID：{user_id}。")
    if movie.empty:
        raise ToolError(f"未知 MovieID：{movie_id}。")
    user_stats = pd.read_csv(directory / "user_snapshot_t1.csv")
    movie_stats = pd.read_csv(directory / "movie_snapshot_t1.csv")
    pair_table = _pair_stats(str(directory))
    pair_in_train = (user_id, movie_id) in pair_table.index
    if pair_in_train:
        pair = pair_table.loc[(user_id, movie_id)]
        for table, key, prefix in ((user_stats, "UserID", "user"), (movie_stats, "MovieID", "movie")):
            mask = table[key] == (user_id if key == "UserID" else movie_id)
            table.loc[mask, f"{prefix}_count"] -= int(pair.pair_count)
            table.loc[mask, f"{prefix}_sum"] -= int(pair.pair_sum)
    frame = feature_frame(pd.DataFrame([{"UserID": user_id, "MovieID": movie_id, "Timestamp": manifest["boundaries"]["T1_epoch"]}]),
                          users, movies, user_stats, movie_stats, prior=False)
    details = manifest["classifiers"]["models"][algorithm]
    model_file = directory / details["model_file"]
    if not model_file.exists():
        raise ToolError(f"{algorithm} 模型文件不存在。")
    model = joblib.load(model_file)
    probability = float(model.predict_proba(frame[manifest["classifiers"]["feature_columns"]])[0, 1])
    return {"status": "completed", "tool": algorithm, "UserID": user_id, "MovieID": movie_id,
            "movie_title": str(movie.iloc[0].title), "genres": str(movie.iloc[0].genres),
            "positive_probability": probability, "predicted_high_rating": probability >= details["threshold"],
            "threshold": details["threshold"], "parameter": details["parameter"],
            "estimator_params": details["estimator_params"],
            "test_metrics": details["test"], "validation_metrics": details["validation"],
            "feature_columns": manifest["classifiers"]["feature_columns"],
            "user_train_ratings": int(user_stats.loc[user_stats.UserID == user_id, "user_count"].sum()),
            "movie_train_ratings": int(movie_stats.loc[movie_stats.MovieID == movie_id, "movie_count"].sum()),
            "target_pair_in_training": bool(pair_in_train),
            "history_warning": "用户或电影在 T1 前少于 5 条其他评分，预测主要依赖静态属性。" if (
                int(user_stats.loc[user_stats.UserID == user_id, "user_count"].sum()) < 5 or
                int(movie_stats.loc[movie_stats.MovieID == movie_id, "movie_count"].sum()) < 5) else None,
            "history_policy": manifest["classifiers"]["history_policy"],
            "limitation": "本次是基于 T1 历史快照的情景预测；训练期内出现过的目标组合已从聚合特征扣除，但仍不能当作样本外准确率。测试集指标才是独立时间段评价。",
            **result_context(manifest)}


def logistic_regression(user_id: int, movie_id: int) -> dict:
    return _classify("logistic_regression", user_id, movie_id)


def decision_tree(user_id: int, movie_id: int) -> dict:
    return _classify("decision_tree", user_id, movie_id)


def _cluster(algorithm: str, user_id: int | None = None, n_clusters: int | None = None,
             user_ids: list[int] | None = None) -> dict:
    manifest, directory = experiment()
    details = manifest["analysis"]["clusters"][algorithm]
    if n_clusters is not None and not 2 <= n_clusters <= 12:
        raise ToolError("群体数量必须介于 2 和 12 之间。")
    if n_clusters is not None and n_clusters != details["n_clusters"]:
        raise ToolError(f"当前 {algorithm} 只保存 {details['n_clusters']} 群结果；可用 --clusters {n_clusters} 训练新实验。")
    assignments = pd.read_csv(directory / details["assignments_file"])
    result = {"status": "completed", "tool": algorithm, "n_clusters": details["n_clusters"],
              "silhouette": details["silhouette"], "candidate_silhouettes": details["candidate_silhouettes"],
              "seconds": details["seconds"], "groups": details["groups"],
              "estimator_params": details["estimator_params"],
              "users": len(assignments), **result_context(manifest),
              "profile_policy": manifest["analysis"]["profile_policy"]}
    profiles = pd.read_csv(directory / "user_profiles.csv")
    if user_id is not None:
        if not isinstance(user_id, int) or user_id <= 0:
            raise ToolError("UserID 必须为正整数。")
        selected = assignments.loc[assignments.UserID == user_id]
        if selected.empty:
            raise ToolError(f"UserID {user_id} 不在训练期画像中；无法直接推断新用户的群体。")
        result["UserID"] = user_id
        result["cluster"] = int(selected.iloc[0].cluster)
        rating_count = int(profiles.loc[profiles.UserID == user_id, "rating_count"].iloc[0])
        result["rating_count"] = rating_count
        if rating_count < 5:
            result["history_warning"] = f"UserID {user_id} 在训练期只有 {rating_count} 条评分，群体解释依据较少。"
    if user_ids is not None:
        if not user_ids or any(not isinstance(value, int) or value <= 0 for value in user_ids):
            raise ToolError("用户集合必须包含正整数 UserID。")
        selected = assignments[assignments.UserID.isin(user_ids)]
        missing = sorted(set(user_ids) - set(selected.UserID))
        if missing:
            raise ToolError(f"以下 UserID 不在训练期画像中：{missing}。")
        result["selected_users"] = selected.merge(profiles[["UserID", "rating_count"]], on="UserID").to_dict(orient="records")
        low_history = result["selected_users"]
        low_history = [row["UserID"] for row in low_history if row["rating_count"] < 5]
        if low_history:
            result["history_warning"] = f"训练期评分少于 5 条的用户：{low_history}；这些群体解释依据较少。"
        result["subset_note"] = "返回指定用户在全体训练期模型中的已有群体归属，不对该子集重新聚类。"
    profile_rows = profiles
    selected_ids = user_ids if user_ids is not None else [user_id] if user_id is not None else None
    if selected_ids is not None:
        profile_rows = profile_rows[profile_rows.UserID.isin(selected_ids)]
    result["profiles_sample"] = profile_rows.head(5).to_dict(orient="records")
    return result


def kmeans(user_id: int | None = None, n_clusters: int | None = None,
           user_ids: list[int] | None = None) -> dict:
    return _cluster("kmeans", user_id, n_clusters, user_ids)


def agglomerative(user_id: int | None = None, n_clusters: int | None = None,
                  user_ids: list[int] | None = None) -> dict:
    return _cluster("agglomerative", user_id, n_clusters, user_ids)


def _projection(algorithm: str, components: int | None = None, user_ids: list[int] | None = None,
                retention_threshold: float | None = None, comparison_tools: list[str] | None = None,
                n_clusters: int | None = None) -> dict:
    manifest, directory = experiment()
    details = manifest["analysis"]["projections"][algorithm]
    if components is not None and components < 2:
        raise ToolError("用于二维图的目标维数至少为 2。")
    if components is not None and components != details["components"]:
        raise ToolError(f"当前实验只保存 {details['components']} 维坐标；可用 --components {components} 重新训练。")
    if retention_threshold is not None and not 0 <= retention_threshold <= 1:
        raise ToolError("信息保留阈值必须介于 0 和 1 之间。")
    comparison_tools = comparison_tools or ["kmeans", "agglomerative"]
    if any(name not in details["comparison"] for name in comparison_tools):
        raise ToolError("请求的聚类对照算法没有已登记结果。")
    if n_clusters is not None:
        if not 2 <= n_clusters <= 12:
            raise ToolError("群体数量必须介于 2 和 12 之间。")
        for name in comparison_tools:
            actual = manifest["analysis"]["clusters"][name]["n_clusters"]
            if n_clusters != actual:
                raise ToolError(f"当前 {name} 的降维对照只保存 {actual} 群结果；可用 --clusters {n_clusters} 训练新实验。")
    coords = pd.read_csv(directory / details["coordinates_file"])
    selected_users = None
    if user_ids is not None:
        if not user_ids or any(not isinstance(value, int) or value <= 0 for value in user_ids):
            raise ToolError("用户集合必须包含正整数 UserID。")
        selected = coords[coords.UserID.isin(user_ids)]
        missing = sorted(set(user_ids) - set(selected.UserID))
        if missing:
            raise ToolError(f"以下 UserID 没有已登记的降维坐标：{missing}。")
        selected_users = selected.to_dict(orient="records")
    sample = coords.head(1000)
    if selected_users:
        sample = pd.concat([sample, coords[coords.UserID.isin(user_ids)]]).drop_duplicates("UserID")
    model = joblib.load(directory / f"{algorithm}.joblib")
    features = manifest["analysis"]["profile_features"]
    directions = []
    for index, weights in enumerate(model.components_):
        strongest = sorted(zip(features, weights), key=lambda pair: abs(pair[1]), reverse=True)[:5]
        directions.append({"component": index + 1,
                           "explained_variance_ratio": float(model.explained_variance_ratio_[index]),
                           "top_features": [{"feature": feature, "weight": float(weight)} for feature, weight in strongest]})
    result = {"status": "completed", "tool": algorithm, "original_dimensions": manifest["analysis"]["original_dimensions"],
              "components": details["components"], "explained_variance_ratio_sum": details["explained_variance_ratio_sum"],
              "reconstruction_mse": details["reconstruction_mse"], "transform_seconds": details["transform_seconds"],
              "peak_rss_increment_kib": details["peak_rss_increment_kib"], "coordinate_bytes": details["coordinate_bytes"],
              "input_matrix_bytes": details["input_matrix_bytes"],
              "comparison": {name: details["comparison"][name] for name in comparison_tools},
              "comparison_groups": {name: manifest["analysis"]["clusters"][name]["n_clusters"] for name in comparison_tools},
              "estimator_params": details["estimator_params"], "coordinate_count": len(coords),
              "coordinates_sample": sample.to_dict(orient="records"),
              "coordinates_file": str(directory / details["coordinates_file"]),
              "resource_note": manifest["analysis"]["resource_note"],
              "component_directions": directions,
              "direction_note": "权重对应标准化后的画像特征，绝对值越大，对该分量的贡献越大；正负表示方向，整轴符号翻转不改变几何关系。",
              **result_context(manifest)}
    if selected_users is not None:
        result["selected_users"] = selected_users
        result["subset_note"] = "坐标来自全体训练期模型；聚类对照指标仍基于全体训练期用户，不对所选用户重新拟合。"
    if retention_threshold is not None:
        result["retention_threshold"] = retention_threshold
        result["retention_met"] = details["explained_variance_ratio_sum"] >= retention_threshold
    return result


def pca(components: int | None = None, user_ids: list[int] | None = None,
        retention_threshold: float | None = None, comparison_tools: list[str] | None = None,
        n_clusters: int | None = None) -> dict:
    return _projection("pca", components, user_ids, retention_threshold, comparison_tools, n_clusters)


def truncated_svd(components: int | None = None, user_ids: list[int] | None = None,
                  retention_threshold: float | None = None, comparison_tools: list[str] | None = None,
                  n_clusters: int | None = None) -> dict:
    return _projection("truncated_svd", components, user_ids, retention_threshold, comparison_tools, n_clusters)


TOOLS = {"logistic_regression": logistic_regression, "decision_tree": decision_tree,
         "kmeans": kmeans, "agglomerative": agglomerative,
         "pca": pca, "truncated_svd": truncated_svd}


def list_experiments() -> list[dict]:
    history = []
    for path in sorted((RUNTIME / "experiments").glob("*/report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            history.append({
                "experiment_id": report["experiment_id"],
                "status": report["status"],
                "data_version": report["cleaned_data_version"],
                "T1": report["boundaries"]["T1"], "T2": report["boundaries"]["T2"],
                "logistic_test_f1": report["classifiers"]["models"]["logistic_regression"]["test"]["f1"],
                "tree_test_f1": report["classifiers"]["models"]["decision_tree"]["test"]["f1"],
                "kmeans_silhouette": report["analysis"]["clusters"]["kmeans"]["silhouette"],
                "agglomerative_silhouette": report["analysis"]["clusters"]["agglomerative"]["silhouette"],
                "components": report["components"],
                "scope": "最终实现" if (path.parent / "pair_snapshot_t1.csv").is_file() else "开发过程实验",
            })
        except (ValueError, KeyError, OSError):
            continue
    return history


Task = Literal["classification", "clustering", "reduction", "quality", "history", "mixed", "conversation"]
TOOL_TASKS: dict[str, Task] = {
    "logistic_regression": "classification", "decision_tree": "classification",
    "kmeans": "clustering", "agglomerative": "clustering",
    "pca": "reduction", "truncated_svd": "reduction",
    "quality_report": "quality", "hadoop_streaming": "quality", "experiment_history": "history",
}
USER_ID = {"type": "integer", "minimum": 1, "description": "MovieLens 用户编号。"}
USER_IDS = {"type": "array", "items": USER_ID, "minItems": 1, "uniqueItems": True,
            "description": "查询这些用户在已保存模型中的结果。"}
CLUSTERS = {"type": "integer", "minimum": 2, "maximum": 12,
            "description": "要求的群体数量；与已保存实验不一致时返回错误。省略则使用已保存值。"}
CLASSIFICATION_SCHEMA = {
    "type": "object", "properties": {
        "mode": {"type": "string", "enum": ["predict", "metrics"],
                 "description": "predict 预测指定用户和电影；metrics 只读取模型评价指标。"},
        "user_id": USER_ID,
        "movie_id": {"type": "integer", "minimum": 1, "description": "MovieLens 电影编号。"},
    }, "required": ["mode"], "additionalProperties": False,
    "if": {"properties": {"mode": {"const": "predict"}}},
    "then": {"required": ["user_id", "movie_id"]},
    "else": {"not": {"anyOf": [{"required": ["user_id"]}, {"required": ["movie_id"]}]}},
}
CLUSTERING_SCHEMA = {
    "type": "object", "properties": {"user_id": USER_ID, "user_ids": USER_IDS, "n_clusters": CLUSTERS},
    "additionalProperties": False, "not": {"required": ["user_id", "user_ids"]},
}
REDUCTION_SCHEMA = {
    "type": "object", "properties": {
        "components": {"type": "integer", "minimum": 2,
                       "description": "要求的目标维数。省略则读取已保存维数；本工具不会重新训练。"},
        "user_ids": USER_IDS,
        "retention_threshold": {"type": "number", "minimum": 0, "maximum": 1,
                                "description": "核对现有解释方差比例是否达标，例如 20% 填 0.2。"},
        "comparison_tools": {"type": "array", "minItems": 1, "uniqueItems": True,
                             "items": {"type": "string", "enum": ["kmeans", "agglomerative"]},
                             "description": "降维前后对照的聚类算法；省略时使用两种。"},
        "n_clusters": CLUSTERS,
    }, "additionalProperties": False,
}


def _definition(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOL_DEFINITIONS = [
    _definition("logistic_regression", "用已训练的逻辑回归预测评分是否至少为 4 分，或读取验证/测试指标。预测必须有用户和电影编号。", CLASSIFICATION_SCHEMA),
    _definition("decision_tree", "用已训练的决策树预测评分是否至少为 4 分，或读取验证/测试指标。预测必须有用户和电影编号。", CLASSIFICATION_SCHEMA),
    _definition("kmeans", "读取 K-Means 用户群体、偏好摘要及轮廓系数。可查询一个用户或用户集合；不重新聚类。", CLUSTERING_SCHEMA),
    _definition("agglomerative", "读取 Ward 层次聚类用户群体、偏好摘要及轮廓系数。可查询一个用户或用户集合；不推断新用户。", CLUSTERING_SCHEMA),
    _definition("pca", "读取 PCA 坐标、解释方差、重构误差、资源开销与降维前后聚类对照，可核对信息保留阈值。", REDUCTION_SCHEMA),
    _definition("truncated_svd", "读取 Truncated SVD 坐标、解释方差、重构误差、资源开销与降维前后聚类对照，可核对信息保留阈值。", REDUCTION_SCHEMA),
    _definition("quality_report", "读取已有 Hadoop 清洗质量报告。查询质量、清洗结果或解释报告时使用。", {"type": "object", "properties": {}, "additionalProperties": False}),
    _definition("experiment_history", "读取已登记的实验版本、参数与指标摘要。", {"type": "object", "properties": {}, "additionalProperties": False}),
    _definition("hadoop_streaming", "执行 Hadoop 清洗并更新当前数据版本。仅在用户明确要求执行清洗时使用；查询报告用 quality_report。仅支持默认规则。", {
        "type": "object", "properties": {"rule_version": {"type": "string", "enum": ["default-v1"]}},
        "required": ["rule_version"], "additionalProperties": False,
    }),
]
TOOL_VALIDATORS = {item["function"]["name"]: Draft202012Validator(item["function"]["parameters"])
                   for item in TOOL_DEFINITIONS}


def execute_tool(name: str, arguments: dict) -> dict:
    """Validate model arguments before invoking an explicitly registered function."""
    if name not in TOOL_VALIDATORS:
        raise ToolError(f"未登记工具：{name}。")
    try:
        TOOL_VALIDATORS[name].validate(arguments)
        json.dumps(arguments, allow_nan=False)
    except ValidationError as exc:
        field = ".".join(str(part) for part in exc.absolute_path) or "参数"
        raise ToolError(f"{name} 的 {field} 无效：{exc.message}") from exc
    if name in ("logistic_regression", "decision_tree"):
        if arguments["mode"] == "metrics":
            manifest, _ = experiment()
            result = {**manifest["classifiers"]["models"][name],
                      "feature_columns": manifest["classifiers"]["feature_columns"],
                      **result_context(manifest)}
        else:
            result = TOOLS[name](arguments["user_id"], arguments["movie_id"])
    elif name in TOOLS:
        result = TOOLS[name](**arguments)
        if TOOL_TASKS[name] == "reduction":
            result["coordinate_label_policy"] = "坐标表的 kmeans/agglomerative 标签来自原始画像空间。降维空间仅提供整体聚类对照指标，未返回逐用户标签。"
            result["variance_policy"] = "explained_variance_ratio_sum 是保留的方差比例；原空间与降维空间的轮廓系数分别按各自距离计算。"
    elif name == "quality_report":
        result = source_report()
    elif name == "experiment_history":
        result = {"experiments": list_experiments()}
    else:
        result = run_pipeline(f"使用 {arguments['rule_version']} 默认规则清洗 MovieLens 1M 并评估五维质量")
    if TOOL_TASKS[name] == "quality" and result.get("status") == "completed":
        samples = {}
        for table, details in result["cleaning"].items():
            with artifact_path(details["clean_file"]).open(encoding="utf-8") as stream:
                samples[table] = [line.rstrip("\n") for line in islice(stream, 5)]
        result = {**result, "clean_samples": samples}
    if result.get("cleaned_data_version"):
        result = {**result, "data_version": result["cleaned_data_version"]}
    return {"status": "completed", **result, "tool": name, "task": TOOL_TASKS[name]}
