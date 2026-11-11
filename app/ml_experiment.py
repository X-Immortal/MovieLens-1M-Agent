#!/usr/bin/env python3
"""Reproducible iteration-two experiments on an immutable Hadoop clean run."""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RUNTIME = Path(os.environ.get("RUNTIME_DIR", "/runtime"))
NUMERIC = ["release_year", "user_count", "user_mean", "movie_count", "movie_mean"]
CATEGORICAL = ["gender", "age", "occupation"]
SEED = 20261020


def artifact_path(value: str) -> Path:
    """Map the container paths in an iteration-one report to this runtime mount."""
    path = Path(value)
    if path.exists():
        return path
    if value.startswith("/runtime/"):
        return RUNTIME / value.removeprefix("/runtime/")
    return path


def source_report() -> dict:
    path = RUNTIME / "latest_report.json"
    if not path.exists():
        raise FileNotFoundError("缺少迭代一报告 runtime/latest_report.json；请先运行 Hadoop 清洗。")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("最新 Hadoop 报告未完成；请选择已完成的清洗报告。")
    for table in ("ratings", "users", "movies"):
        if not artifact_path(report["cleaning"][table]["clean_file"]).is_file():
            raise FileNotFoundError(f"缺少清洗后的 {table}.dat")
    boundaries = report.get("boundaries", {})
    if not boundaries.get("T1_epoch") or not boundaries.get("T2_epoch"):
        raise ValueError("迭代一报告没有登记 T1/T2。")
    return report


def read_clean(report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def file(table: str) -> Path:
        return artifact_path(report["cleaning"][table]["clean_file"])

    ratings = pd.read_csv(file("ratings"), sep="::", engine="python", header=None,
                          names=["UserID", "MovieID", "Rating", "Timestamp"],
                          dtype=np.int64)
    users = pd.read_csv(file("users"), sep="::", engine="python", header=None,
                        names=["UserID", "gender", "age", "occupation", "zip"],
                        dtype=str, encoding="utf-8", encoding_errors="replace")
    users["UserID"] = users["UserID"].astype(np.int64)
    movies = pd.read_csv(file("movies"), sep="::", engine="python", header=None,
                         names=["MovieID", "title", "genres"], dtype=str,
                         encoding="utf-8", encoding_errors="replace")
    movies["MovieID"] = movies["MovieID"].astype(np.int64)
    genres = sorted({genre for value in movies["genres"] for genre in value.split("|")})
    for genre in genres:
        movies[f"genre:{genre}"] = movies["genres"].str.split("|").map(lambda parts: int(genre in parts))
    movies["release_year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)$")[0], errors="coerce").fillna(0)
    expected = report["quality"]["summaries"]["after"]
    for table, frame in (("ratings", ratings), ("users", users), ("movies", movies)):
        if len(frame) != expected[table]["total"]:
            raise ValueError(f"{table} 行数与已登记清洗报告不一致；停止训练以避免混用数据版本。")
    return ratings, users, movies


def split_ratings(ratings: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t1 = report["boundaries"]["T1_epoch"]
    t2 = report["boundaries"]["T2_epoch"]
    parts = (ratings[ratings.Timestamp <= t1].copy(),
             ratings[(ratings.Timestamp > t1) & (ratings.Timestamp <= t2)].copy(),
             ratings[ratings.Timestamp > t2].copy())
    actual = dict(zip(("train", "validation", "test"), map(len, parts)))
    if actual != report["time_split"]["after"]:
        raise ValueError("T1/T2 切分计数与迭代一报告不一致。")
    return parts


def strict_prior(train: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Aggregate equal-timestamp events together so no peer event leaks into history."""
    grouped = train.groupby([key, "Timestamp"], sort=True).Rating.agg(["size", "sum"]).reset_index()
    grouped[f"{prefix}_count"] = grouped.groupby(key)["size"].cumsum() - grouped["size"]
    grouped[f"{prefix}_sum"] = grouped.groupby(key)["sum"].cumsum() - grouped["sum"]
    return grouped[[key, "Timestamp", f"{prefix}_count", f"{prefix}_sum"]]


def snapshot(train: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    result = train.groupby(key).Rating.agg(["size", "sum"]).reset_index()
    return result.rename(columns={"size": f"{prefix}_count", "sum": f"{prefix}_sum"})


def feature_frame(events: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame,
                  user_stats: pd.DataFrame, movie_stats: pd.DataFrame, *, prior: bool) -> pd.DataFrame:
    frame = events[["UserID", "MovieID", "Timestamp"]].copy()
    frame = frame.merge(users[["UserID", *CATEGORICAL]], on="UserID", how="left", validate="many_to_one")
    movie_cols = [col for col in movies if col.startswith("genre:")]
    frame = frame.merge(movies[["MovieID", "release_year", *movie_cols]], on="MovieID", how="left", validate="many_to_one")
    for key, prefix, stats in (("UserID", "user", user_stats), ("MovieID", "movie", movie_stats)):
        join = [key, "Timestamp"] if prior else [key]
        frame = frame.merge(stats, on=join, how="left", validate="many_to_one")
        count = frame[f"{prefix}_count"].fillna(0).astype(np.float32)
        frame[f"{prefix}_mean"] = (frame[f"{prefix}_sum"].fillna(0) / count.replace(0, np.nan)).fillna(3.5).astype(np.float32)
        frame[f"{prefix}_count"] = np.log1p(count).astype(np.float32)
        frame = frame.drop(columns=f"{prefix}_sum")
    if frame[[*CATEGORICAL, "release_year"]].isna().any().any():
        raise ValueError("清洗数据包含无法关联的用户或电影。")
    return frame[[*NUMERIC, *CATEGORICAL, *movie_cols]]


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else None,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "samples": int(len(y)),
    }


def train_classifiers(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                      users: pd.DataFrame, movies: pd.DataFrame, output: Path) -> dict:
    start = time.perf_counter()
    train_features = feature_frame(train, users, movies,
                                   strict_prior(train, "UserID", "user"),
                                   strict_prior(train, "MovieID", "movie"), prior=True)
    user_at_t1 = snapshot(train, "UserID", "user")
    movie_at_t1 = snapshot(train, "MovieID", "movie")
    user_at_t1.to_csv(output / "user_snapshot_t1.csv", index=False)
    movie_at_t1.to_csv(output / "movie_snapshot_t1.csv", index=False)
    pair_at_t1 = train.groupby(["UserID", "MovieID"]).Rating.agg(["size", "sum"]).reset_index()
    pair_at_t1.rename(columns={"size": "pair_count", "sum": "pair_sum"}).to_csv(output / "pair_snapshot_t1.csv", index=False)
    valid_features = feature_frame(valid, users, movies, user_at_t1, movie_at_t1, prior=False)
    test_features = feature_frame(test, users, movies, user_at_t1, movie_at_t1, prior=False)
    labels = [(part.Rating.to_numpy() >= 4).astype(np.int8) for part in (train, valid, test)]
    genre_cols = [col for col in train_features if col.startswith("genre:")]
    input_cols = [*NUMERIC, *CATEGORICAL, *genre_cols]

    def prep() -> ColumnTransformer:
        return ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("genre", "passthrough", genre_cols),
        ], sparse_threshold=0.3)

    specs = {
        "logistic_regression": [("C=0.1", LogisticRegression(C=0.1, solver="liblinear", max_iter=150, random_state=SEED)),
                                ("C=1", LogisticRegression(C=1, solver="liblinear", max_iter=150, random_state=SEED))],
        "decision_tree": [("max_depth=8", DecisionTreeClassifier(max_depth=8, min_samples_leaf=80, random_state=SEED)),
                          ("max_depth=14", DecisionTreeClassifier(max_depth=14, min_samples_leaf=80, random_state=SEED))],
    }
    results = {}
    for name, candidates in specs.items():
        best = None
        for parameter, estimator in candidates:
            model = Pipeline([("preprocessor", prep()), ("classifier", estimator)])
            model.fit(train_features[input_cols], labels[0])
            probabilities = model.predict_proba(valid_features[input_cols])[:, 1]
            for threshold in np.arange(0.30, 0.71, 0.05):
                metric = classification_metrics(labels[1], probabilities, float(threshold))
                score = (metric["f1"], metric["roc_auc"] or 0)
                if best is None or score > best[0]:
                    best = (score, parameter, float(round(threshold, 2)), model, metric)
        assert best is not None
        _, parameter, threshold, model, validation = best
        test_probability = model.predict_proba(test_features[input_cols])[:, 1]
        test_metric = classification_metrics(labels[2], test_probability, threshold)
        joblib.dump(model, output / f"{name}.joblib", compress=3)
        results[name] = {"parameter": parameter, "threshold": threshold,
                         "estimator_params": model.named_steps["classifier"].get_params(deep=False),
                         "validation": validation, "test": test_metric,
                         "model_file": f"{name}.joblib"}
    return {"models": results, "feature_columns": input_cols,
            "history_policy": "训练事件只用严格早于事件时间戳的评分，同秒事件一起排除。验证与测试使用 T1 训练期快照；交互预测若目标组合曾在训练期评分，先从 T1 聚合快照中扣除该组合全部评分。",
            "split_counts": dict(zip(("train", "validation", "test"), map(len, (train, valid, test)))),
            "seconds": round(time.perf_counter() - start, 3)}


def user_profiles(train: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    movie_genres = movies[["MovieID", "genres"]].copy()
    joined = train[["UserID", "MovieID", "Rating"]].merge(movie_genres, on="MovieID", how="left")
    overall = joined.groupby("UserID").Rating.agg(["mean", "size"]).rename(columns={"mean": "overall_mean", "size": "rating_count"})
    exploded = joined.assign(genre=joined.genres.str.split("|")).explode("genre")
    exploded = exploded.merge(overall[["overall_mean"]], left_on="UserID", right_index=True)
    exploded["deviation"] = exploded.Rating - exploded.overall_mean
    grouped = exploded.groupby(["UserID", "genre"]).agg(count=("Rating", "size"), deviation=("deviation", "sum"))
    genres = sorted(exploded.genre.dropna().unique())
    profile = overall.copy()
    profile["log_rating_count"] = np.log1p(profile.rating_count)
    features = ["overall_mean", "log_rating_count"]
    for genre in genres:
        per_genre = grouped.xs(genre, level="genre") if genre in grouped.index.get_level_values("genre") else pd.DataFrame()
        count = per_genre["count"].reindex(profile.index).fillna(0) if not per_genre.empty else pd.Series(0, index=profile.index)
        deviation = per_genre["deviation"].reindex(profile.index).fillna(0) if not per_genre.empty else pd.Series(0, index=profile.index)
        preference = f"preference:{genre}"
        frequency = f"frequency:{genre}"
        profile[preference] = deviation / (count + 5)  # unseen -> neutral, not disliked
        profile[frequency] = count / profile.rating_count
        features.extend((preference, frequency))
    profile.index.name = "UserID"
    return profile.reset_index(), features


def silhouette_sample(x: np.ndarray, labels: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(x), size=min(1800, len(x)), replace=False)
    if len(np.unique(labels[indices])) < 2:
        return float("nan")
    return float(silhouette_score(x[indices], labels[indices]))


def cluster_summary(profile: pd.DataFrame, x: np.ndarray, labels: np.ndarray) -> list[dict]:
    grouped = profile.assign(cluster=labels)
    result = []
    for cluster, group in grouped.groupby("cluster"):
        positions = np.flatnonzero(labels == cluster)
        center = x[positions].mean(axis=0)
        distance = np.sum((x[positions] - center) ** 2, axis=1)
        representative_ids = profile.iloc[positions[np.argsort(distance)[:3]]].UserID.astype(int).tolist()
        preferences = {key.removeprefix("preference:"): round(float(group[key].mean() - profile[key].mean()), 3)
                       for key in profile if key.startswith("preference:")}
        top = sorted(preferences.items(), key=lambda pair: pair[1], reverse=True)[:3]
        result.append({"cluster": int(cluster), "users": int(len(group)),
                       "mean_rating": round(float(group.overall_mean.mean()), 3),
                       "median_rating_count": int(group.rating_count.median()),
                       "preferred_genres": top,
                       "representative_user_ids": representative_ids,
                       "explanation": "、".join(name for name, value in top if value > 0) or "类型偏好与总体接近"})
    return result


def train_segments(train: pd.DataFrame, movies: pd.DataFrame, output: Path, components: int,
                   cluster_k: int | None = None) -> dict:
    profile, feature_cols = user_profiles(train, movies)
    profile.to_csv(output / "user_profiles.csv", index=False)
    scaler = StandardScaler()
    x = scaler.fit_transform(profile[feature_cols].to_numpy(dtype=np.float32))
    joblib.dump(scaler, output / "profile_scaler.joblib")
    if not 2 <= components < x.shape[1]:
        raise ValueError(f"目标维数必须介于 2 和 {x.shape[1] - 1} 之间。")
    if cluster_k is not None and not 2 <= cluster_k <= min(12, len(x) - 1):
        raise ValueError("群体数量必须介于 2 和 12 之间，且小于用户数。")
    clusters = {}
    best_k = {}
    for name in ("kmeans", "agglomerative"):
        candidates = []
        for k in ((cluster_k,) if cluster_k is not None else (4, 6, 8)):
            before = time.perf_counter()
            model = (KMeans(n_clusters=k, n_init=10, random_state=SEED)
                     if name == "kmeans" else AgglomerativeClustering(n_clusters=k, linkage="ward"))
            labels = model.fit_predict(x)
            score = silhouette_sample(x, labels)
            candidates.append((score, k, model, labels, time.perf_counter() - before))
        score, k, model, labels, seconds = max(candidates, key=lambda row: row[0])
        best_k[name] = k
        joblib.dump(model, output / f"{name}.joblib", compress=3)
        profile[["UserID"]].assign(cluster=labels).to_csv(output / f"{name}_assignments.csv", index=False)
        clusters[name] = {"n_clusters": k, "silhouette": round(score, 5),
                          "estimator_params": model.get_params(deep=False),
                          "candidate_silhouettes": {str(row[1]): round(row[0], 5) for row in candidates},
                          "seconds": round(seconds, 3), "groups": cluster_summary(profile, x, labels),
                          "assignments_file": f"{name}_assignments.csv",
                          "new_user_policy": "仅登记训练期已有用户；层次聚类不支持直接预测新用户。"}
    projections = {}
    for name in ("pca", "truncated_svd"):
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        before = time.perf_counter()
        model = PCA(n_components=components, random_state=SEED) if name == "pca" else TruncatedSVD(n_components=components, random_state=SEED)
        coordinates = model.fit_transform(x)
        projection_seconds = time.perf_counter() - before
        joblib.dump(model, output / f"{name}.joblib", compress=3)
        coordinate_frame = pd.DataFrame({"UserID": profile.UserID, **{f"component_{i + 1}": coordinates[:, i] for i in range(components)}})
        for cluster_name in clusters:
            coordinate_frame[cluster_name] = pd.read_csv(output / clusters[cluster_name]["assignments_file"]).cluster
        coordinate_frame.to_csv(output / f"{name}_coordinates.csv", index=False)
        reconstructed = model.inverse_transform(coordinates)
        reduction = {}
        for cluster_name, k in best_k.items():
            before_cluster = time.perf_counter()
            reduced_model = (KMeans(n_clusters=k, n_init=10, random_state=SEED)
                             if cluster_name == "kmeans" else AgglomerativeClustering(n_clusters=k, linkage="ward"))
            reduced_labels = reduced_model.fit_predict(coordinates)
            reduction[cluster_name] = {"original_silhouette": clusters[cluster_name]["silhouette"],
                                       "reduced_silhouette": round(silhouette_sample(coordinates, reduced_labels), 5),
                                       "original_cluster_seconds": clusters[cluster_name]["seconds"],
                                       "reduced_cluster_seconds": round(time.perf_counter() - before_cluster, 3),
                                       "comparison_note": "原始与降维后使用同一聚类算法、相同用户和群体数量；轮廓系数分别在各自特征空间计算。"}
        projections[name] = {"components": components,
                             "estimator_params": model.get_params(deep=False),
                             "explained_variance_ratio_sum": round(float(np.sum(model.explained_variance_ratio_)), 5),
                             "reconstruction_mse": round(float(np.mean((x - reconstructed) ** 2)), 5),
                             "transform_seconds": round(projection_seconds, 3),
                             "peak_rss_increment_kib": max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before),
                             "input_matrix_bytes": int(x.nbytes),
                             "coordinate_bytes": int(coordinates.nbytes),
                             "comparison": reduction, "coordinates_file": f"{name}_coordinates.csv"}
    return {"users": len(profile), "original_dimensions": x.shape[1], "profile_features": feature_cols,
            "profile_policy": "仅使用 T1 前评分；未看过的类型偏好记为相对个人平均分的中性偏差 0，频率为 0；偏差使用 5 条平滑。",
            "clusters": clusters, "projections": projections,
            "resource_note": "input_matrix_bytes 与 coordinate_bytes 仅衡量 NumPy 特征矩阵和坐标数组；peak_rss_increment_kib 为进程历史峰值增长，可能因前序步骤已分配内存而为 0，不代表降维无需内存。"}


def run_experiment(components: int = 2, cluster_k: int | None = None) -> dict:
    report = source_report()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(4).hex()
    output = RUNTIME / "experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "status.json").write_text(json.dumps({"status": "running", "stage": "读取清洗数据"}), encoding="utf-8")
    try:
        ratings, users, movies = read_clean(report)
        train, valid, test = split_ratings(ratings, report)
        (output / "status.json").write_text(json.dumps({"status": "running", "stage": "训练两种分类模型"}), encoding="utf-8")
        classifiers = train_classifiers(train, valid, test, users, movies, output)
        (output / "status.json").write_text(json.dumps({"status": "running", "stage": "训练聚类与降维模型"}), encoding="utf-8")
        segments = train_segments(train, movies, output, components, cluster_k)
        manifest = {"status": "completed", "experiment_id": run_id,
                    "source_task_id": report["task_id"], "cleaned_data_version": report["cleaned_data_version"],
                    "rule_version": report["rule_version"], "boundaries": report["boundaries"],
                    "seed": SEED, "components": components, "requested_clusters": cluster_k,
                    "command": f"python3 /app/ml_experiment.py train --components {components}" +
                               (f" --clusters {cluster_k}" if cluster_k is not None else ""),
                    "classifiers": classifiers, "analysis": segments,
                    "algorithm_note": "课程分类、聚类、降维 PPT 与作业四材料未在仓库找到；分类按任务指定算法，聚类与降维采用常见课程算法，需与课程材料最终核对。"}
        (output / "report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "status.json").write_text(json.dumps({"status": "completed", "stage": "全部实验完成"}), encoding="utf-8")
        (RUNTIME / "latest_experiment.json").write_text(json.dumps({"experiment_id": run_id, "report": str(output / "report.json")}, ensure_ascii=False), encoding="utf-8")
        return manifest
    except Exception as exc:
        (output / "status.json").write_text(json.dumps({"status": "failed", "stage": "实验中断", "error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 MovieLens 迭代二的六种算法")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--clusters", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.components, args.clusters), ensure_ascii=False, indent=2))
