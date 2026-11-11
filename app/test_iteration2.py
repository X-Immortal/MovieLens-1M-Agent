"""Checks for temporal features, persisted tools, and error boundaries."""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ml_tools
import pandas as pd
import pipeline
import web_server
from ml_experiment import source_report, strict_prior
from ml_tools import TOOLS, ToolError, experiment


class TemporalFeatureTests(unittest.TestCase):
    def test_same_timestamp_events_are_excluded(self):
        events = pd.DataFrame([
            {"UserID": 1, "Timestamp": 10, "Rating": 5},
            {"UserID": 1, "Timestamp": 10, "Rating": 1},
            {"UserID": 1, "Timestamp": 11, "Rating": 4},
        ])
        prior = strict_prior(events, "UserID", "user")
        self.assertEqual(prior.loc[prior.Timestamp == 10, "user_count"].iloc[0], 0)
        self.assertEqual(prior.loc[prior.Timestamp == 11, "user_count"].iloc[0], 2)
        self.assertEqual(prior.loc[prior.Timestamp == 11, "user_sum"].iloc[0], 6)

    def test_registered_split_matches_model(self):
        source = source_report()
        manifest, _ = experiment()
        self.assertEqual(manifest["classifiers"]["split_counts"], source["time_split"]["after"])
        self.assertEqual(manifest["boundaries"], source["boundaries"])
        self.assertEqual(manifest["cleaned_data_version"], source["cleaned_data_version"])

    def test_rejected_rule_keeps_previous_completed_report(self):
        with TemporaryDirectory() as temporary, patch.object(pipeline, "RUNTIME_DIR", Path(temporary)):
            latest = Path(temporary) / "latest_report.json"
            latest.write_text('{"status":"completed","task_id":"protected"}', encoding="utf-8")
            result = pipeline.run_pipeline("执行自定义清洗规则")
            self.assertEqual(result["failure_code"], "unsupported_rule")
            self.assertEqual(json.loads(latest.read_text(encoding="utf-8"))["task_id"], "protected")


class ToolTests(unittest.TestCase):
    def test_six_tools_load_persisted_results(self):
        results = {
            "logistic_regression": TOOLS["logistic_regression"](1, 1),
            "decision_tree": TOOLS["decision_tree"](1, 1),
            "kmeans": TOOLS["kmeans"](1),
            "agglomerative": TOOLS["agglomerative"](1),
            "pca": TOOLS["pca"](),
            "truncated_svd": TOOLS["truncated_svd"](),
        }
        self.assertEqual(set(results), set(TOOLS))
        self.assertTrue(all(item["status"] == "completed" for item in results.values()))
        self.assertEqual(len({item["experiment_id"] for item in results.values()}), 1)
        self.assertEqual(results["pca"]["components"], 2)

    def test_invalid_ids_and_dimension_fail_clearly(self):
        with self.assertRaisesRegex(ToolError, "未知 UserID"):
            TOOLS["logistic_regression"](999999, 1)
        with self.assertRaisesRegex(ToolError, "未知 MovieID"):
            TOOLS["decision_tree"](1, 999999)
        with self.assertRaisesRegex(ToolError, "训练期画像"):
            TOOLS["agglomerative"](999999)
        with self.assertRaisesRegex(ToolError, "只保存"):
            TOOLS["pca"](3)
        with self.assertRaisesRegex(ToolError, "只保存"):
            TOOLS["kmeans"](1, 5)

    def test_training_pair_is_removed_from_prediction_history(self):
        _, directory = experiment()
        user_id, movie_id = 1, 1
        pair = pd.read_csv(directory / "pair_snapshot_t1.csv")
        selected = pair[(pair.UserID == user_id) & (pair.MovieID == movie_id)]
        self.assertEqual(len(selected), 1)
        user = pd.read_csv(directory / "user_snapshot_t1.csv")
        original_count = int(user.loc[user.UserID == user_id, "user_count"].iloc[0])
        prediction = TOOLS["logistic_regression"](user_id, movie_id)
        self.assertTrue(prediction["target_pair_in_training"])
        self.assertEqual(prediction["user_train_ratings"], original_count - int(selected.pair_count.iloc[0]))

    def test_missing_model_and_version_mismatch(self):
        with (TemporaryDirectory() as temporary, patch.object(ml_tools, "RUNTIME", Path(temporary)),
              self.assertRaisesRegex(ToolError, "尚未训练")):
            experiment()
        source = source_report()
        changed = {**source, "cleaned_data_version": "different-version"}
        with (patch.object(ml_tools, "source_report", return_value=changed),
              self.assertRaisesRegex(ToolError, "版本")):
            experiment()


class WebProgressTests(unittest.TestCase):
    def test_running_jobs_show_registered_pipeline_and_training_stage(self):
        with TemporaryDirectory() as temporary, patch.object(web_server, "RUNTIME", Path(temporary)):
            for folder, job in (("runs", {"kind": "agent", "is_clean": True}),
                                ("experiments", {"kind": "train", "is_clean": False})):
                path = Path(temporary) / folder / "sample" / "status.json"
                path.parent.mkdir(parents=True)
                path.write_text('{"status":"running","stage":"正在处理真实数据"}', encoding="utf-8")
                self.assertEqual(web_server.current_stage({**job, "started_at": time.time() - 2,
                                                           "stage": "正在启动"}), "正在处理真实数据")


if __name__ == "__main__":
    unittest.main()
