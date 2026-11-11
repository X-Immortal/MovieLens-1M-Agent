"""Opt-in checks against the configured DeepSeek service and saved models."""
from __future__ import annotations

import os
import unittest

from agent import MovieLensAgent


@unittest.skipUnless(os.environ.get("RUN_LIVE_AGENT_TESTS") == "1", "设置 RUN_LIVE_AGENT_TESTS=1 才会调用 DeepSeek")
class LiveAgentTests(unittest.TestCase):
    def ask(self, prompt, history=None):
        answer = MovieLensAgent().run(prompt, history)
        print({"prompt": prompt, "status": answer["status"], "calls": answer["tool_calls"]}, flush=True)
        self.assertTrue(answer.get("summary"), answer.get("error"))
        return answer

    def test_prediction_switch_and_explanation(self):
        history = []
        first = self.ask("一号用户会喜欢第一部电影吗？用逻辑回归估计打高分的可能性。", history)
        self.assertEqual(first["tools"], ["logistic_regression"])
        self.assertEqual((first["results"][0]["UserID"], first["results"][0]["MovieID"]), (1, 1))
        second = self.ask("同一个人同一部片，换成决策树再算一下。", history)
        self.assertEqual(second["tools"], ["decision_tree"])
        self.assertEqual((second["results"][0]["UserID"], second["results"][0]["MovieID"]), (1, 1))
        third = self.ask("这两个概率为什么不一样？只解释已有结果即可。", history)
        self.assertEqual(third["tools"], [])

    def test_subset_reduction_and_percentage_followup(self):
        history = []
        first = self.ask("把一号、二号、三号用户在 K-Means 里各归哪组告诉我。", history)
        self.assertEqual(first["tools"], ["kmeans"])
        self.assertEqual([row["UserID"] for row in first["results"][0]["selected_users"]], [1, 2, 3])
        second = self.ask("还是这三个人，查 PCA 的二维坐标，用四个群的 K-Means 对照，看看能不能保留至少百分之二十的信息。", history)
        self.assertEqual(second["tools"], ["pca"])
        result = second["results"][0]
        self.assertEqual(len(result["selected_users"]), 3)
        self.assertEqual(result["retention_threshold"], 0.2)
        self.assertFalse(result["retention_met"])
        self.assertEqual(set(result["comparison"]), {"kmeans"})

    def test_mixed_tasks_and_default_algorithm_pairs(self):
        answer = self.ask("先预测用户一对电影一会不会给高分，再查这个用户属于哪个群体。每类都对照两种算法。")
        self.assertEqual(answer["status"], "completed")
        self.assertEqual(set(answer["tools"]), {"logistic_regression", "decision_tree", "kmeans", "agglomerative"})
        self.assertEqual(answer["task"], "mixed")

    def test_missing_identifier_is_clarified_then_resolved(self):
        history = []
        first = self.ask("用决策树预测一下用户二会不会打高分。", history)
        self.assertEqual(first["tools"], [])
        second = self.ask("电影编号是三。", history)
        self.assertEqual(second["tools"], ["decision_tree"])
        self.assertEqual((second["results"][0]["UserID"], second["results"][0]["MovieID"]), (2, 3))

    def test_quality_query_and_unavailable_parameters(self):
        quality = self.ask("清洗之后完整性改善了多少？读现有报告就好。")
        self.assertEqual(quality["tools"], ["quality_report"])
        unavailable = self.ask("用 PCA 看用户一的坐标，降维前后都用五个群的 K-Means 对照。")
        self.assertEqual(unavailable["status"], "failed")
        self.assertTrue(any("只保存 4 群" in call.get("error", "") for call in unavailable["tool_calls"]))

    def test_reduction_pair_and_classification_metrics(self):
        reduction = self.ask("比较 PCA 和 SVD 的二维信息保留与重构误差。")
        self.assertEqual(set(reduction["tools"]), {"pca", "truncated_svd"})
        metrics = self.ask("逻辑回归和决策树在测试集上的表现谁更好？")
        self.assertEqual(set(metrics["tools"]), {"logistic_regression", "decision_tree"})
        self.assertTrue(all(call["arguments"]["mode"] == "metrics" for call in metrics["tool_calls"]))


if __name__ == "__main__":
    unittest.main()

