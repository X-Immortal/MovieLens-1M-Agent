"""Protocol and session tests; these tests do not call an external model."""
from __future__ import annotations

import copy
import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import ml_tools
import web_server
from agent import (
    DeepSeekClient,
    LLMError,
    MovieLensAgent,
    conversation_history,
    model_result,
)
from ml_tools import TOOL_DEFINITIONS, TOOL_VALIDATORS, ToolError, execute_tool


def call(name, arguments, identifier="call-1"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def calling(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls),
            "reasoning_content": "provider continuation field"}


def saying(content):
    return {"role": "assistant", "content": content}


class ScriptedClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class ToolSchemaTests(unittest.TestCase):
    def test_definitions_are_valid_json_schemas(self):
        for validator in TOOL_VALIDATORS.values():
            validator.check_schema(validator.schema)
        self.assertTrue(set(ml_tools.TOOLS).issubset(TOOL_VALIDATORS))

    def test_invalid_model_arguments_do_not_reach_algorithms(self):
        cases = [
            ("logistic_regression", {"mode": "predict", "user_id": 1}),
            ("logistic_regression", {"mode": "predict", "user_id": True, "movie_id": 1}),
            ("decision_tree", {"mode": "metrics", "user_id": 1}),
            ("decision_tree", {"mode": "predict", "user_id": "1", "movie_id": 1}),
            ("kmeans", {"user_id": 1, "user_ids": [1, 2]}),
            ("kmeans", {"user_ids": [1, 1]}), ("kmeans", {"user_ids": []}),
            ("kmeans", {"n_clusters": 13}), ("kmeans", {"user_id": None}),
            ("pca", {"components": 1}), ("pca", {"retention_threshold": 20}),
            ("pca", {"comparison_tools": ["unknown"]}), ("pca", {"shell": "anything"}),
            ("pca", {"retention_threshold": float("nan")}), ("pca", []),
            ("unregistered", {}), ("hadoop_streaming", {"rule_version": "custom"}),
        ]
        with patch.dict(ml_tools.TOOLS, {name: Mock() for name in ml_tools.TOOLS}):
            for name, arguments in cases:
                with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                    execute_tool(name, arguments)
            for fn in ml_tools.TOOLS.values():
                fn.assert_not_called()

    def test_metrics_are_read_without_prediction(self):
        manifest = {"experiment_id": "exp-1", "cleaned_data_version": "data-1",
                    "rule_version": "default-v1", "boundaries": {"T1_epoch": 10, "T2_epoch": 20},
                    "classifiers": {"split_counts": {"train": 8, "validation": 2, "test": 2},
                                    "feature_columns": ["age"], "models": {
                        "decision_tree": {"test": {"f1": 0.75}}}}}
        with patch.object(ml_tools, "experiment", return_value=(manifest, Path("."))), patch.dict(
                ml_tools.TOOLS, {"decision_tree": Mock()}) as registered:
            result = execute_tool("decision_tree", {"mode": "metrics"})
            self.assertEqual(result["test"]["f1"], 0.75)
            registered["decision_tree"].assert_not_called()


class AgentConversationTests(unittest.TestCase):
    def test_model_selects_sequential_mixed_tools_and_receives_results(self):
        client = ScriptedClient(
            calling(call("decision_tree", {"mode": "predict", "user_id": 2, "movie_id": 3})),
            calling(call("kmeans", {"user_id": 2}, "call-2")), saying("根据工具结果回答"))
        results = [{"status": "completed", "task": "classification", "tool": "decision_tree",
                    "positive_probability": 0.81},
                   {"status": "completed", "task": "clustering", "tool": "kmeans", "cluster": 2}]
        history = []
        with patch("agent.execute_tool", side_effect=results) as execute:
            answer = MovieLensAgent(client).run("帮我看看他喜不喜欢这部，再看看同好", history)
        self.assertEqual(execute.call_args_list[0].args, ("decision_tree", {"mode": "predict", "user_id": 2, "movie_id": 3}))
        self.assertEqual(answer["task"], "mixed")
        self.assertEqual(answer["results"], results)
        self.assertEqual(answer["summary"], "根据工具结果回答")
        self.assertEqual(json.loads(client.requests[1][0][-1]["content"])["positive_probability"], 0.81)
        self.assertEqual(history[1]["reasoning_content"], "provider continuation field")
        self.assertEqual([message["role"] for message in history],
                         ["user", "assistant", "tool", "assistant", "tool", "assistant"])

    def test_parallel_calls_all_have_matching_tool_replies(self):
        client = ScriptedClient(calling(call("pca", {}, "pca"), call("truncated_svd", {}, "svd")),
                                saying("对照结果"))
        with patch("agent.execute_tool", return_value={"status": "completed", "task": "reduction", "tool": "pca"}):
            MovieLensAgent(client).run("对照降维")
        replies = [message for message in client.requests[1][0] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in replies], ["pca", "svd"])

    def test_followup_uses_history_and_can_answer_without_tools(self):
        client = ScriptedClient(saying("请提供电影编号。"),
                                calling(call("decision_tree", {"mode": "predict", "user_id": 1, "movie_id": 7})),
                                saying("预测结果"), saying("这是模型估计的概率。"))
        history = []
        agent = MovieLensAgent(client)
        self.assertEqual(agent.run("用决策树看看一号用户会不会喜欢", history)["tools"], [])
        with patch("agent.execute_tool", return_value={"status": "completed", "task": "classification", "tool": "decision_tree"}):
            agent.run("电影是七号", history)
        final = agent.run("这个概率怎么理解", history)
        self.assertEqual(final["tools"], [])
        self.assertIn("一号用户", client.requests[-1][0][1]["content"])
        self.assertEqual(final["summary"], "这是模型估计的概率。")

    def test_invalid_arguments_return_to_model_and_successful_results_survive(self):
        bad = call("pca", {})
        bad["function"]["arguments"] = "{"
        client = ScriptedClient(calling(bad, call("kmeans", {"user_id": 1}, "good")), saying("部分完成"))
        with patch("agent.execute_tool", return_value={"status": "completed", "task": "clustering", "tool": "kmeans"}) as execute:
            answer = MovieLensAgent(client).run("查看分群和坐标")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(answer["status"], "partial")
        replies = [json.loads(m["content"]) for m in client.requests[1][0] if m["role"] == "tool"]
        self.assertEqual([r["status"] for r in replies], ["failed", "completed"])

    def test_tool_error_is_available_for_parameter_correction(self):
        client = ScriptedClient(calling(call("kmeans", {"user_id": "1"})),
                                calling(call("kmeans", {"user_id": 1}, "corrected")), saying("已修正"))
        with patch("agent.execute_tool", side_effect=[
            ToolError("user_id 必须为整数"),
            {"status": "completed", "task": "clustering", "tool": "kmeans"}
        ]):
            answer = MovieLensAgent(client).run("一号用户在哪个群")
        self.assertIn("必须为整数", client.requests[1][0][-1]["content"])
        self.assertEqual(len(answer["results"]), 1)

    def test_connection_failure_after_tool_preserves_results_and_context(self):
        client = ScriptedClient(calling(call("kmeans", {})), LLMError("请求超时"))
        history = []
        with patch("agent.execute_tool", return_value={"status": "completed", "task": "clustering", "tool": "kmeans"}):
            answer = MovieLensAgent(client).run("分群", history)
        self.assertEqual(answer["status"], "partial")
        self.assertEqual(answer["error"], "请求超时")
        self.assertEqual(history[-2]["role"], "tool")
        self.assertEqual(history[-1]["content"], "请求超时")

    def test_missing_credentials_reports_failure(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), patch("agent.execute_tool") as execute:
            answer = MovieLensAgent().run("预测用户一对电影一的高分概率")
        self.assertEqual(answer["status"], "failed")
        self.assertIn("DEEPSEEK_API_KEY", answer["error"])
        execute.assert_not_called()

    def test_call_limit_requests_final_answer_without_tools(self):
        client = ScriptedClient(calling(call("kmeans", {})), saying("当前结果"))
        with patch("agent.MAX_TOOL_ROUNDS", 1), patch("agent.execute_tool", return_value={
                "status": "completed", "task": "clustering", "tool": "kmeans"}):
            MovieLensAgent(client).run("分析")
        self.assertEqual(client.requests[-1][1]["tool_choice"], "none")

    def test_malformed_calls_do_not_leave_unanswered_protocol_messages(self):
        invalid = calling(call("kmeans", {}), call("pca", {}))
        history = []
        answer = MovieLensAgent(ScriptedClient(invalid)).run("分析", history)
        self.assertEqual(answer["status"], "failed")
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])

    def test_history_trimming_keeps_complete_turns(self):
        history = []
        for index in range(15):
            history.extend([{"role": "user", "content": str(index)},
                            calling(call("kmeans", {}, str(index))),
                            {"role": "tool", "tool_call_id": str(index), "content": "{}"}, saying("结果")])
        MovieLensAgent(ScriptedClient(saying("追问回答"))).run("继续", history)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(sum(m["role"] == "user" for m in history), 12)

    def test_plot_context_retains_requested_users(self):
        result = {"coordinates_sample": [{"UserID": i} for i in range(1000)],
                  "selected_users": [{"UserID": 900}]}
        compact = model_result(result)
        self.assertEqual(len(compact["coordinates_sample"]), 10)
        self.assertEqual(compact["selected_users"], [{"UserID": 900}])
        self.assertEqual(len(result["coordinates_sample"]), 1000)


class ClientProtocolTests(unittest.TestCase):
    def test_request_contains_tools_and_full_messages(self):
        response = {"choices": [{"finish_reason": "tool_calls", "message": calling(call("pca", {}))}]}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only", "DEEPSEEK_BASE_URL": "https://example.test/v1"}):
            client = DeepSeekClient()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = client.complete([saying("previous answer")], tools=TOOL_DEFINITIONS)
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(body["tools"], TOOL_DEFINITIONS)
        self.assertEqual(body["messages"], [saying("previous answer")])
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "pca")

    def test_http_failure_does_not_expose_response_body(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}):
            client = DeepSeekClient()
        error = urllib.error.HTTPError("https://example.test", 401, "secret in provider error", {}, None)
        with patch("urllib.request.urlopen", side_effect=error), self.assertRaisesRegex(LLMError, "HTTP 401") as raised:
            client.complete([], tools=TOOL_DEFINITIONS)
        self.assertNotIn("secret", str(raised.exception))


class SessionTests(unittest.TestCase):
    def test_saved_context_survives_agent_recreation_and_isolated_sessions(self):
        with TemporaryDirectory() as temporary, patch.object(web_server, "SESSIONS", Path(temporary)), patch.dict(web_server.JOBS, {}, clear=True):
            session_id = "a" * 32
            web_server.save_session({"id": session_id, "messages": []})
            web_server.JOBS["job"] = {"state": "running"}
            client = ScriptedClient(saying("请提供电影编号。"))
            with patch.object(web_server, "MovieLensAgent", return_value=MovieLensAgent(client)):
                web_server.do_agent("job", session_id, "用户一会喜欢吗", [])
            session = web_server.get_session(session_id)
            restored = conversation_history(session)
            self.assertEqual(restored[0]["content"], "用户一会喜欢吗")
            restored.append(saying("changed"))
            self.assertEqual(len(session["llm_messages"]), 2)
            self.assertEqual(conversation_history({"messages": []}), [])

    def test_legacy_session_keeps_question_and_real_results(self):
        session = {"messages": [{"role": "user", "content": "用户一"},
                               {"role": "assistant", "answer": {
                                   "summary": "已预测", "results": [{"UserID": 1, "MovieID": 2, "positive_probability": 0.8}]}}]}
        restored = conversation_history(session)
        self.assertEqual(restored[0]["content"], "用户一")
        self.assertIn('"positive_probability": 0.8', restored[1]["content"])


if __name__ == "__main__":
    unittest.main()

