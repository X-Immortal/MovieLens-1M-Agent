# 迭代一严格交付检查清单

本文件记录迭代一当时的交付检查；其中的 `app/streamlit_app.py` 后来已替换为 `app/web_server.py` 和 `app/web/`。当前迭代二的核对结果与现场操作顺序见 [迭代二快捷检查与演示路径](docs/迭代二快捷检查与演示.md)。

本清单对应需求文档 `docs/迭代一_Hadoop数据清洗与Agent基础.md` 和 `docs/引言_项目总体要求与汇报安排.md`。最终真实运行任务：`20260930T134854Z-57b1da18`。

| 要求 | 结果 | 具体证据 |
| --- | --- | --- |
| 自然语言任务提交 | PASS | `app/streamlit_app.py` 的请求输入框；`app/agent.py` 的 `MovieLensAgent.run(request)`；最终报告的 `request` 字段。 |
| 唯一任务 ID | PASS | `app/pipeline.py` 生成 UTC 时间戳 + UUID；`runtime/runs/20260930T134854Z-57b1da18/`。 |
| running/completed/failed 状态 | PASS | `status_history.jsonl` 记录最终任务 `running -> completed`；不支持规则任务 `runtime/audit/unsupported-test/runs/20260930T134522Z-bad3932c/status_history.jsonl` 记录 `running -> failed`。 |
| 清洗前五维评估 | PASS | `quality_before/{users,movies,ratings}/part-00000`；报告 `quality.before`。 |
| Hadoop 实际清洗 | PASS | `execution.engine=Hadoop Streaming`；`logs/clean_*.log` 含 Hadoop local job 完成记录；`clean/*.dat` 由 reducer 输出。 |
| 清洗后评估 | PASS | `quality_after/{users,movies,ratings}/part-00000`；报告 `quality.after`。 |
| 前后使用完全相同评分规则 | PASS | 同一 `quality_mapper.py`/`quality_reducer.py` 处理前后数据；报告 `scoring_rule_fingerprint` 相同；`quality_scores/part-00000` 由独立 Hadoop reducer 产生并与报告逐项匹配。 |
| 修复、去重、冲突、隔离计数 | PASS | 报告 `cleaning.*.repaired/duplicates/conflicts/rejected`；冲突另存为 `*.conflicts.jsonl`，重复另存为 `*.duplicates.dat`，隔离另存为 `*.rejected.jsonl`。 |
| 代表性原始异常记录及原因 | PASS | 报告 `reject_reason_counts`、`reject_examples`、`duplicate_examples`、`conflict_examples`；页面“异常原因与代表性原始行”。 |
| 清洗后数据版本 | PASS | 报告 `cleaned_data_version=ml-1m-clean-v1-20260930T134854Z-57b1da18`。 |
| 清洗规则版本 | PASS | 报告 `rule_version=default-v1` 和 `rule_resolution`。 |
| 固定 T1/T2 | PASS | `app/pipeline.py` 注册固定 epoch；报告 `T1=2002-08-04T23:40:58Z`、`T2=2002-11-16T20:45:24Z`。 |
| T1/T2 区间自动验证 | PASS | 报告 `time_split`；最终 Hadoop 结果为清洗前 `train/validation/test=1,062,919/5,437/5,368`，清洗后 `989,404/5,437/5,368`，且分别等于有效 ratings 总数。 |
| 评分方法与局限 | PASS | 报告 `quality.method`；页面“评分口径与 Hadoop 原始摘要”。 |
| 清洗数据样例与获取 | PASS | 页面展示每张清洗后表前五行并提供 `.dat` 下载；`clean/*.dat`。 |
| 评估报告获取 | PASS | 页面“下载本次评估报告”；`report.json` 和 `runtime/latest_report.json`。 |
| DeepSeek 解释与追问 | PASS | `app/agent.py` 只把当前报告事实发送给 DeepSeek；最终实测追问引用 `reject_reason_counts`，没有生成报告外数字。 |
| 不支持的自定义规则不被静默忽略 | PASS | 请求“请只保留评分为5分的记录”得到 `failure_code=unsupported_rule` 和明确错误；证据见 `runtime/audit/unsupported-test/...`。 |
| 无 mock/占位结果 | PASS | 最终报告所有分数来自 Hadoop `quality_scores/part-00000`；数据文件、日志和 reducer 输出均实际存在。 |

最终运行摘要：原始 users/movies/ratings 为 `6,946/4,465/1,150,241`，清洗后为 `6,112/3,941/1,000,209`；五维评分为 `91.94/97.65/89.48/93.35/90.12` 到 `100/100/100/100/100`。这些数字仅来自任务 `20260930T134854Z-57b1da18` 的实际 Hadoop 输出。
