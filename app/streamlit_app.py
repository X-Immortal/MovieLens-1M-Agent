#!/usr/bin/env python3
import json
from pathlib import Path

import streamlit as st

from agent import MovieLensAgent
from pipeline import PipelineError


st.set_page_config(page_title="MovieLens Hadoop Agent", page_icon="🎬", layout="wide")
st.title("MovieLens 1M · Hadoop 数据治理 Agent")
st.caption("评分与清洗由真实 Hadoop Streaming mapper/reducer 产生；DeepSeek 只负责解释结果。")

if "report" not in st.session_state:
    st.session_state.report = None
if "agent" not in st.session_state:
    st.session_state.agent = MovieLensAgent()

request = st.text_area(
    "自然语言请求",
    value="请使用默认规则清洗 MovieLens 1M，评估清洗前后的 Accurate、Complete、Unique、Up-to-date、Consistent，并说明仍存在的问题。",
    height=100,
)
if st.button("运行 Hadoop 清洗与评估", type="primary"):
    with st.status("正在执行 Hadoop Streaming…", expanded=True) as status:
        try:
            report = st.session_state.agent.run(request)
            st.session_state.report = report
            if report.get("status") == "completed":
                status.update(label=f"任务完成：{report['task_id']}", state="complete")
            else:
                status.update(label=f"任务未执行：{report['task_id']}", state="error")
                st.error(report.get("error", "任务失败"))
        except PipelineError as exc:
            status.update(label="任务失败", state="error")
            st.error(str(exc))

report = st.session_state.report
if report:
    if report.get("status") != "completed":
        st.warning(f"任务状态：{report.get('status')}；任务 ID：{report.get('task_id')}")
        st.json(report)
        st.stop()
    quality = report["quality"]
    st.subheader("五维质量对比")
    cols = st.columns(5)
    for col, name in zip(cols, ("Accurate", "Complete", "Unique", "Up-to-date", "Consistent")):
        col.metric(name, f"{quality['after'][name]:.2f}", f"{quality['delta'][name]:+.2f}")

    st.subheader("执行结果")
    st.json({
        "status": report["status"],
        "task_id": report["task_id"],
        "data_version": report["data_version"],
        "cleaned_data_version": report["cleaned_data_version"],
        "rule_version": report["rule_version"],
        "T1/T2": report["boundaries"],
        "execution": report["execution"],
    })

    st.subheader("清洗处置")
    st.dataframe([
        {
            "表": table,
            "原始行数": quality["summaries"]["before"][table]["total"],
            "清洗后行数": quality["summaries"]["after"][table]["total"],
            "保留": report["cleaning"][table]["cleaned"],
            "隔离": report["cleaning"][table]["rejected"],
            "去重": report["cleaning"][table]["duplicates"],
        }
        for table in ("users", "movies", "ratings")
    ], hide_index=True, use_container_width=True)

    st.subheader("T1/T2 时间区间验证")
    st.json({"boundaries": report["boundaries"], "counts": report["time_split"]})

    with st.expander("评分口径与 Hadoop 原始摘要"):
        st.json({"method": quality["method"], "before": quality["before"], "after": quality["after"], "summaries": quality["summaries"]})
    with st.expander("异常原因与代表性原始行"):
        for table in ("users", "movies", "ratings"):
            st.markdown(f"**{table}**")
            st.json({
                "reject_reason_counts": report["cleaning"][table]["reject_reason_counts"],
                "reject_examples": report["cleaning"][table]["reject_examples"],
                "duplicate_examples": report["cleaning"][table]["duplicate_examples"],
                "conflict_examples": report["cleaning"][table]["conflict_examples"],
            })
    with st.expander("Agent 解释"):
        st.write(report.get("agent_explanation", "未生成解释。"))

    st.subheader("结果追问")
    question = st.text_input("基于本次任务继续提问", placeholder="例如：哪些问题被隔离，为什么不能说已经修复？")
    if st.button("询问 Agent") and question.strip():
        with st.spinner("DeepSeek 正在根据本次 Hadoop 事实回答…"):
            st.write(st.session_state.agent.follow_up(report, question.strip()))

    st.subheader("清洗后数据下载")
    report_path = Path(report["execution"]["task_dir"]) / "report.json"
    if report_path.exists():
        st.download_button("下载本次评估报告", report_path.read_bytes(), file_name="report.json", mime="application/json", key="download-report")
    for table in ("users", "movies", "ratings"):
        path = Path(report["cleaning"][table]["clean_file"])
        if path.exists():
            st.download_button(
                f"下载 {table}.dat", path.read_bytes(), file_name=f"{table}.clean.dat", mime="text/plain", key=f"download-{table}",
            )
            with st.expander(f"查看 {table} 清洗后样例"):
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
                st.code("\n".join(lines) or "（无保留记录）")
else:
    st.info("输入请求后点击按钮。页面不会展示预置分数，只有 Hadoop 任务成功后才会出现结果。")
