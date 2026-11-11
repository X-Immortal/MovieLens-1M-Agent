# MovieLens 1M Hadoop 数据治理 Agent（迭代一 MVP）

这个仓库提供一个可以端到端演示的最小实现：用户在 Streamlit 输入一次自然语言请求，Agent 启动 Hadoop Streaming 任务，先对 `ml-1m` 原始数据评分，再用同一套规则清洗、去重和隔离，最后对清洗结果再次评分。DeepSeek API 只用于解释已经生成的 Hadoop 事实，五维分数和清洗文件不会由模型生成。

## 运行环境

- Docker Engine、Docker Compose 和 Bash。
- Python 3.11 或更新版本，用于数据初始化；脚本只使用标准库。
- DeepSeek 解释需要在 `.env` 中配置 `DEEPSEEK_API_KEY`，可选项见 `.env.example`。

首次构建会拉取 `bde2020/hadoop-base:2.0.0-hadoop3.2.1-java8`，复制 Hadoop/Java 运行时到 Python 镜像并安装 Streamlit，镜像较大，请预留磁盘和网络时间。

## 数据初始化与启动

初始化脚本从[课程网盘](https://box.nju.edu.cn/d/5366547b75e549ac814b/files/?p=%2Fml-1m.zip&dl=1)下载 `ml-1m-local-raw-v1`，校验压缩包及数据文件的 SHA-256，将三份数据文件与许可说明 `README` 写入 `ml-1m/`。已有相同数据时直接通过；已有文件不完整或属于其他版本时停止并保留原文件。

```bash
python3 scripts/init_data.py
./run.sh
```

`run.sh` 会自动调用初始化脚本，也可直接执行它完成数据准备和服务启动。缺少 `.env` 时，启动脚本从 `.env.example` 创建配置文件。配置 API 后执行 `docker compose up -d --force-recreate`；API 未配置时仍可执行 Hadoop 清洗与质量评分，模型解释不可用。

浏览器打开 <http://localhost:8501>，点击“运行 Hadoop 清洗与评估”。运行日志和报告会保存在宿主机 `runtime/runs/<task_id>/`，最新报告是 `runtime/latest_report.json`。停止服务：

```bash
docker compose down
```

也可以不用前端直接运行 Agent CLI：

```bash
docker compose run --rm movielens-agent python3 /app/agent.py \
  '请使用默认规则清洗 MovieLens 1M，并解释清洗前后五维变化'
```

## Hadoop 任务链

`app/pipeline.py` 对三个 `.dat` 文件分别启动真实的 `hadoop jar ... hadoop-streaming-*.jar` 作业，强制使用 `fs.defaultFS=file:///` 和 Hadoop local framework，使单容器即可复现，不依赖宿主机安装 Hadoop。每个质量作业由 `quality_mapper.py` 与 `quality_reducer.py` 产生 JSON 摘要；每个清洗作业由 `clean_mapper.py` 与 `clean_reducer.py` 产生清洗行、隔离行和重复行。驱动程序只读取 Hadoop 输出并组装报告，不计算或填充分数。

清洗规则版本为 `default-v1`：

- `users.dat`：字段数量、ID、性别、年龄编码、职业编码和邮编必须有效。
- `movies.dat`：字段数量、MovieID、标题和 18 个已知类型必须有效。
- `ratings.dat`：字段数量、用户/电影 ID、1—5 整数评分和历史时间窗口必须有效，并且必须引用有效用户和电影。
- 通过业务键去重：用户按 UserID，电影按 MovieID，评分按 UserID + MovieID + Timestamp。被隔离或去重的行仍保存在任务目录中，不能被表述为“已修复”。
- 报告还包含每张表的隔离原因计数、少量原始行示例和去重示例，便于 Agent 追问时引用实际证据。

五维分数使用同一套 Hadoop 输出口径：

| 维度 | 定义 |
| --- | --- |
| Accurate | 语义约束满足率（评分范围、ID、时间窗口等） |
| Complete | 必需字段非空且字段数量正确的行比例 |
| Unique | 业务键去重后的行数 / 总行数 |
| Up-to-date | ratings 时间戳落在 2000-01-01 至 2004-11-09 窗口内的比例 |
| Consistent | 解析、类别取值及 ratings 跨表引用均满足规则的行比例 |

格式正确不能证明用户自填人口属性的现实真实性；`Up-to-date` 也只是历史窗口判断。`T1/T2` 暂按有效 ratings 时间范围的 80%/90% 固定切分登记，后续迭代应沿用任务报告里的边界。

## 当前数据检查

初始化脚本获取的实验输入包含用户记录 6,946 条、电影记录 4,465 条和评分记录 1,150,241 条。该版本在 MovieLens 1M 官方原版的基础上增加了约 15% 的记录，包含字段数量错误、分隔符错误、重复行和越界时间戳等异常。任务报告给出原始与清洗后计数、隔离数、去重数和五维评分。

本次容器验证任务 `20260930T134854Z-57b1da18` 已完整跑通：原始 `users/movies/ratings` 行数为 `6,946 / 4,465 / 1,150,241`，清洗后为 `6,112 / 3,941 / 1,000,209`；隔离行数为 `499 / 302 / 114,025`，完全重复行数为 `118 / 117 / 17,257`，业务键冲突行数为 `217 / 105 / 18,750`，修复行数为 `0 / 25 / 0`。五维分数（清洗前 → 清洗后）为 `Accurate 91.94 → 100.00`、`Complete 97.65 → 100.00`、`Unique 89.48 → 100.00`、`Up-to-date 93.35 → 100.00`、`Consistent 90.12 → 100.00`。T1/T2 区间验证由 Hadoop 输出：清洗前 train/validation/test 为 `1,062,919 / 5,437 / 5,368`，清洗后为 `989,404 / 5,437 / 5,368`。这些数字来自 `runtime/latest_report.json` 中对应任务的 Hadoop 输出；重新运行会生成新的任务标识和报告。完整交付核对见 `DELIVERY_CHECKLIST.md`。

## 目录

```text
docker-compose.yml          单容器可复现 Hadoop + Streamlit 环境
Dockerfile                  Hadoop 3.2.1/Java 8 + Python 3.11 + Streamlit
run.sh                      初始化数据、构建并启动
scripts/init_data.py        下载并校验课程实验数据
.env.example                API 配置模板
app/pipeline.py             Hadoop 作业编排与报告聚合
app/hadoop_jobs/            Streaming mapper/reducer
app/agent.py                Agent 工具层与 DeepSeek 调用
app/streamlit_app.py        简易前端
ml-1m/                      本地输入数据（只读挂载）
runtime/                    运行报告、清洗数据与日志
docs/                       课程需求文档
```

## 故障定位

如果任务失败，先查看 `runtime/runs/<task_id>/logs/`，或执行 `docker compose logs -f movielens-agent`。页面在 Hadoop 作业未成功时不会展示预置分数。DeepSeek 暂时不可用时，Hadoop 报告仍然保留，页面会明确标注解释调用失败，不会替换真实评分。
