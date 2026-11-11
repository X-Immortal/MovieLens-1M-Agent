# MovieLens 1M 数据分析 Agent

系统使用 Hadoop Streaming 清洗数据并评估五维质量，在统一时间切分上训练分类、聚类和降维模型。DeepSeek 通过 Tool Calling 选择工具、生成参数；Python 校验并执行工具，再将结果返回模型解释。网页支持连续追问、会话管理、算法对照和报告下载。

## 准备与启动

需要 Docker Engine、Docker Compose 和 Bash；数据初始化脚本需要 Python 3.11 或更新版本，只使用标准库。以下命令均在项目根目录执行；Windows 可使用 WSL。命令中的 `/app` 是容器内的代码目录。

1. 执行初始化脚本，从[课程数据链接](https://box.nju.edu.cn/d/5366547b75e549ac814b/files/?p=%2Fml-1m.zip&dl=1)获取本项目使用的 `ml-1m-local-raw-v1` 数据版本：

```bash
python3 scripts/init_data.py
```

脚本校验压缩包与文件的 SHA-256，将三份数据文件及许可说明 `README` 放入 `ml-1m/`。重复运行时校验已有数据；已有文件不完整或属于其他版本时停止，并保留原文件。可用 `--data-dir <目录>` 指定其他空目录。

2. 参照 `.env.example` 在项目根目录创建 `.env`，设置 `DEEPSEEK_API_KEY`。按需配置 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。
3. 执行启动脚本：

```bash
./run.sh
```

`run.sh` 自动校验或下载课程数据，并在 `.env` 不存在时从 `.env.example` 创建配置文件。

打开 <http://localhost:8501>。首次运行时，在页面输入“执行默认规则清洗 MovieLens 1M 并评估五维质量”；完成后打开“分析资料 → 实验管理”，点击“训练新实验”。已有数据和模型时可直接查询。

修改 API 配置后执行 `docker compose up -d --force-recreate`。使用 `docker compose down` 停止服务。源码和原始数据只读挂载，生成结果保存在 `runtime/`。API 未配置或不可用时，自然语言请求返回错误，已保存的对话与报告仍可查看。

## 使用方式

| 任务 | 示例问题 | 主要结果 |
| --- | --- | --- |
| 分类 | 用户 1 会给电影 1 打高分吗？ | 逻辑回归与决策树的概率、类别、阈值及评价指标 |
| 聚类 | 比较用户 1、2、3 在两种聚类算法中的归属。 | K-Means 与层次聚类的群体、画像和偏好摘要 |
| 降维 | 比较 PCA 与 SVD 的二维降维结果。 | 坐标图、方差比例、重构误差、聚类对照与资源开销 |
| 数据质量 | 清洗前后的数据质量有什么变化？ | 五维评分、数据处置计数、异常原因与数据样例 |

在同一会话中可以继续问“同一个用户和电影，换成决策树”“为什么这两个概率不同”。展开“分析依据”查看参数、版本、时间切分和工具调用；“分析资料”提供报告及数据文件下载，散点图下方可下载 PNG。

会话列表右侧的删除按钮用于删除对应对话。正在分析的会话需等待完成；新建空白对话在发送第一条消息后保存。

## Agent 工具接口

调用流程为：用户问题和会话消息 → DeepSeek 生成工具调用 → Python 校验参数并执行 → 工具结果返回 DeepSeek → 生成回答。模型可以组合调用，也可以基于已有结果回答解释性追问。接口遵循 [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)。

| 工具 | 参数与用途 |
| --- | --- |
| `logistic_regression`、`decision_tree` | `mode=predict` 时需要 `user_id`、`movie_id`；`mode=metrics` 读取验证和测试指标 |
| `kmeans`、`agglomerative` | 查询全部群体、单个 `user_id` 或 `user_ids` 集合；可指定已保存的 `n_clusters` |
| `pca`、`truncated_svd` | 查询 `components`、`user_ids`，核对 `retention_threshold`，通过 `comparison_tools` 和 `n_clusters` 选择聚类对照 |
| `quality_report` | 读取清洗报告与样例 |
| `experiment_history` | 读取实验版本与指标摘要 |
| `hadoop_streaming` | 按 `rule_version=default-v1` 执行清洗与评分 |

完整参数 Schema 见 [ml_tools.py](app/ml_tools.py)。用户未指定算法时，每类任务对照两种算法。查询使用已登记模型；未保存的维数或群数返回明确状态。保留阈值用于检查已有方差比例，用户集合用于查询已有归属与坐标。清洗工具只响应明确的执行请求。

每次请求最多执行 6 轮工具调用，随后生成总结；模型上下文保留最近 12 轮对话。网页接收绘图坐标样本，模型消息保留前 10 行及指定用户坐标。

## 数据与评价口径

初始化脚本获取的 `ml-1m-local-raw-v1` 包含用户记录 6,946 条、电影记录 4,465 条和评分记录 1,150,241 条，与随文实验输入一致。该版本基于 [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)，保留官方版全部记录，并分别增加 906、582 和 150,032 条记录，三张表均增加约 15%。

T1 是训练期截止时间，T2 是验证期截止时间。训练评分的时间戳不晚于 T1，验证评分位于 T1 之后至 T2，测试评分位于 T2 之后。

分类标签为 `Rating >= 4`。训练事件的历史统计只使用严格更早的评分，同秒事件相互排除；验证、测试和交互预测使用 T1 快照。验证集用于选择参数与阈值，测试集用于最终评价。训练期出现过的用户与电影组合在交互预测时会从聚合特征中扣除，其预测结果属于情景估计。

聚类和降维使用训练期用户画像。类型偏差经平滑处理，未观察到的类型使用中性偏差。降维前后对照使用相同用户、群数和聚类算法；轮廓系数在各自空间计算，方差保留和重构误差单独报告。

实验设置、数据规模、评价结果和适用范围见[实验报告](docs/迭代二实验报告.md)。

## 训练与运行产物

数据版本变化后重新训练。默认生成二维表示，也可以指定维数或群数：

```bash
docker compose run --rm --no-deps movielens-agent python3 /app/ml_experiment.py train
docker compose run --rm --no-deps movielens-agent python3 /app/ml_experiment.py train --components 3
docker compose run --rm --no-deps movielens-agent python3 /app/ml_experiment.py train --clusters 5
```

| 项目相对路径 | 内容 |
| --- | --- |
| `runtime/latest_report.json` | 最近一次清洗报告及数据、规则版本和时间边界 |
| `runtime/runs/<task_id>/` | 清洗任务的数据、报告和日志 |
| `runtime/latest_experiment.json` | 最近一次成功实验的索引 |
| `runtime/experiments/<experiment_id>/` | 模型、画像、群体归属、降维坐标、T1 快照和实验报告 |
| `runtime/sessions/` | 会话消息与工具调用记录 |

这些目录由运行过程创建。随文的[实验结果数据](docs/results/iteration2.json)保存报告所用统计和参数，供阅读与核对。

## 验证与提交

协议和会话检查不调用外部模型；算法检查需要完成清洗与训练：

```bash
docker compose exec -T movielens-agent python3 -m unittest -v test_agent test_iteration2 test_sessions
curl -fsS http://localhost:8501/health
```

联网检查需要有效的 API 配置，会产生 API 用量：

```bash
docker compose exec -T -e RUN_LIVE_AGENT_TESTS=1 movielens-agent python3 -m unittest -v test_agent_live
```

- [演示步骤](docs/迭代二快捷检查与演示.md)
- [交付检查记录](DELIVERY_CHECKLIST.md)
- [提交说明](提交说明.md)

