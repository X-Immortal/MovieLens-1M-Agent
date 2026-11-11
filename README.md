# MovieLens 1M 数据分析 Agent

这个项目使用 Hadoop Streaming 清洗 MovieLens 1M 并计算五维数据质量分数，在登记的清洗数据和 T1/T2 时间边界上训练六种机器学习算法。网页工作台提供自然语言查询、会话历史、分类预测、用户群体、二维降维图和报告下载；页面通过轻量 HTTP 服务调用现有 Agent，不需要前端构建工具。DeepSeek 可辅助识别含糊请求；结果解释由 Agent 根据实际工具输出生成。未配置 API 时，常见中文请求和全部确定性查询仍可用。

## 启动

需要 Docker Engine、Docker Compose、Bash 和 Python 3.11 或更新版本。数据初始化脚本只使用 Python 标准库。如果 `runtime/latest_report.json` 和已登记的清洗文件不存在，先从页面明确发起“执行清洗 MovieLens 1M 并评估五维质量”，再点击“训练新实验”。提交包不包含原始数据和本机运行产物，具体准备步骤见 `提交说明.md`。

初始化脚本从[课程网盘](https://box.nju.edu.cn/d/5366547b75e549ac814b/files/?p=%2Fml-1m.zip&dl=1)下载 `ml-1m-local-raw-v1`，校验压缩包及文件的 SHA-256，将三份数据文件和许可说明保存到 `ml-1m/`。该数据版本与随文实验输入一致。

```bash
python3 scripts/init_data.py
./run.sh
```

`run.sh` 自动调用初始化脚本，也可直接运行。重复运行会校验已有数据；已有文件不完整或属于其他版本时停止并保留原文件。

打开 <http://localhost:8501>。运行 `docker compose down` 停止服务。启动脚本会在缺少 `.env` 时从 `.env.example` 创建配置文件；DeepSeek 可选。Docker 会将源代码只读挂载到容器，将所有运行产物写入 `runtime/`。设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 时只通过 `.env` 或环境变量传入，不要提交密钥。

已有模型直接加载，不会因普通提问重复训练。若清洗数据版本改变，在页面点击“训练新实验”，或运行：

```bash
docker compose run --rm --no-deps movielens-agent python3 /app/ml_experiment.py train
# 指定二维之外的目标维数，例如 3 维
docker compose run --rm --no-deps movielens-agent python3 /app/ml_experiment.py train --components 3
# 固定两种聚类算法均使用 5 群
docker compose run --rm --no-deps movielens-agent python3 /app/ml_experiment.py train --clusters 5
```

训练创建 `runtime/experiments/<experiment_id>/`，并在成功完成后更新 `runtime/latest_experiment.json`。每次运行都生成独立目录，迭代一清洗数据与报告不会被训练程序覆盖。要保持默认二维图，请使用默认训练命令。也可以在容器内调用 Agent：

```bash
docker compose run --rm --no-deps movielens-agent python3 /app/agent.py '预测 UserID 1 对 MovieID 1 是否打高分'
docker compose run --rm --no-deps movielens-agent python3 /app/agent.py '比较 K-Means 与层次聚类的用户群体'
docker compose run --rm --no-deps movielens-agent python3 /app/agent.py '比较 PCA 与 SVD 的二维降维结果'
```

## 分析口径

| 任务 | 算法与数据 | 结果 |
| --- | --- | --- |
| 数据质量 | Hadoop Streaming 对原始、清洗后数据使用同一套 `default-v1` 规则 | Accurate、Complete、Unique、Up-to-date、Consistent 及隔离、冲突和去重记录 |
| 分类 | 逻辑回归、决策树；标签为 `Rating >= 4` | 高分概率、阈值、Precision、Recall、F1、ROC-AUC 和混淆矩阵 |
| 聚类 | K-Means、Ward 层次聚类；训练期用户类型偏好画像 | 群体归属、大小、偏好、代表用户和轮廓系数 |
| 降维 | PCA、Truncated SVD；复用同一用户画像 | 坐标、解释方差比例、重构误差、运行时间、数据数组大小及聚类对照 |

分类训练事件的历史计数与均值只使用严格早于事件时间戳的评分，同秒事件不互相借用。验证集、测试集和交互预测都使用 T1 截止时的训练历史快照。逻辑回归与决策树在同一数据切分和特征上比较；验证集选择参数及阈值，测试集用于最终评价。交互预测若发现目标组合曾在训练期出现，会先从历史聚合中扣除该组合评分并给出样本内提示；该预测不能当成样本外评价。

用户画像只用训练期评分。未看过的电影类型对应中性偏差，不表示用户不喜欢该类型。聚类仅为训练期已有用户登记群体；可查询单个用户或“用户 1、2、3”这类集合在已有模型中的归属，不会为该子集重新聚类。层次聚类没有新用户直接预测接口。降维前后固定用户集合、群体数与对应聚类算法；两个空间的轮廓系数分别在各自空间计算，不能解释成相同量纲的信息保留率。PCA 和 SVD 在当前已中心化的稠密特征矩阵上可能得到近似相同的二维表示。

## 产物与版本

- `runtime/latest_report.json`：最近一次 Hadoop 清洗报告，包含清洗数据版本、规则版本及 T1/T2。
- `runtime/runs/<task_id>/`：独立的 Hadoop 任务文件、日志和清洗数据。
- `runtime/latest_experiment.json`：最近一次成功实验的索引。
- `runtime/experiments/<experiment_id>/report.json`：模型参数、时间边界、验证和测试指标、画像及降维报告。
- 同一实验目录中的 `*.joblib`：分类管道、聚类和降维模型；`*.csv`：T1 快照、用户画像、群体归属和坐标。
- `app/ml_experiment.py` 提供 `source_report()`、`read_clean()`；`app/ml_tools.py` 提供 `experiment()` 及六个独立算法函数，可供迭代三读取当前版本。

当前清洗版本、实验指标、课程材料核对和 DeepSeek 联网验证结果见 [迭代二实验报告](docs/迭代二实验报告.md)。已核对项目外的课程目录 `D:\南京大学\课程\大数据分析`：分类 PPT 与作业四材料明确包含并选用了逻辑回归和决策树；该目录目前未见聚类、降维 PPT，相关算法仍需在教师提供课件后核对。

汇报前的检查命令、需求核对状态和约 7 分钟的操作顺序见 [迭代二快捷检查与演示](docs/迭代二快捷检查与演示.md)。

## 验证

```bash
docker compose run --rm --no-deps movielens-agent python3 -m unittest -v test_iteration2
docker compose ps
curl -fsS http://localhost:8501/health
```

测试覆盖 T1/T2 和数据版本对应关系、同秒历史特征隔离、六个独立工具、Agent 路由、未知 ID 与非法维数。Hadoop 回归应在单独的 `RUNTIME_DIR` 下执行，以免改写当前默认报告。2026-10-21 已使用现有 `.env` 凭证实际调用 DeepSeek API，并验证英文用户请求能经联网意图识别调用两种聚类工具；这次验证不保证外部服务此后始终可用。
