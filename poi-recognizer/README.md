# POI-Recognizer

用 PaddleOCR 的 PP-OCRv5 mobile 训练两个模型：**det 找文字位置，rec 读出文字**。

整个训练就是下面六步，一个命令顺序执行：

```text
1. 准备 ReCTS 数据
2. ReCTS 训练 det ← 官方预训练检测权重
3. ReCTS 训练 rec ← 官方预训练识别权重
4. 准备 LSVT 数据（train_images + train_labels.json）
5. LSVT 微调 det  ← 第 2 步的最佳权重
6. LSVT 微调 rec  ← 第 3 步的最佳权重
```

第二阶段重新初始化优化器。任何一步失败就停止；重跑会新建目录，从第一步开始。

## 先配置数据

只需修改 [configs/train.json](configs/train.json)，路径相对 `poi-recognizer/`。默认目录：

```text
data/
├── photo-rects/train/
│   ├── img/    # ReCTS 有标注的训练图片
│   └── gt/     # 每张图一个 JSON
└── photo-lsvt/
    ├── train_images/
    └── train_labels.json
```

默认配置已指向上述数据目录。ReCTS 只使用官方 train 中的有标注数据，标注读取 `lines`，不使用 `chars`；无标注的官方 test 不参与。

常改参数只有这几个：

| 配置项 | 含义 | 默认值 |
| --- | --- | --- |
| `rects_epochs` | ReCTS 的 det、rec 各训练多少轮 | 50 |
| `epochs` | LSVT 的 det、rec 各训练多少轮 | 50 |
| `batch_size` | 每个模型的训练 batch | 8 |
| `learning_rate` | 学习率 | 0.0001 |
| `gpu` | 物理 GPU 编号 | 0 |

默认是 **4 个训练任务，每个 50 轮**。沿用已有 `myenv` 环境、本地 PaddleOCR 源码和官方权重，入口不会安装或下载。

## 启动

```bash
conda activate myenv
cd /data2/wangjiaji/AR/poi-recognizer
bash train.sh
```

需要先检查能否跑通时，手动执行 `bash train.sh --smoke`：每阶段 train/val 各至多四张可解码图片，每个模型一轮，batch=2。这不能代表训练效果。

终端显示当前第几步、初始权重、日志位置；每轮训练和验证用一条动态进度条显示，轮结束后只输出一行训练/验证指标摘要。警告和错误仍立即显示。日志不保存逐 batch 刷新；重定向到文件时不输出动态进度条。Ctrl+C 中断。训练配置统一维护在 `configs/train.json`。

## 训练结果在哪

每次运行写入独立目录 `data/outputs/training/train-<时间>/`：

```text
train-<时间>/
├── settings.json         # 本次配置
├── rects/                # 第一阶段的样例、配置、日志和最佳模型
├── data/
│   ├── examples/         # 约 1% 图像和配套检测/识别标签（含 train/val）
│   │   ├── images/       # 抽取的检测原图副本
│   │   ├── crops/        # 抽取的识别裁剪图
│   │   ├── det_*.txt / rec_*.txt
│   │   └── manifest.json # 原始/保留数量、抽样比例和 seed
│   └── …                 # report.json、skipped.jsonl、完整划分编号
├── det.yml / rec.yml     # 实际交给 PaddleOCR 的配置
├── det.log / rec.log     # 每轮摘要、指标和必要诊断
└── det/ / rec/           # 最终检测/识别模型
```

最终权重为 `det/best_accuracy.pdparams` 和 `rec/best_accuracy.pdparams`；ReCTS 的两份最佳权重也保留。数据过滤原因见各阶段 `data/report.json` 和 `data/skipped.jsonl`。

训练时仍使用完整数据，并临时保存检查点。**四个训练任务全部成功后自动精简**：每阶段、每个任务、每个 train/val 集合独立按 seed=42 抽取 1%（向上取整，非空集合至少一条），保留图像及配套标签，标签中的图片路径相对 `examples/`。删除全量裁剪缓存、全量训练/验证标签、`latest.*`、`iter_epoch_*`、优化器、重复的 `best_model/` 和模型目录内的重复配置；保留最佳权重及其小型 `.states`、实际训练配置、日志、数据划分和质量报告。原始数据不改动。失败或中断时不自动清理。

旧运行可在训练结束后手动精简，命令会检查四个任务的最终轮保存记录和最佳权重：

```bash
python -m poi_recognizer.retention --run data/outputs/training/train-<时间>
```

如果旧训练进程仍在运行，可加 `--wait-pid <训练主进程PID>`，等待该进程结束后再检查；不会重启或干扰训练。已加载代码的训练进程不会即时采用新的终端显示逻辑，新启动的训练才生效。

## 看实际识别效果

训练结束后手动运行：

```bash
bash evaluate.sh --run data/outputs/training/train-<时间>
```

打开输出报告的 `index.html`，看原图、预测文字、漏检、误检和错字。默认使用最终模型，至多抽取 100 张验证图。**精简目录只评估保留的约 1% 验证样例**，终端、报告和统计文件会标明这一范围；`--limit 0` 使用全部保留样例，不代表完整验证集。尚未精简的旧目录仍读取完整 `data/det_val.txt`。`curves.html` 查看 LSVT 训练曲线，`summary.json` 查看指标和耗时。更多选项见 `bash evaluate.sh --help`。

同一批图比较训练前后效果：

```bash
bash evaluate.sh --run data/outputs/training/train-<时间> --weights pretrained --limit 100 --seed 42
bash evaluate.sh --run data/outputs/training/train-<时间> --weights best --limit 100 --seed 42
```

训练中的 rec 准确率是在标注裁剪上计算的；整张街景的效果要看上述评估报告。报告按多边形 IoU≥0.5 匹配，文字去空格后整条比较，不是官方排行榜分数。耗时包含检测、裁剪、识别，磁盘读取另计；默认 FP32、检测最长边 960、识别 batch=6、无方向分类器。训练机速度不能代替路侧设备速度。

## 数据规则与代码检查

ReCTS 从官方 train 中按标注文件名排序、seed=42 打乱，再按 **7:1:2** 划分训练、验证和测试集。现有 20000 份标注对应 14000/2000/4000 张；各集合编号保存到 `rects/data/train_ids.txt`、`val_ids.txt`、`test_ids.txt`。测试集只预留编号，训练准备不读取其图片或标注，也不生成测试裁剪。

LSVT 固定编号：0–10499 训练，10500–11999 验证，12000–14999 测试预留。两种数据均先划分再检查图片，缺图不补号、不挪集合；测试集不参与训练和开发验证。

检测保留有效多边形，忽略文字为 `###`；识别使用文字区域裁剪，未知字符、控制字符、超长文字及无效裁剪整条跳过并记录。原图、原始标签、官方源码和初始权重只读；训练产物按上述规则精简。

环境：Python 3.10、Paddle GPU 3.1.1、本地 PaddleOCR v3.2.0。源码默认在 `data/vendor/PaddleOCR-sparse`，官方权重在 `data/models/pretrained`。

```bash
python -m pytest -q
ruff check src tests
black --check --workers 1 src tests
mypy
```
