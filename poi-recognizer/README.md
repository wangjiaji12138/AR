# POI-Recognizer

PP-OCRv5 mobile：det 找文字位置，rec 读取文字。一个命令依次执行 **ReCTS 检测/识别训练 → LSVT 检测/识别微调**。ReCTS 从官方预训练权重开始，LSVT 从本次 ReCTS 最佳权重开始，第二阶段重置优化器；任何任务失败即停止。

2026-10-18 按用户审视后的范围重构：统一预处理、简化训练、GPU 执行优化、两个数据集各取 1/10 后重新划分。此前完整评估重建、20% 微调、多组对照、额外增强、长度扩展和自动早停方案不再实施。[to_check.md](to_check.md) 保留为历史检查记录，其中旧数据比例和划分不适用于本次重训。

## 训练

使用已有 Conda `myenv`（Python 3.10、Paddle GPU 3.1.1）和本地 PaddleOCR v3.2.0，不安装或下载依赖。

```bash
conda activate myenv
cd /data2/wangjiaji/AR/poi-recognizer
bash train.sh
```

参数统一放在 [configs/train.json](configs/train.json)：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` | 50 | 四个训练任务各 50 轮 |
| `sample_fraction` | 0.1 | 各数据集抽取的原图比例 |
| `batch_size` | 16 | 训练 batch 和识别验证 batch；检测验证为 1 |
| `learning_rate` | 0.0001 | 沿用简单统一学习率，无参数搜索 |
| `amp` | true | Paddle 原生 O1 混合精度和动态 loss scaling；关闭可回到 FP32 |
| `gpu` | 1 | 物理 GPU 编号，可用 `bash train.sh --gpu 2` 覆盖 |

数据路径相对本项目目录：

```text
data/photo-rects/train/img/       # ReCTS 图片
data/photo-rects/train/gt/        # 逐图 JSON，使用 lines
data/photo-lsvt/train_images/    # 当前本地提供 gt_0 至 gt_14999
data/photo-lsvt/train_labels.json
data/vendor/PaddleOCR-sparse/    # 官方源码，只读
data/models/pretrained/         # 官方 det/rec 权重，只读
```

`bash train.sh --smoke` 使用同样的抽样划分，每阶段 train/val 各至多四张可解码原图，四个任务各一轮、batch=2。它验证训练衔接，不代表模型质量。训练和验证显示进度，文件保留每轮摘要和诊断；Ctrl+C 中断，重跑会新建目录并重新开始。

## 1/10 抽样与重新划分

先对原图 ID 排序，再用 seed=42 打乱，取 `floor(原图数量 × 0.1)` 张，然后按 7:1:2 划分。两种数据独立抽样；检测和识别使用相同原图集合。

| 数据集 | 抽样来源 | 选中原图 | train | val | test |
| --- | --- | --- | --- | --- | --- |
| ReCTS | 官方有标注 train 的 20000 张 | 2000 | 1400 | 200 | 400 |
| LSVT | 本地提供的编号 0–14999，共 15000 张 | 1500 | 1050 | 150 | 300 |

LSVT 标签文件有 30000 条图片记录，但本地图片仅提供前 15000 张，本轮以这 15000 张为抽样来源。这里的 1/10 **包含重新划分出的训练、验证、测试集合**，并非只缩减旧训练集。识别裁剪数量由选中原图中的有效实例决定，不要求恰好为旧裁剪数量的 1/10。

先抽样、划分，再检查图片和标签。缺图/坏图、无效几何、忽略文字及识别不可训练标签记录到报告，不补样、不跨集合挪动。测试只保留 ID，不生成标签或裁剪，不参与训练和调参。原始数据和旧产物只读。

**本次划分与历史划分不同**，指标不能直接与旧验证结果比较。重新训练从官方预训练权重开始，不使用曾在旧训练集上训练过的项目权重来评估新的留出集。

## 统一预处理与 GPU 优化

检测训练内验证和整图推理都使用最长边上限 960、相同官方缩放/后处理规则；训练仍保留官方随机裁剪增强。

识别训练、验证和整图推理共用 [preprocessing.py](src/poi_recognizer/preprocessing.py)：高度 48，等比例缩放，宽度至少 320 并向上对齐至 8，右侧补零。宽文本按需扩宽，训练/验证批内补齐到最大宽度；推理沿用官方宽高比排序减少 padding。未设硬截断宽度，极宽输入会增加显存消耗，可先减小 batch。

标注框和预测框使用同一裁剪函数：四点框透视裁剪，多点框使用最小旋转外接矩形；高宽比达到 1.5 时旋转。它不能拉直弯曲文字。识别保持原字典、23 字符标签上限及保守的 40 时间步 CTC 过滤，不扩展模型/标签长度；动态宽度解决图像压缩问题，不自动放开标签长度。仍保留 `RecAug`，不恢复拼接、CopyPaste 或增加扰动。

GPU 改动保持简单：训练启用原生 O1 混合精度；识别验证从逐条改为批量；保留数据加载 worker 和 CPU 线程限制；整图推理复用已加载的预测器，默认不额外进行每阶段 GPU 全局同步。是否提速、质量是否变化以真实运行结果为准，A800 结果不代表路端设备。

## 输出

每次运行保存到新的 `data/outputs/training/train-<时间>/`：

```text
settings.json                   # 本次配置
rects/data/                     # ReCTS 划分、标签、裁剪、过滤报告
rects/det.yml / rec.yml          # 实际训练配置
rects/det.log / rec.log
rects/det/ / rec/                # ReCTS 最佳权重与原生检查点
data/                           # LSVT 划分、标签、裁剪、过滤报告
det.yml / rec.yml
det.log / rec.log
det/ / rec/                     # 最终模型
```

最佳权重为各任务目录的 `best_accuracy.pdparams`。每个 `data/` 保存 `train_ids.txt / val_ids.txt / test_ids.txt`、`det_*.txt / rec_*.txt`、`report.json` 和 `skipped.jsonl`。

**训练结束后不再自动抽取 1% 或删除标签/裁剪**，这次数据本身已缩小，保留即可复查。历史 `retention` 工具仍在，仅供旧运行兼容，不属于新训练流程。

## 查看效果

```bash
bash evaluate.sh --run data/outputs/training/train-<时间> --gpu 1 --limit 0
```

报告 `index.html` 展示原图、预测、漏检、误检和错字，`curves.html` 为训练曲线。新运行的 `--limit 0` 使用本次重新划分后的全部有效验证图；默认 `--limit 100` 最多 100 张。历史已精简目录仍明确标记为约 1% 保留样例，未增加旧数据恢复入口。

检测与整图指标沿用现有多边形 IoU≥0.5 一对一匹配，文字去普通空格后整段比较；训练日志中的 rec Acc 是 GT 裁剪准确率，不能代替整图质量。没有新增店名或地图 POI 匹配计分。

推理默认 FP32、识别 batch=16，可用 `--rec-batch 6` 调整。`--det-side-len` 仅供显式实验，改变后不再与默认训练验证分辨率一致。`--profile-stages` 开启额外同步以获得分阶段计时；默认仅报告完整 OCR 和读图耗时。官方预测器将结果复制到 CPU 后返回，因此完整 OCR 计时包含已完成的计算与前后处理；导出、初始化、预热、画图和报告不计入。每次评估仍独立导出、初始化一次，图片间复用预测器，不新增服务框架或 TensorRT 路径。

## 检查

```bash
python -m pytest -q
ruff check src tests
black --check --workers 1 src tests
mypy
```

合成测试覆盖抽样隔离、缩放/补齐像素一致性、检测验证参数、训练衔接和现有评估行为。真实模型冒烟、训练与性能结果需另行说明，不能以单测替代。
