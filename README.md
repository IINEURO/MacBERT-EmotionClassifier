# MacBERT-EmotionClassifier

基于 `hfl/chinese-macbert-base` 的中文社交媒体情绪分类项目，支持训练、命令行预测和 Gradio 可视化推理。

## Features

- MacBERT + `AutoModelForSequenceClassification`
- 支持类别权重、FP16（CUDA 可用时）
- 支持 `usual / virus / all` 三种数据子集训练
- 支持单条与批量 JSON 预测
- 提供紫色渐变风格 Gradio UI

## Project Structure

```text
MacBERT-EmotionClassifier/
├── .github/workflows/ci.yml
├── configs/train_macbert_base.yaml
├── src/macbert_emotion_classifier/
├── gradio_app.py
├── train.py
├── predict.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Environment Setup

```bash
cd MacBERT-EmotionClassifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Layout

默认读取 `./WBDataSet`，目录结构如下：

```text
WBDataSet/
├── train/
│   ├── usual_train.txt
│   └── virus_train.txt
└── eval（刷榜数据集）/
    ├── usual_eval_labeled.txt
    └── virus_eval_labeled.txt
```

你也可以通过参数覆盖：

```bash
python3 train.py --dataset-root /path/to/WBDataSet
```

## Training

使用默认配置训练：

```bash
python3 train.py
```

常用覆盖参数示例：

```bash
python3 train.py --dataset-subset usual
python3 train.py --no-fp16
python3 train.py --max-length 128
```

最佳模型默认输出到：

```text
outputs/macbert-base-wbdataset/best_model
```

## Prediction (CLI)

单条文本预测：

```bash
python3 predict.py \
  --checkpoint ./outputs/macbert-base-wbdataset/best_model \
  --text "今天心情非常好，终于完成实验了"
```

批量 JSON 预测（输入为含 `content` 字段的列表）：

```bash
python3 predict.py \
  --checkpoint ./outputs/macbert-base-wbdataset/best_model \
  --input-json /path/to/input.json \
  --output-json ./outputs/predictions.json
```

## Gradio UI

```bash
python3 gradio_app.py \
  --checkpoint ./outputs/macbert-base-wbdataset/best_model \
  --host 0.0.0.0 \
  --port 7860
```

## Labels

默认类别顺序（`label2id`）：

1. angry
2. fear
3. happy
4. neutral
5. sad
6. surprise

## GitHub Upload Guide

本仓库已包含 `.gitignore`，默认不会提交以下内容：

- `.venv/`
- `outputs/`
- `__pycache__/`
- 本地数据目录 `WBDataSet/`

上传步骤：

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## License

MIT License. See `LICENSE`.
