from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from macbert_emotion_classifier.predictor import EmotionPredictor


DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "outputs/macbert-base-wbdataset/best_model")
_PREDICTOR_CACHE: Dict[Tuple[str, int], EmotionPredictor] = {}


CUSTOM_CSS = """
body {
  background: linear-gradient(120deg, #4a1d96 0%, #7c3aed 48%, #a855f7 100%);
}
.gradio-container {
  max-width: 1000px !important;
  margin: 0 auto;
  background: transparent !important;
}
.app-shell {
  background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.34);
  border-radius: 16px;
  backdrop-filter: blur(4px);
  padding: 18px;
}
.card {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 14px;
  border: 1px solid rgba(124, 58, 237, 0.25);
}
.app-title h1,
.app-title p {
  color: #ffffff !important;
}
@media (max-width: 768px) {
  .app-shell {
    padding: 10px;
    border-radius: 12px;
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio UI for MacBERT emotion classifier")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def _get_predictor(checkpoint_dir: str, max_length: int) -> EmotionPredictor:
    ckpt = str(Path(checkpoint_dir).expanduser().resolve())
    path = Path(ckpt)
    if not path.exists():
        raise ValueError(f"Checkpoint path does not exist: {path}")

    key = (ckpt, int(max_length))
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = EmotionPredictor(ckpt, max_length=int(max_length))
    return _PREDICTOR_CACHE[key]


def _build_label_prob_rows(predictor: EmotionPredictor, probs: Sequence[float]) -> List[List[float | str]]:
    id2label = predictor.model.config.id2label
    rows: List[List[float | str]] = []
    keys = list(id2label.keys())
    if not keys:
        return rows

    if isinstance(keys[0], str):
        ordered = sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])
    else:
        ordered = sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])

    for idx, label in ordered:
        prob = float(probs[idx]) if idx < len(probs) else 0.0
        rows.append([str(label), round(prob, 6)])
    return rows


def predict_single(
    text: str,
    checkpoint_dir: str,
    max_length: int,
    top_k: int,
) -> tuple[str, float, Dict[str, float], List[List[float | str]]]:
    if not text or not text.strip():
        raise gr.Error("请输入要预测的文本。")

    try:
        predictor = _get_predictor(checkpoint_dir, int(max_length))
        result = predictor.predict_texts([text.strip()], batch_size=1)[0]
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    rows = _build_label_prob_rows(predictor, result["probabilities"])
    rows_sorted = sorted(rows, key=lambda x: float(x[1]), reverse=True)
    top_rows = rows_sorted[: int(top_k)]
    score_map = {str(label): float(score) for label, score in rows_sorted}

    return result["pred_label"], float(result["confidence"]), score_map, top_rows


def predict_batch(
    text_block: str,
    checkpoint_dir: str,
    max_length: int,
    batch_size: int,
) -> List[List[float | str]]:
    lines = [line.strip() for line in text_block.splitlines() if line.strip()]
    if not lines:
        raise gr.Error("请至少输入一行文本。")

    try:
        predictor = _get_predictor(checkpoint_dir, int(max_length))
        preds = predictor.predict_texts(lines, batch_size=int(batch_size))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    rows: List[List[float | str]] = []
    for pred in preds:
        rows.append([pred["content"], pred["pred_label"], round(float(pred["confidence"]), 6)])
    return rows


def build_demo(default_checkpoint: str) -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="MacBERT Emotion UI") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                # MacBERT Emotion Classifier
                中文社交媒体情绪分类界面
                """,
                elem_classes=["app-title"],
            )

            with gr.Row(equal_height=True):
                checkpoint_dir = gr.Textbox(
                    label="Checkpoint 路径",
                    value=default_checkpoint,
                    scale=4,
                    elem_classes=["card"],
                )
                max_length = gr.Slider(
                    label="max_length",
                    minimum=32,
                    maximum=512,
                    step=1,
                    value=128,
                    scale=2,
                    elem_classes=["card"],
                )

            with gr.Tabs():
                with gr.Tab("单条预测"):
                    with gr.Column(elem_classes=["card"]):
                        single_text = gr.Textbox(
                            label="输入文本",
                            lines=5,
                            placeholder="例如：今天终于把实验跑通了，心情很好。",
                        )
                        top_k = gr.Slider(
                            label="Top-K 概率展示",
                            minimum=1,
                            maximum=6,
                            step=1,
                            value=6,
                        )
                        predict_btn = gr.Button("开始预测", variant="primary")

                    with gr.Row():
                        pred_label = gr.Textbox(label="预测标签", interactive=False, elem_classes=["card"])
                        confidence = gr.Number(label="置信度", interactive=False, elem_classes=["card"])

                    score_label = gr.Label(label="类别概率分布", num_top_classes=6, elem_classes=["card"])
                    score_table = gr.Dataframe(
                        headers=["label", "probability"],
                        datatype=["str", "number"],
                        label="Top-K 结果",
                        interactive=False,
                        elem_classes=["card"],
                    )

                    predict_btn.click(
                        fn=predict_single,
                        inputs=[single_text, checkpoint_dir, max_length, top_k],
                        outputs=[pred_label, confidence, score_label, score_table],
                    )

                with gr.Tab("批量预测"):
                    with gr.Column(elem_classes=["card"]):
                        batch_input = gr.Textbox(
                            label="按行输入文本",
                            lines=10,
                            placeholder="每行一条文本，点击后批量预测。",
                        )
                        batch_size = gr.Slider(
                            label="batch_size",
                            minimum=1,
                            maximum=128,
                            step=1,
                            value=32,
                        )
                        batch_btn = gr.Button("批量预测", variant="primary")

                    batch_output = gr.Dataframe(
                        headers=["content", "pred_label", "confidence"],
                        datatype=["str", "str", "number"],
                        label="批量预测结果",
                        interactive=False,
                        elem_classes=["card"],
                    )

                    batch_btn.click(
                        fn=predict_batch,
                        inputs=[batch_input, checkpoint_dir, max_length, batch_size],
                        outputs=[batch_output],
                    )

            gr.Examples(
                examples=[
                    ["今天太开心了，终于拿到满意的实验结果！"],
                    ["最近压力很大，感觉有点焦虑。"],
                    ["这个消息太突然了，我有点震惊。"],
                ],
                inputs=[single_text],
                label="示例输入",
            )

    return demo


if __name__ == "__main__":
    args = parse_args()
    app = build_demo(args.checkpoint)
    app.launch(server_name=args.host, server_port=args.port)
