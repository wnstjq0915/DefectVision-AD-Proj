from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "docs" / "assets" / "results"
DEFAULT_DEPLOY_ROOT = ROOT / "deploy"
DEPLOY_ROOT = Path(os.environ.get("DEFECTVISION_DEPLOY_ROOT", str(DEFAULT_DEPLOY_ROOT)))
MODEL_ROOT = Path(os.environ.get("DEFECTVISION_MODEL_ROOT", str(DEPLOY_ROOT / "models")))
REGISTRY_PATH = Path(os.environ.get("DEFECTVISION_REGISTRY", str(DEPLOY_ROOT / "model_registry.json")))
SAMPLES_MANIFEST_PATH = Path(os.environ.get("DEFECTVISION_SAMPLES", str(DEPLOY_ROOT / "demo_samples_manifest.json")))
SAMPLES_ROOT = DEPLOY_ROOT

st.set_page_config(
    page_title="DefectVision-AD Demo",
    page_icon="🔎",
    layout="wide",
)


def read_json(path: Path, default: Any):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


@st.cache_data(show_spinner=False)
def load_tables():
    tables = {}
    for name in ["all_results", "leaderboard", "best_by_category", "model_summary", "category_summary"]:
        path = RESULT_DIR / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()
    return tables


@st.cache_data(show_spinner=False)
def load_registry_and_samples():
    registry = read_json(REGISTRY_PATH, {"models": []})
    samples = read_json(SAMPLES_MANIFEST_PATH, [])
    return registry, samples


@st.cache_resource(show_spinner=True)
def load_cached_predictor(checkpoint_path: str, image_size: int | None = None):
    from src.inference import load_predictor
    return load_predictor(checkpoint_path, device="auto", image_size=image_size)


def resolve_checkpoint(entry: dict[str, Any]) -> Path:
    rel = entry.get("checkpoint_relative_path")
    if rel:
        # registry stores models/mvtec/category/model.pt; MODEL_ROOT points to deploy/models.
        rel_path = Path(rel)
        if rel_path.parts and rel_path.parts[0] == "models":
            rel_path = Path(*rel_path.parts[1:])
        path = MODEL_ROOT / rel_path
        if path.exists():
            return path
    drive_path = entry.get("checkpoint_drive_path")
    if drive_path and Path(drive_path).exists():
        return Path(drive_path)
    return MODEL_ROOT / "missing.pt"


def model_entries_for_category(registry: dict[str, Any], category: str):
    return [m for m in registry.get("models", []) if m.get("category") == category]


def sample_path(sample: dict[str, Any]) -> Path:
    p = Path(sample.get("sample_path", ""))
    if p.is_absolute():
        return p
    return SAMPLES_ROOT / p


def show_metric_cards(best_df: pd.DataFrame):
    if best_df.empty:
        st.info("아직 best_by_category.csv가 없습니다. Colab 01 노트북을 먼저 실행하세요.")
        return
    cols = st.columns(4)
    cols[0].metric("평가 category 수", len(best_df))
    if "accuracy" in best_df:
        cols[1].metric("평균 Accuracy", f"{pd.to_numeric(best_df['accuracy'], errors='coerce').mean():.3f}")
    if "f1" in best_df:
        cols[2].metric("평균 F1", f"{pd.to_numeric(best_df['f1'], errors='coerce').mean():.3f}")
    if "selection_score" in best_df:
        cols[3].metric("평균 Selection score", f"{pd.to_numeric(best_df['selection_score'], errors='coerce').mean():.3f}")


def display_result_figures():
    figure_files = [
        "fig_model_average_selection_score.png",
        "fig_category_average_accuracy.png",
        "fig_best_model_by_category.png",
        "fig_selection_score_heatmap.png",
        "fig_accuracy_by_category_model.png",
    ]
    existing = [RESULT_DIR / f for f in figure_files if (RESULT_DIR / f).exists()]
    if not existing:
        st.info("README/Streamlit용 결과 그래프가 아직 없습니다. Colab 01, 02 노트북을 실행하세요.")
        return
    for path in existing:
        st.image(str(path), caption=path.name, use_container_width=True)


def predict_and_show(entry: dict[str, Any], image_path: Path):
    checkpoint_path = resolve_checkpoint(entry)
    if not checkpoint_path.exists():
        st.error(
            "선택한 category의 checkpoint를 찾지 못했습니다.\n\n"
            f"찾은 경로: `{checkpoint_path}`\n\n"
            "EC2에서는 deploy bundle을 압축 해제하고 `DEFECTVISION_DEPLOY_ROOT` 또는 `DEFECTVISION_MODEL_ROOT`를 설정해야 합니다."
        )
        return

    image_size = int(entry.get("image_size", 256) or 256)
    predictor = load_cached_predictor(str(checkpoint_path), image_size=image_size)
    pred = predictor.predict_image(image_path)

    from src.visualize import make_heatmap, overlay_heatmap

    score = float(pred["score"])
    threshold = float(pred["threshold"])
    label = pred["label"]
    anomaly_map = pred["anomaly_map"]
    overlay = overlay_heatmap(image_path, anomaly_map)
    heatmap = make_heatmap(anomaly_map)

    cols = st.columns(3)
    cols[0].image(str(image_path), caption="선택 이미지", use_container_width=True)
    cols[1].image(heatmap, caption="Anomaly map", use_container_width=True)
    cols[2].image(overlay, caption="Heatmap overlay", use_container_width=True)

    st.subheader("판정 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "불량 의심" if label == "anomaly" else "정상")
    c2.metric("Anomaly score", f"{score:.6f}")
    c3.metric("Threshold", f"{threshold:.6f}")


def main():
    st.title("DefectVision-AD: 산업 제품 결함 이상탐지 데모")
    st.caption("MVTec AD category별 모델 성능 비교와 랜덤 test image 추론 데모")

    tables = load_tables()
    registry, samples = load_registry_and_samples()
    best_df = tables.get("best_by_category", pd.DataFrame())
    all_df = tables.get("all_results", pd.DataFrame())

    tab_dashboard, tab_demo, tab_files = st.tabs(["성능 대시보드", "랜덤 이미지 추론", "배포 파일 상태"])

    with tab_dashboard:
        st.header("모델·category별 성능 비교")
        show_metric_cards(best_df)
        if not all_df.empty:
            st.subheader("전체 평가 결과")
            st.dataframe(all_df, use_container_width=True)
        if not best_df.empty:
            st.subheader("category별 최고 모델")
            st.dataframe(best_df, use_container_width=True)
        st.subheader("README용 시각화")
        display_result_figures()

    with tab_demo:
        st.header("학습에 사용하지 않은 test 이미지 랜덤 추론")
        if not registry.get("models"):
            st.warning("model_registry.json이 없습니다. Colab 01/02/03 노트북을 먼저 실행하세요.")
            return

        categories = sorted({m["category"] for m in registry.get("models", [])})
        category = st.selectbox("Category", categories)
        entries = model_entries_for_category(registry, category)
        if not entries:
            st.error("선택한 category의 모델 registry가 없습니다.")
            return

        entry_labels = [f"{e['model']} | score={e.get('selection_score', None)}" for e in entries]
        selected_idx = st.selectbox("사용할 모델", range(len(entries)), format_func=lambda i: entry_labels[i])
        entry = entries[selected_idx]

        category_samples = [s for s in samples if s.get("category") == category]
        if not category_samples:
            st.warning("이 category의 demo sample이 없습니다. Colab 01에서 sample pool을 생성하세요.")
            return

        session_key = f"random_samples_{category}"
        if st.button("랜덤 test 이미지 10개 뽑기", type="primary") or session_key not in st.session_state:
            st.session_state[session_key] = random.sample(category_samples, k=min(10, len(category_samples)))

        selected_samples = st.session_state[session_key]
        labels = [f"{i+1:02d}. {Path(s['sample_path']).name} | 실제={s.get('label_name')} | defect={s.get('defect_type')}" for i, s in enumerate(selected_samples)]
        pick = st.selectbox("이미지 선택", range(len(selected_samples)), format_func=lambda i: labels[i])
        sample = selected_samples[pick]
        img_path = sample_path(sample)

        if not img_path.exists():
            st.error(f"sample image가 없습니다: {img_path}")
            return

        st.info(f"실제 label: {sample.get('label_name')} / defect_type: {sample.get('defect_type')}")
        predict_and_show(entry, img_path)

    with tab_files:
        st.header("배포 파일 상태")
        st.write("Repository root:", ROOT)
        st.write("Result dir:", RESULT_DIR, RESULT_DIR.exists())
        st.write("Deploy root:", DEPLOY_ROOT, DEPLOY_ROOT.exists())
        st.write("Model root:", MODEL_ROOT, MODEL_ROOT.exists())
        st.write("Registry:", REGISTRY_PATH, REGISTRY_PATH.exists())
        st.write("Samples manifest:", SAMPLES_MANIFEST_PATH, SAMPLES_MANIFEST_PATH.exists())
        st.write("모델 수:", len(registry.get("models", [])))
        st.write("sample 수:", len(samples))


if __name__ == "__main__":
    main()
