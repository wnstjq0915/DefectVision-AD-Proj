# Project Plan

## Scope

DefectVision-AD detects product image anomalies and visualizes suspected defect
regions. The baseline implementation uses reconstruction error from a PyTorch
AutoEncoder. A lightweight PatchCore-style memory bank is included for feature
based comparison.

## Milestones

1. Dataset loaders for MVTec AD and VisA.
2. Image preprocessing with resize, normalization, grayscale, blur, edge, and
   histogram options.
3. AutoEncoder baseline training and threshold estimation.
4. Inference outputs: score, prediction, heatmap, binary mask, and overlay.
5. Evaluation metrics for image-level classification and optional pixel AUROC.
6. Streamlit demo for checkpoint-based image inspection.

## Commands

```bash
pip install -r requirements.txt
python -m src.train --config configs/model_autoencoder.yaml
python -m src.inference --checkpoint outputs/checkpoints/autoencoder.pt --input data/raw/mvtec/bottle/test
python -m src.evaluate --config configs/model_autoencoder.yaml --checkpoint outputs/checkpoints/autoencoder.pt
streamlit run app/streamlit_app.py
```
