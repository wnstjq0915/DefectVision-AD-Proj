## 실험 결과: MVTec AD + VisA

> Colab에서 MVTec AD와 VisA의 category별 모델을 학습/평가한 결과입니다. 대용량 checkpoint와 원본 데이터는 Google Drive에 보관하고, GitHub에는 README용 CSV/PNG 산출물만 저장합니다.

### 모델 선택 기준

정확도만 기준으로 고르면 class imbalance가 있는 anomaly detection에서 모델을 잘못 선택할 수 있으므로, 아래 가중 평균을 사용했습니다.

```text
selection_score = 0.40*AUROC + 0.25*F1 + 0.25*Pixel_AUROC + 0.10*Accuracy
계산 불가능한 지표는 제외하고 남은 지표의 가중치 합으로 재정규화
```

### 성능 시각화

#### 모델별 평균 선택 점수
![모델별 평균 선택 점수](docs/assets/results/fig_model_average_selection_score.png)

#### 데이터셋/모델별 평균 image AUROC
![데이터셋/모델별 평균 image AUROC](docs/assets/results/fig_dataset_model_auroc.png)

#### category별 최고 모델
![category별 최고 모델](docs/assets/results/fig_best_model_by_category.png)

#### category × model 선택 점수 heatmap
![category × model 선택 점수 heatmap](docs/assets/results/fig_selection_score_heatmap.png)

### Top 10 Leaderboard

| Rank | Dataset | Category | Model | Selection | Accuracy | Precision | Recall | F1 | AUROC | Pixel AUROC |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mvtec | bottle | patchcore | 0.990 | 0.988 | 1.000 | 0.984 | 0.992 | 0.998 | 0.977 |
| 2 | mvtec | bottle | autoencoder | 0.547 | 0.506 | 0.806 | 0.460 | 0.586 | 0.567 | 0.493 |

### Heatmap sample panels

![sample](docs/assets/results/samples/mvtec_bottle_patchcore_0_000_panel.png)

![sample](docs/assets/results/samples/mvtec_bottle_patchcore_1_000_panel.png)

### 저장된 산출물

- `docs/assets/results/all_results.csv`: 전체 실험 결과
- `docs/assets/results/leaderboard.csv`: selection score 기준 전체 순위
- `docs/assets/results/best_by_category.csv`: category별 최고 모델
- `docs/assets/results/*.png`: README용 성능 그래프
- `docs/assets/results/samples/*.png`: heatmap sample panel
