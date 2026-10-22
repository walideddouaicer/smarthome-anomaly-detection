# Smart Home Anomaly Detection

A deep learning system that watches smart-home camera footage and flags anomalous events such as falls, accidents, intrusions and wildlife. Built as the final project for the *Artificial Intelligence & Computer Vision* course (S7, ENSA Kénitra, January 2026).

**Results:** 80.0% accuracy · 0.85 ROC-AUC on the held-out test set.

## How it works

The model is a hybrid spatial-temporal network:

1. **Spatial features ("the eyes"):** 16 frames are sampled evenly from each clip, resized to 224×224, and passed through an ImageNet-pretrained **ResNet50** with its classifier removed. Each frame becomes a 2048-d feature vector.
2. **Temporal model ("the brain"):** the sequence of 16 vectors goes into a 2-layer **bidirectional LSTM** (hidden size 256, dropout 0.4). A small MLP head outputs an anomaly probability.
3. **Fighting overfitting:** phase 1 overfit the small dataset, so phase 2 doubles the training data with horizontal-flip augmentation and retrains with early stopping and LR scheduling. Test-time augmentation was also tried as an ablation.

The decision threshold used by the demo app is `0.6774`.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/SmartHomeBench.ipynb` | Final notebook: data prep, feature extraction, phase 1 & 2 training, TTA ablation, evaluation |
| `notebooks/SmartHomeBench_submission.ipynb` | Version submitted with the report |
| `notebooks/SmartHomeBench_first.ipynb` | Early exploration notebook |
| `app/app.py` | Streamlit demo: upload an MP4 and get an anomaly score |
| `app/smarthome_phase2_augmented.pth` | Trained Bi-LSTM weights (phase 2, augmented) |
| `demo_videos/` | Short labeled clips (`normal_*`, `abnormal_*`) for trying the demo |
| `results/test_results_arrays.npz` | Test-set `scores` and `labels` (721 clips) used for the ROC / confusion-matrix plots |
| `report/report.pdf` | Full project report |

## Running the demo

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

The trained weights (`app/smarthome_phase2_augmented.pth`, ~25 MB) are included, so the demo runs out of the box. Run it from the `app/` folder so it can find them.

## Dataset

The project uses the [SmartHome-Bench](https://github.com/Xinyi-0724/SmartHome-Bench-LLM) dataset of smart-home camera clips with normal/abnormal annotations. The dataset is not redistributed here; get it from the original authors. The notebooks expect `raw_video/` and `Video_Annotation.csv` from the dataset.

## Credits

- **Walid Eddouaicer**: design, implementation, training, evaluation, demo app and report
- **Adil Dani**: sourcing the dataset and organizing the project files
