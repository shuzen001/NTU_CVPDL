# CVPDL HW2 — 長尾物件偵測訓練指南

本專案以 `HW2_solution.ipynb` 為核心，從原始資料集出發進行 YOLOv12m 長尾偵測模型的訓練與推論。以下說明如何建立環境、準備資料、進行訓練，以及產出競賽格式的預測結果（`submission.csv`）。

## 環境需求
- Python 3.10 以上（建議使用 Conda 管理環境）。
- NVIDIA GPU（建議 12 GB 以上記憶體）與相容的 CUDA 驅動。
- 作業系統：Linux / Windows / WSL2 / macOS（GPU 需視平台而定）。

### 建立虛擬環境
```bash
conda create -n cvpdl-hw2 python=3.10
conda activate cvpdl-hw2
pip install -r requirements.txt
```

> ⚠️ 若需要特定 CUDA 版本的 PyTorch，請依照 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 先安裝對應的 `torch` 與 `torchvision`，再執行 `pip install -r requirements.txt --no-deps` 補齊其餘套件。

## 資料結構
專案假設目前工作目錄為 `H2/CVPDL_hw2/CVPDL_hw2/`，且資料集已放置於：

```
CVPDL_hw2/
├── train/
│   ├── imgXXXX.png      # 訓練影像
│   └── imgXXXX.txt      # 對應標註（class,x,y,w,h）
├── test/
│   └── imgXXXX.png      # 測試影像（無標註）
└── src/HW2_solution.ipynb
```

Notebook 會於專案底下自動建立：
- `yolo_dataset/`：YOLO 資料夾結構（含 train/val/test）。
- `models/yolov12m.yaml`：自訂模型結構（已提供）。
- `runs/<run_name>/`：Ultralytics 的訓練結果（含 `weights/best.pt`）。
- `submission.csv`：推論後匯出的競賽檔案。

## Notebook 執行流程
1. **啟動 Jupyter**
   ```bash
   jupyter lab  # 或 jupyter notebook
   ```
   開啟 `src/HW2_solution.ipynb`，依序執行下列章節。

2. **Section 2 – 設定與檢查**
   - `cfg = TrainConfig()` 會自動指向目前工作目錄並檢查 `train/`、`test/` 是否存在。
   - 可在此調整超參數（例如 `image_size`、`num_epochs`、`use_kfold` 等）。

3. **Section 3 – 資料集轉換與重採樣**
   - 執行 `dataset_metadata = load_dataset_metadata(cfg)` 與 `prepare_yolo_dataset(...)`，建立 YOLO 標註與目錄結構。
   - 啟用 `cfg.use_oversampling` / `cfg.use_undersampling` 時，`hybrid_resample_dataset` 會對長尾類別進行重採樣，並重新寫入 YOLO 標註。
   - 此步驟完成後會得到 `cfg.dataset_yaml`（預設 `yolo_dataset/yolov12_dataset.yaml`）。

4. **Section 5 – 載入模型骨架**
   - `model = YOLO(cfg.model_yaml)` 會以 `models/yolov12m.yaml` 建立 YOLOv12m 自訂結構，從隨機權重開始訓練。

5. **Section 6 – 開始訓練**
   - 主要訓練命令如下，可依需求調整 epoch、batch size、影像尺寸、增強策略：
     ```python
     train_results = model.train(
         data=str(cfg.dataset_yaml),
         epochs=cfg.num_epochs,
         imgsz=cfg.image_size,
         batch=cfg.batch_size,
         lr0=cfg.learning_rate,
         optimizer="AdamW",
         weight_decay=cfg.weight_decay,
         cls=0.5,
         momentum=0.9,
         device=0 if torch.cuda.is_available() else "cpu",
         project=str(cfg.runs_dir),
         name=cfg.run_name,
         pretrained=False,
         workers=cfg.num_workers,
         seed=cfg.seed,
         patience=cfg.patience,
         warmup_epochs=cfg.warmup_epochs,
         verbose=True,
         amp=True,
         cos_lr=True,
         close_mosaic=10,
         translate=0.1,
         scale=0.7,
         shear=0.0,
         perspective=0.0,
         flipud=0.0,
         fliplr=0.5,
         mosaic=0.2,
         mixup=0.0,
         copy_paste=0.5,
         hsv_h=0.015,
         hsv_s=0.7,
         hsv_v=0.4,
     )
     ```
   - 完成後可於 `runs/<run_name>/weights/` 取得 `best.pt`（或 `last.pt`）。
   - 若啟用 `cfg.use_kfold = True`，Notebook 會呼叫 `run_kfold_training` 逐折訓練並保存各折權重。

6. **Section 8 – 產生推論與 submission**
   - 先以 `best_model = YOLO(best_weights_path)` 讀取權重。
   - 執行 `run_inference_to_submission(best_model, cfg)` 產生 `submission.csv`，內容符合 Kaggle 所需的 `PredictionString` 格式。
   - 若有多個 `submission.csv`，可使用 Section 9 的 `merge_submission_csvs` 進行 CSV 融合。

7. **批次執行（選用）**
   - 如需一次完成整個流程，可使用：
     ```bash
     jupyter nbconvert --execute --to notebook --inplace src/HW2_solution.ipynb
     ```
     請先確認 GPU/時間成本足夠；此指令將依序執行 Notebook 全部區塊。

## 輸出檔案位置
- `runs/<run_name>/weights/best.pt`：最佳模型權重。
- `runs/<run_name>/results.csv`：Ultralytics 訓練指標（可於 Section 6.1 視覺化）。
- `submission.csv`（或 `submission_fused_*.csv`）：最終提交檔案。

## 常見調整建議
- 記憶體不足時可降低 `cfg.image_size` 或 `cfg.batch_size`。
- 若不需長尾重採樣，將 `cfg.use_oversampling = False`、`cfg.use_undersampling = False`。
- 修改 `cfg.run_name` 以區分不同實驗；所有輸出會依 run name 分開儲存。
- 推論時欲調整閾值，可在 `run_inference_to_submission` 傳入 `score_thresh`、`iou_thresh`。

如有額外自訂流程，建議複製 `HW2_solution.ipynb` 並另存版本，避免覆寫原始設定。祝順利完成 HW2！🎯
