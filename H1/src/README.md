# Pig Detection with YOLOv11

此專案使用 Ultralytics YOLOv11 模型進行豬隻偵測，為 NTU CVPDL 作業的一部分。完整流程實作於 `pig_detection_training.ipynb`，涵蓋資料前處理、模型訓練以及產出提交檔案。

## 資料集結構

請將提供的資料放置在與筆記本同一層的專案根目錄：

```
train/
  img/
    00000001.jpg
    ...
  gt.txt
test/
  img/
    00000001.jpg
    ...
sample_submission.csv
pig_detection_training.ipynb
requirements.txt
```

## 環境安裝

開發時使用 Python 3.11.10（Conda 環境名稱為 `CV`）。

```bash
conda create -n CV python=3.11.10
conda activate CV
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

建議使用支援 CUDA 的 GPU 進行訓練，請自行確認驅動程式與 CUDA 版本相容。

## 執行 Notebook

利用 Jupyter Lab 或 Notebook 依序執行所有儲存格：

```bash
python -m pip install jupyter  # 若尚未安裝
jupyter lab pig_detection_training.ipynb
```

Notebook 主要流程如下：

1. 將 `train/gt.txt` 轉換為 YOLO 標註格式，並於 `yolo_dataset/` 建立所需檔案。
2. 以設定的超參數訓練 `yolo11m` 模型，訓練結果儲存在 `yolo_runs/yolo11x_pig_from_yaml2/`。
3. 將最佳權重保存至 `yolo_runs/yolo11x_pig_from_yaml2/weights/best.pt`。
4. 對 `test/img` 進行推論並產生 `submission_yolo.csv`。


