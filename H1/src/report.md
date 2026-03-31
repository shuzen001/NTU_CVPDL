# 豬隻偵測模型報告

## 1. 模型介紹

本專案以 Ultralytics YOLOv11m 為核心偵測器。YOLOv11 採用 CBS 模塊堆疊的骨幹網路結合 PAN-FPN 頸部結構與 C2f Head，能同時在多尺度特徵圖上預測邊界框與類別。針對作業要求，不載入預訓練權重，搭配自訂增強策略從零開始訓練，以避免既有資料偏移的影響。

```
輸入影像 (1024×1024)
        │
        ▼
  Backbone：C2f + SPPF 堆疊
        │        ┌───────────┐
        ├──────▶│ P3/8 特徵 │
        │        └───────────┘
        │        ┌───────────┐
        ├──────▶│ P4/16 特徵│
        │        └───────────┘
        │        ┌───────────┐
        └──────▶│ P5/32 特徵│
                 └───────────┘
                         │
                         ▼
              PAN-FPN 頸部融合 (C3k2 交錯)
                         │
                         ▼
              檢測頭 (C2PSA + Detect)
                         │
                         ▼
               邊界框與置信度輸出
```

主要調整如下：
- 以 `yolo11m.yaml` 為基礎，但關閉 `pretrained`預訓練權重，改為隨機初始化以配合高度客製化的資料分佈。
- 啟用 `dropout=0.3`、`mixup=0.2` 與 `label_smoothing=0.03`，提高對遮擋與標註不確定性的魯棒性。
- 使用 `close_mosaic=10` 在訓練後期關閉馬賽克增強，減少過度增強導致的樣本偏差。
- 調整 `conf` 與 `iou` 推論閾值（0.001 與 0.68），以保留更多候選框再由後處理篩選。

## 2. 實作細節

**資料前處理**
- 將 `train/gt.txt` 讀入為 DataFrame，依影像編號整理成 `frame → bbox` 對應表。
- 以 `torch.randperm` 隨機打亂影像後，採 80%：20% 劃分訓練與驗證集，並固定隨機種子 (`SEED=13324837`) 確保可重現性。
- 將原始 VOC 風格標註轉換為 YOLO `(x_center, y_center, width, height)` 正規化格式，篩除無效或超出範圍的框。
- 透過 `symlink`建立 `yolo_dataset/images/{train,val,test}` 與 `labels/` 目錄結構，並輸出 `pig_detection.yaml` 描述檔。

**資料增強與訓練策略**
- 影像尺寸固定為 1024，批次大小 10，使用自動混合精度 (`amp=True`) 與 `cache=True` 加速。
- 色彩與幾何增強設定：
  - `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`
  - `translate=0.08`, `scale=0.6`, `fliplr=0.5`
  - `mosaic=1.0`、`mixup=0.2`、`close_mosaic=10`
- 優化器採 Ultralytics 預設（SGD 變體），關鍵超參數如下：

| 超參數 | 值 |
| --- | --- |
| `epochs` | 50 |
| `batch` | 10 |
| `imgsz` | 1024 |
| `lr0` | 0.005 |
| `momentum` | 0.937 |
| `weight_decay` | 3e-3 |
| `warmup_epochs` | 8.0 |
| `patience` | 35 |
| `cos_lr` | True |
| `dropout` | 0.3 |
| `mixup` | 0.2 |
| `label_smoothing` | 0.03 |

**損失函數與最佳化**
- 沿用 YOLOv11 內建 TaskAligned Assign 練與 DFL (Distribution Focal Loss) + CIoU 邊界框回歸，以及 BCE 類別與物件性損失，無額外修改。
- 透過 `cos_lr=True` 進行餘弦退火學習率排程，並在驗證指標未提升時透過 `patience` 控制提前停止。

**推論設定**
- 驗證與生成提交檔案時統一使用 `conf=0.001`、`iou=0.68`、`augment=True`，搭配 TTA 保留多尺度偵測。
- 推論函式 `create_submission` 將 YOLO 預測轉為競賽所需的 `PredictionString` 格式並輸出至 `submission_yolo.csv`。

## 3. 結果分析

**量化指標**

| 指標 | 數值 (驗證集) |
| --- | --- |
| Precision | 0.9839 |
| Recall | 0.9754 |
| mAP@0.5 | 0.9923 |
| mAP@0.5:0.95 | 0.8646 |
| mAP@0.75 | 0.9330 |

- 與訓練初期隨機初始化狀態（mAP 幾近 0）相比，最終模型在 mAP@0.5:0.95 上獲得 0.8646 的絕對提升，顯示資料增強與學習率排程有效提升收斂速度與表現。
- 高 Precision 與 Recall 代表模型在正確檢測與避免誤報之間取得良好平衡，適合用於提交。


- 透過 notebook 中的 `analyze_submission_results(submission_df)`，可進一步檢視預測框數量與置信度分布，協助定位極端樣本或低信心檢測；若在部署前觀察到低信心框比例升高，可再調整 `conf` 閾值或增補資料。

## 4. 結果發現

### 初期

一開始把信心度、NMS閾值設得比較高，結果發現模型偵測到的框數量偏少，但是卻很精準，這代表模型學到的東西是對的，然而繳交預測結果的分數卻不高。後來把信心度閾值調低，結果發現模型偵測到的框數量變多了，雖然有些框的信心度比較低，但是整體來說繳交預測結果的分數變高了。經過查找相關的評估方法發現，這可能是由於本次作業的評估標準是mAP@0.5:0.95，這個指標會考慮到不同IoU閾值下的平均精度，因此在某些IoU閾值下，較低信心度的框可能會被計算進去，從而提高整體的mAP分數。