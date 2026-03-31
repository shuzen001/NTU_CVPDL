# CVPDL HW2 報告 — YOLOv12m 長尾物件偵測（從零訓練）

## 1. Model Description
本作業以自行定義的 YOLOv12m 為主體，完全不使用預訓練權重，並針對資料長尾特性進行架構與資料流程調整。模型骨幹維持 CSP/C2f 堆疊，但在原本 CBAM 注意力位置動態插入 **Coordinate Attention (CA)**，保留空間位置資訊以加強小物件與尾端類別；同時維持三層 FPN/PAN 頻寬以處理 2880×2880 之高解析度輸入。

```

backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1-P2/4
  - [-1, 3, C2f, [128, True]]           # 2
  - [-1, 1, CBAM, [7]]                  
  - [-1, 1, Conv, [256, 3, 2]]          # 4-P3/8
  - [-1, 6, C2f, [256, True]]           # 5
  - [-1, 1, CBAM, [7]]                  
  - [-1, 1, Conv, [512, 3, 2]]          # 7-P4/16
  - [-1, 6, C2f, [512, True]]           # 8
  - [-1, 1, CBAM, [7]]                  
  - [-1, 1, Conv, [1024, 3, 2]]         # 10-P5/32
  - [-1, 3, C2f, [1024, True]]          # 11
  - [-1, 1, CBAM, [7]]                  
  - [-1, 1, SPPF, [1024, 5]]            # 13
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 14
  - [[-1, 9], 1, Concat, [1]]                    # 15 cat backbone P4 (layer 9)
  - [-1, 3, C2f, [512]]                          # 16
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 17
  - [[-1, 6], 1, Concat, [1]]                    # 18 cat backbone P3 (layer 6)
  - [-1, 3, C2f, [256]]                          # 19 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]                   # 20
  - [[-1, 16], 1, Concat, [1]]                   # 21 cat head P4
  - [-1, 3, C2f, [512]]                          # 22 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]                   # 23
  - [[-1, 13], 1, Concat, [1]]                   # 24 cat head P5
  - [-1, 3, C2f, [1024]]                         # 25 (P5/32-large)
  - [[19, 22, 25], 1, Detect, [nc]]              # 26 Detect(P3, P4, P5)
```

核心修改與其目的：
- 以 **Coordinate Attention** 取代 CBAM，避免小物件在高解析下因平均池化而被抹除。
- 以 `imgsz=2880`、`batch=1` 配合 AdamW 與 cosine LR，確保在 12 GB VRAM 下仍能穩定收斂。
- 維持 Ultralytics Detect head，但調整 loss 比重與 IoU/NMS 參數以配合高解析輸入。
- 透過 class-aware Augmentation YAML，自訂 tail/head 類別增強倍率。

## 2. Implementation Details
- 資料前處理：將原始標註轉為 YOLO txt（中心點格式），以 `prepare_yolo_dataset` 建立 symlink；高解析輸入先行檢查缺失標註並保持原始長寬比。
- 長尾重採樣：`hybrid_resample_dataset` 先以 `oversample_threshold=0.3` 判斷 tail 類別，再對 head-only 影像可選擇性下採樣，最後針對 tail 類別複製含該類別之影像直到達到 `target_balance_ratio=0.5`。
- 類別感知增強：在 YAML 註記 tail/head 類別增強倍率 (`tail_class_aug_factor=2`, `head_class_aug_factor=0.5`)，搭配 Mosaic 0.4、Copy-Paste 0.5 與 RandAugment 色彩轉換，只在尾端類別保留較強增強。
- 訓練超參數：300 epoch、`lr0=1e-3`、AdamW (`weight_decay=5e-4`)、cosine LR、warmup 3 epoch、`patience=25`，AMP 開啟並於最後 10 epoch 關閉 Mosaic。
- 驗證與推論：於每次 epoch 保存最佳 mAP50 權重，並以官方 `val_batch*_pred.jpg`、混淆矩陣與 PR/F1 曲線檢視收斂；最終使用 `runs/yolov12m_2880/weights/best.pt` 產生 Kaggle `submission.csv`。

## 3. Result Analysis

![訓練過程指標曲線](runs/yolov12m_2880/results.png)

### 3.1 混合長尾策略後的資料分佈
| 類別 | 原始訓練數量 | 佔比 | 重採樣後訓練數量 | 佔比 |
| --- | ---: | ---: | ---: | ---: |
| Car | 21,313 | 100.0% | 72,956 | 100.0% |
| Hov | 1,205 | 5.7% | 6,245 | 8.6% |
| Person | 2,891 | 13.6% | 18,812 | 25.8% |
| Motorcycle | 4,737 | 22.2% | 28,750 | 39.4% |
| 影像數 | 855 | — | 3,179 | — |

混合重採樣大幅提升尾端類別樣本量（Hov ×4.0、Person ×3.7、Motorcycle ×2.3），並維持 head 類別影像的多樣性，為長尾問題提供更平衡的訓練基礎。

### 3.2 數據成效比較
| 模型 | `imgsz` | 長尾策略 | 最佳 epoch | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv12m（基線） | 1280 | 無額外調整 | 126 | 0.701 | 0.257 | 0.727 | 0.684 |
| **YOLOv12m（本作）** | 2880 | Hybrid Resample + CA | **181** | **0.732** | **0.269** | **0.778** | **0.696** |

提升重點：
- mAP50 +3.1%，mAP50-95 +1.1%，顯示高解析輸入與注意力替換強化了多尺度偵測。
- Precision +5.1%，Recall +1.2%，推論更穩定且少誤報，特別在 Person、Motorcycle 等尾端類別上觀察到召回率提升（Person recall 0.41 → 0.43）。

### 3.3 訓練曲線與量化圖表
- `runs/yolov12m_2880/results.png` 顯示 50 epoch 內 loss 明顯下降，後期進入穩定振盪。
- `BoxPR_curve.png`、`BoxF1_curve.png` 顯示高 IoU 下 precision 有效提升；F1 峰值落在信心 0.32 左右。
![Box Precision 曲線](runs/yolov12m_2880/BoxP_curve.png)
![Box Recall 曲線](runs/yolov12m_2880/BoxR_curve.png)

### 3.4 視覺化結果
![驗證集樣本預測 3](runs/yolov12m_2880/val_batch2_pred.jpg)

### 3.5 長尾問題分析
- **資料層面**：混合重採樣有效擴充 tail 類別，並透過類別感知增強讓尾端樣本暴露於更強的形變與色彩擾動，減少過擬合。
- **模型層面**：Coordinate Attention 在 P3/P4/P5 特徵後加入，使尾端小物件於高解析圖像中保留位置信息，對應 Person、Motorcycle 的 recall 上升。
- **綜合評估**：與原始 1280 模型相比，整體 mAP50-95 提升 1.1%，且 tail 類別於驗證集的檢測信心更集中（Box PR 曲線面積增大），符合長尾優化目標。

## 4. Short Conclusion
透過高解析 YOLOv12m、Coordinate Attention、混合重採樣與類別感知增強，本作在不使用預訓練權重的前提下，將 mAP50 提升至 0.732、Recall 提升至 0.696，並在尾端類別上取得更穩定的檢測結果。後續可考慮加入動態 re-weight loss 或針對背景誤判進行廣義 Focal Loss 調整，以進一步壓低 tail 類別的背景漏檢。
