# MNIST RGB DDPM 報告

## 1. 模型描述

本專案實作 Denoising Diffusion Probabilistic Model (DDPM)，能生成 28×28 的 RGB MNIST 手寫數字。核心使用 UNet 去噪網路，每一步都預測加諧的雜訊，並由含 1,000 個線性 beta 係數（1e-4 → 0.02）的高斯 diffusion 流程包裝。殘差模組內嵌入正弦時間編碼，讓模型在不同噪聲層級都能調整對應特徵。

**架構示意**

```
輸入 (3×28×28)
  │ Conv 3→64
  │
  ├─ 下採樣 1：ResBlock(64→64) → ResBlock(64→64) → Downsample
  ├─ 下採樣 2：ResBlock(64→128) → ResBlock(128→128) → Downsample
  └─ 下採樣 3：ResBlock(128→256) → ResBlock(256→256)
        │
        ├─ 瓶頸：ResBlock(256→256) → ResBlock(256→256)
        ↓
  ┌─ 上採樣 3： concat skip + ResBlock(512→256) → ResBlock(256→256)
  ├─ 上採樣 2： concat skip + ResBlock(384→128) → ResBlock(128→128) → Upsample
  └─ 上採樣 1： concat skip + ResBlock(192→64)  → ResBlock(64→64)  → Upsample
  │
輸出頭：GroupNorm → SiLU → Conv 64→3（預測噪聲）
```

編碼器與解碼器對稱，跳接保留高解析細節，輸出像素級噪聲殘差供 diffusion 取樣器逐步去噪。

## 2. 實作細節

- **資料與前處理**：使用提供的 `minst/`（60,000 PNG）。影像轉為 RGB、縮放至 28×28，轉 tensor 後再正規化到 [-1, 1]；為保持字型結構，不額外做資料增強。
- **訓練設定**
  - 批次 1,024、訓練 50 epoch、Adam 學習率 2e-4、梯度裁剪 1.0。
  - Diffusion 步數 1,000，線性 beta；每個 iteration 隨機抽取時間步。
  - 256 維正弦時間編碼接兩層 MLP，輸入至所有殘差方塊。
  - 每 epoch 儲存 checkpoint；每 10 epoch 產生 64 張預覽樣本方便追蹤品質。
- **損失函數**：標準 DDPM 噪聲預測 MSE（`MSE(predicted_noise, ε)`），與生成品質高度相關。
- **生成與評估**
  - 完訓後以 batch 128 生成 10,000 張影像，編號存成 PNG 並壓縮。
  - 以 `python -m pytorch_fid generated_images mnist.npz` 對官方 RGB MNIST 統計計算 FID。
  - 額外腳本重播從純噪聲到成品的反向過程，提供報告所需的 diffusion 可視化。

## 3. 結果分析

### 訓練動態（Loss）

| Epoch | Loss (MSE) |
| --- | --- |
| 1 | 0.0858 |
| 5 | 0.0299 |
| 10 | 0.0189 |
| 20 | 0.0202 |
| 30 | 0.0133 |
| 40 | 0.0130 |
| 50 | 0.0109 |

Loss 前 10 個 epoch 快速下降，之後維持緩慢趨勢，表示模型已進入細節修飾階段而非大幅結構調整。

### 量化指標

| 指標 | 數值 | 備註 |
| --- | --- | --- |
| FID（10k 樣本） | **15.87** | 生成 10,000 張 RGB 數字後，使用 `pytorch_fid` 與 `mnist.npz` 比對。 |
| 訓練耗時 | 約 15 秒 / epoch | 每個 epoch 有 59 個 batch，在提供的 GPU 上約 3 it/s，可於 ~13 分鐘完成 50 epoch。 |

最終 FID 15.87 較早期 checkpoint（Loss >0.03）有明顯進步，證實完整訓練讓樣本雜訊與偽影顯著降低。

### Diffusion 過程視覺化（5%）

![Diffusion process grid](samples_preview/diffusion_process.png)

每欄代表一個樣本，每列依序展示從純高斯噪聲（最上方）到最終數字（底部）的去噪軌跡。時間步平均切為七段（每約 142 步記錄一次），方便觀察筆畫逐步成形並貼入書面報告。

## 4. 簡短結論

在 RGB MNIST 上訓練具線性噪聲排程的精簡 UNet-DDPM，可生成多樣且銳利的手寫數字。大型批次、梯度裁剪與定期預覽使訓練穩定，Loss 降至 0.0109、10k 樣本 FID 15.9，滿足作業需求並提供可重現的取樣流程與視覺化證據。後續可探索 cosine beta 或 classifier-free guidance 以進一步提升感知品質，但現有流程已完整展示 diffusion 模型的成效。
