# MNIST RGB DDPM

此專案依照 NTU CVP/DL 作業要求，使用 Denoising Diffusion Probabilistic Model（DDPM）從頭實作 28×28 RGB MNIST 的訓練、取樣與評估流程。主要工作流程實現在 `mnist_ddpm_workflow.ipynb`

## 功能總覽

- 讀取 `minst/` 中的 60,000 張 RGB MNIST PNG，正規化到 [-1, 1]。
- 實作含殘差/跳接的 UNet 去噪器與 Gaussian Diffusion 模組（1,000 線性 beta 步驟）。
- 以 batch size 1,024、50 epochs 訓練，並支援 checkpoint 儲存、生成 10,000 張影像、壓縮及 FID 計算。
- 產生 diffusion process 視覺化（純噪聲 → 成品八階段）供報告使用。

## 目錄結構

```
H3/
├─ mnist_ddpm_workflow.ipynb   # 主要 Notebook，含完整流程與圖表
├─ checkpoints/                # 訓練權重（model_epoch_X.pt, model_final.pt）
├─ samples_preview/            # 每 10 epochs 預覽、diffusion_process.png
├─ generated_images/           # 10,000 張最終 PNG
├─ mnist.npz                   # 官方提供的 mean/cov 統計
└─ minst/                      # MNIST RGB PNG 資料夾（需自行放置）
```

## 環境與安裝

```bash
pip install -r requirements.txt
```

主要依賴：

- PyTorch / TorchVision
- tqdm
- Pillow
- matplotlib
- pytorch-fid（計算 FID）

> **注意**：若要啟用 GPU 訓練，請先安裝對應 CUDA 版本的 PyTorch。

## 資料準備

1. 將提供的 RGB MNIST PNG 解壓縮到 `minst/`。
2. 將官方統計（`mnist.npz`，內含 mean/cov）放在專案根目錄。

Notebook 與腳本會依照作業指定的資料夾結構自動讀取上述路徑。

## Notebook 使用流程

1. 依序執行 `mnist_ddpm_workflow.ipynb` 內的 cell。
2. 可調整 `config` 參數（影像尺寸、batch size、timesteps 等）。
3. `RUN_TRAINING = True` 時會直接開始訓練；完成後可設定 `RUN_GENERATION = True` 產生 10,000 張影像。
4. Notebook 末端包含：
   - 影像壓縮（zip）
   - `python -m pytorch_fid` 指令計算 FID
   - `visualize_diffusion_process()`：從純噪聲開始，生成 8×8 圖格，可直接截圖貼到報告。

## Diffusion 視覺化

執行 Notebook 最後一個 cell 會在 `samples_preview/diffusion_process.png` 生成 8（樣本）×8（時間段）的圖格。若需於 CLI 產生，可在環境中載入 `mnist_ddpm_workflow.ipynb` 末段函式或撰寫小腳本載入 checkpoint 後呼叫 `visualize_diffusion_process()`。