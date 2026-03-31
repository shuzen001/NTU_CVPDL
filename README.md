# NTU_CVPDL

這個 repository 整理了我在 NTU CVP/DL 課程中的作業與期末專案實作，內容包含物件偵測、擴散模型，以及文字生成影像相關實驗。

## Repository Structure

### `H1/`
Homework 1，主題是豬隻偵測。

- 主要模型：YOLO 系列
- 主要程式：
  - `train_main.py`
  - `predict_main.py`
  - `pig_detection_training_extracted.py`
  - `src/pig_detection_training.ipynb`
- 補充說明：[`H1/src/README.md`](/home/tony/Downloads/NTU_CVPDL/H1/src/README.md)

### `H2/`
Homework 2，主題是長尾物件偵測。

- 主要模型：YOLOv12m / RT-DETR related experiments
- 主要內容：
  - `CVPDL_hw2/CVPDL_hw2/src/HW2_solution.ipynb`
  - `CVPDL_hw2/CVPDL_hw2/models/`
  - `CVPDL_hw2/CVPDL_hw2/report.*`
- 補充說明：[`H2/CVPDL_hw2/CVPDL_hw2/src/readme.md`](/home/tony/Downloads/NTU_CVPDL/H2/CVPDL_hw2/CVPDL_hw2/src/readme.md)

### `H3/`
Homework 3，主題是 RGB MNIST 的 DDPM 實作。

- 主要內容：
  - `mnist_ddpm_workflow.ipynb`
  - `mnist_ddpm_report.md`
  - `requirements.txt`
- 補充說明：[`H3/README.md`](/home/tony/Downloads/NTU_CVPDL/H3/README.md)

### `final_project/`
期末專案，包含以 Stable Diffusion 為基礎的文字生成影像實驗與介面程式。

- 主要程式：
  - `app.py`
  - `sdxl.py`
  - `sdxl_v2.py`

## Environment

不同作業使用的套件略有差異，建議依各子專案內的 `requirements.txt` 建立獨立環境。

例如：

```bash
cd H3
pip install -r requirements.txt
```

若要執行 Notebook，建議另外安裝 Jupyter：

```bash
pip install jupyterlab
```

## Notes

- 資料集、模型權重、訓練輸出與快取檔案沒有納入版控。
- 若要重現結果，請依各子專案 README 準備對應資料與執行環境。
- 部分大檔案與嵌入的第三方 repo 已透過 `.gitignore` 排除。

## License

此 repository 主要作為課程作業與研究紀錄使用。
