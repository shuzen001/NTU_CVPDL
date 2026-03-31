# %%# 環境設置
from pathlib import Path
import random
import shutil
import yaml

import numpy as np
import pandas as pd
from PIL import Image

import torch

try:
    from ultralytics import YOLO
    from ultralytics.utils import ROOT as ULTRA_ROOT
except ImportError as exc:
    raise ImportError(
        "需要Ultralytics套件。請使用 `pip install -U ultralytics` 安裝以使用YOLOv11模型。"
    ) from exc

print(f"Torch版本: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# %%# 針對RTX 4090和640x360圖片調整的路徑和超參數
DATA_ROOT = Path(".").resolve()
TRAIN_IMG_DIR = DATA_ROOT / "train" / "img"
TRAIN_GT_PATH = DATA_ROOT / "train" / "gt.txt"
TEST_IMG_DIR = DATA_ROOT / "test" / "img"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"
YOLO_OUTPUT_DIR = DATA_ROOT / "yolo_runs"
YOLO_DATA_ROOT = DATA_ROOT / "yolo_dataset"
SUBMISSION_PATH = DATA_ROOT / "submission_yolo.csv"

CFG_NAME = "yolo11.yaml"
MODEL_CFG_ROOT = ULTRA_ROOT / "cfg" / "models"

candidates = [p for p in MODEL_CFG_ROOT.rglob(CFG_NAME) if p.is_file()]
if not candidates:
    local_candidate = DATA_ROOT / CFG_NAME
    if local_candidate.exists():
        candidates.append(local_candidate.resolve())

if not candidates:
    raise FileNotFoundError(
        f"在{MODEL_CFG_ROOT}或目前工作區內找不到所需的架構檔案{CFG_NAME}。"
        "請確認`pip install -U ultralytics`已成功完成，或將YAML檔案放置在此筆記本旁邊。"
    )

YOLO_CFG_PATH = candidates[0].resolve()
print(f"使用YOLO配置: {YOLO_CFG_PATH}")

VAL_SPLIT = 0.2
SEED = 13324837

YOLO_EPOCHS = 50  # 從頭開始收斂需要更長的訓練週期
YOLO_BATCH = 10 # 更高的批次大小適合原生解析度
YOLO_IMAGE_SIZE = 1024  # 匹配原生寬度/高度倍數 (640x360)
YOLO_LR0 = 0.005
YOLO_WEIGHT_DECAY = 3e-3
YOLO_MOMENTUM = 0.937
YOLO_WARMUP_EPOCHS = 8.0
YOLO_PATIENCE = 35

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"訓練圖片: {len(list(TRAIN_IMG_DIR.glob('*.jpg')))}")
print(f"測試圖片: {len(list(TEST_IMG_DIR.glob('*.jpg')))}")
print(f"使用配置{YOLO_CFG_PATH}進行訓練，圖片大小{YOLO_IMAGE_SIZE}，訓練週期{YOLO_EPOCHS}，批次大小{YOLO_BATCH}")
# %%# 將真實標註轉換為YOLO格式並建立資料集配置
YOLO_IMAGES_DIR = YOLO_DATA_ROOT / "images"
YOLO_LABELS_DIR = YOLO_DATA_ROOT / "labels"
for split in ["train", "val", "test"]:
    (YOLO_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (YOLO_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

annotations_df = pd.read_csv(TRAIN_GT_PATH, header=None, names=["frame", "x", "y", "w", "h"])
annotations_map = {}
for _, row in annotations_df.iterrows():
    annotations_map.setdefault(int(row["frame"]), []).append((float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])) )

all_images = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
indices = torch.randperm(len(all_images)).tolist()
val_count = max(1, int(len(all_images) * VAL_SPLIT))

split_assignments = {}
for position, original_idx in enumerate(indices):
    img_path = all_images[original_idx]
    # 根據在打亂列表中的位置決定split
    split_assignments[img_path] = "val" if position < val_count else "train"
      
for img_path in sorted(TEST_IMG_DIR.glob("*.jpg")):
    split_assignments[img_path] = "test"

invalid_box_counter = 0
for img_path, split in split_assignments.items():
    target_img_dir = YOLO_IMAGES_DIR / split
    target_lbl_dir = YOLO_LABELS_DIR / split

    target_img_path = target_img_dir / img_path.name
    if not target_img_path.exists():
        try:
            target_img_path.symlink_to(img_path)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(img_path, target_img_path)

    if split in {"train", "val"}:
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        boxes = annotations_map.get(int(img_path.stem), [])

        yolo_lines = []
        for x, y, w, h in boxes:
            if w <= 0 or h <= 0:
                invalid_box_counter += 1
                continue
            x_center = (x + w / 2.0) / width
            y_center = (y + h / 2.0) / height
            w_norm = w / width
            h_norm = h / height

            if not (0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0):
                invalid_box_counter += 1
                continue

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 1e-6), 1.0)
            h_norm = min(max(h_norm, 1e-6), 1.0)

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = target_lbl_dir / f"{img_path.stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
    else:
        # 為測試圖片建立空的標籤檔案（可選但保持結構一致）
        label_path = target_lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            label_path.touch()

if invalid_box_counter:
    print(f"轉換過程中跳過了{invalid_box_counter}個無效的邊界框")

train_count = len(list((YOLO_IMAGES_DIR / "train").glob("*.jpg")))
val_count = len(list((YOLO_IMAGES_DIR / "val").glob("*.jpg")))
test_count = len(list((YOLO_IMAGES_DIR / "test").glob("*.jpg")))
print(f"YOLO資料集 -> 訓練:{train_count} | 驗證:{val_count} | 測試:{test_count}")

yolo_data_yaml = YOLO_DATA_ROOT / "pig_detection.yaml"
yolo_data_dict = {
    "path": str(YOLO_DATA_ROOT.resolve()),
    "train": str((YOLO_IMAGES_DIR / "train").resolve()),
    "val": str((YOLO_IMAGES_DIR / "val").resolve()),
    "test": str((YOLO_IMAGES_DIR / "test").resolve()),
    "nc": 1,
    "names": ["pig"],
}
with yolo_data_yaml.open("w", encoding="utf-8") as f:
    yaml.safe_dump(yolo_data_dict, f)

print(f"已寫入資料集配置 -> {yolo_data_yaml}")
# %%# 從YAML定義訓練YOLO（隨機初始化）
yolo_device = 0 if torch.cuda.is_available() else "cpu"
print(f"在設備{yolo_device}上開始訓練")

model = YOLO('yolo11m.yaml')
train_results = model.train(
    data=str(yolo_data_yaml),
    epochs=YOLO_EPOCHS,
    imgsz=YOLO_IMAGE_SIZE,
    batch=YOLO_BATCH,
    device=yolo_device,
    pretrained=False,
    lr0=YOLO_LR0,
    momentum=YOLO_MOMENTUM,
    weight_decay=YOLO_WEIGHT_DECAY,
    warmup_epochs=YOLO_WARMUP_EPOCHS,
    patience=YOLO_PATIENCE,
    cache=True,
    workers=8,
    cos_lr=True,
    amp=True,
    # rect=True,
    # multi_scale=True,
    close_mosaic=10,
    dropout=0.3,
    mixup=0.2,
    label_smoothing=0.03,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.08,
    scale=0.6,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    save_period=50,
    project=str(YOLO_OUTPUT_DIR),
    name="yolo11x_pig_from_yaml2",
    exist_ok=True,
)

trainer = train_results if hasattr(train_results, "save_dir") else getattr(model, "trainer", None)
if trainer is None or not hasattr(trainer, "save_dir"):
    raise RuntimeError("無法找到YOLO訓練器的save_dir。請檢查ultralytics版本和訓練運行。")

best_weights = Path(trainer.save_dir) / "weights" / "best.pt"
last_weights = Path(trainer.save_dir) / "weights" / "last.pt"
if best_weights.exists():
    print(f"載入最佳檢查點: {best_weights}")
    best_model = YOLO(best_weights)
elif last_weights.exists():
    print("未找到最佳權重；回退到最後一個週期的權重。")
    best_model = YOLO(last_weights)
else:
    print("未找到已儲存的權重；使用記憶體中的模型實例。")
    best_model = model
# %%# 在驗證集上評估以獲得快速回饋
val_metrics = best_model.val(
    data=str(yolo_data_yaml),
    imgsz=YOLO_IMAGE_SIZE,
    batch=YOLO_BATCH,
    device=yolo_device,
    split="val",
    save_json=False,
    verbose=True,
)
print(val_metrics)
# %%
# 明確打印不同IoU閾值的mAP
print("=== 詳細mAP指標 ===")
print(f"mAP50: {val_metrics.box.map50:.4f}")      # IoU=0.5時的mAP
print(f"mAP50-95: {val_metrics.box.map:.4f}")     # IoU=0.5-0.95的平均mAP
print(f"mAP75: {val_metrics.box.map75:.4f}")      # IoU=0.75時的mAP

# 如果有各類別的mAP
if hasattr(val_metrics.box, 'maps'):
    print(f"Per-class mAP50-95: {val_metrics.box.maps}")
# %%# 在驗證集上掃描置信度/IoU閾值以選擇最佳提交操作點
import numpy as np

conf_grid = np.linspace(0.10, 0.40, 7)
iou_grid = [0.4, 0.5, 0.55, 0.6, 0.65]  # 為NMS調優調整IoU閾值
records = []
best_model = model
yolo_device = 0 if torch.cuda.is_available() else "cpu"

for conf in conf_grid:
    for iou in iou_grid:
        metrics = best_model.val(
            data=str(yolo_data_yaml),
            imgsz=YOLO_IMAGE_SIZE,
            batch=YOLO_BATCH,
            device=yolo_device,
            split="val",
            conf=conf,
            iou=iou,
            verbose=False,
        )
        records.append({
            "conf": conf,
            "iou": iou,
            "map50": metrics.box.map50,
            "map": metrics.box.map,
        })

import pandas as pd
threshold_df = pd.DataFrame(records).sort_values(by=["map", "map50"], ascending=False)
threshold_df.reset_index(drop=True, inplace=True)
threshold_df.head()

BEST_CONF_THRESHOLD = float(threshold_df.iloc[0]["conf"]) if not threshold_df.empty else 0.25
BEST_IOU_THRESHOLD = float(threshold_df.iloc[0]["iou"]) if not threshold_df.empty else 0.6
print(f"選定的閾值 -> 置信度: {BEST_CONF_THRESHOLD:.3f}, IoU: {BEST_IOU_THRESHOLD:.2f}")
# %%BEST_CONF_THRESHOLD = 0.001  # 根據驗證集結果微調
BEST_IOU_THRESHOLD = 0.68
# %%from tqdm import tqdm

# 使用訓練好的YOLO模型和輕量TTA生成提交CSV
@torch.no_grad()
def create_submission(model, sample_path, test_dir, output_path, conf_threshold=BEST_CONF_THRESHOLD, iou_threshold=BEST_IOU_THRESHOLD):
    sample_df = pd.read_csv(sample_path)
    predictions = []

    # 加入tqdm進度條
    for image_id in tqdm(sample_df["Image_ID"], desc="生成預測結果", unit="張"):
        img_path = Path(test_dir) / f"{int(image_id):08d}.jpg"
        if not img_path.exists():
            predictions.append("")
            continue

        results = model.predict(
            source=str(img_path),
            imgsz=YOLO_IMAGE_SIZE,
            conf=conf_threshold,
            iou=iou_threshold,
            device=yolo_device,
            augment=True,
            verbose=False,
        )

        if not results:
            predictions.append("")
            continue

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or boxes.xyxy.numel() == 0:
            predictions.append("")
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        entries = []
        for score, box, cls in zip(scores, xyxy, classes):
            x_min, y_min, x_max, y_max = box
            width = max(0.0, x_max - x_min)
            height = max(0.0, y_max - y_min)
            entries.append(f"{score:.4f} {x_min:.1f} {y_min:.1f} {width:.1f} {height:.1f} {int(cls)}")

        predictions.append(" ".join(entries))

    submission_df = sample_df.copy()
    submission_df["PredictionString"] = predictions
    submission_df.to_csv(output_path, index=False)
    return submission_df

submission_df = create_submission(best_model, SAMPLE_SUB_PATH, TEST_IMG_DIR, SUBMISSION_PATH)
# submission_df = create_submission(best_model, SAMPLE_SUB_PATH, TEST_IMG_DIR, SUBMISSION_PATH, conf_threshold=0.15, iou_threshold=0.65)

print(f"提交檔案已儲存至 {SUBMISSION_PATH}")
submission_df.head()
# %%# 使用其他epoch儲存的權重進行推論
from pathlib import Path
from ultralytics import YOLO

available_weight_paths = sorted(Path("yolo_runs/yolo11x_pig_from_yaml2/weights").glob("**/*.pt"))
if not available_weight_paths:
    raise FileNotFoundError("在 runs/ 內找不到任何 .pt 權重檔案，請確認已將權重檔案保存於此。")

print("可用權重檔案：")
for idx, path in enumerate(available_weight_paths):
    print(f"[{idx}] {path}")



# %%

yolo_device = 0 if torch.cuda.is_available() else "cpu"
selected_index = 0  # 修改為想要使用的權重索引
if selected_index < 0 or selected_index >= len(available_weight_paths):
    raise IndexError(f"selected_index 必須介於 0 和 {len(available_weight_paths) - 1} 之間")
custom_weight_path = available_weight_paths[selected_index]

custom_model = YOLO(str(custom_weight_path))
custom_submission_path = SUBMISSION_PATH.with_name(f"{custom_weight_path.stem}_submission.csv")
custom_submission_df = create_submission(
    custom_model,
    SAMPLE_SUB_PATH,
    TEST_IMG_DIR,
    custom_submission_path,
    conf_threshold=BEST_CONF_THRESHOLD,
    iou_threshold=BEST_IOU_THRESHOLD,
)
print(f"使用 {custom_weight_path} 的預測已儲存至 {custom_submission_path}")
custom_submission_df.head()
# %%# 詳細分析提交結果的統計資訊
def analyze_submission_results(submission_df):
    """分析提交結果的詳細統計資訊"""
    print("=" * 60)
    print("📊 提交結果統計分析")
    print("=" * 60)
    
    total_images = len(submission_df)
    
    # 統計每張圖片的檢測資訊
    detection_stats = []
    bbox_counts = []
    confidence_scores = []
    bbox_sizes = []
    
    for idx, row in submission_df.iterrows():
        pred_string = row['PredictionString']
        if pd.isna(pred_string) or pred_string == '':
            bbox_counts.append(0)
            detection_stats.append({
                'image_id': row['Image_ID'],
                'bbox_count': 0,
                'max_conf': 0,
                'avg_conf': 0,
                'avg_bbox_area': 0
            })
        else:
            # 解析預測字串：score x_min y_min width height class_id
            predictions = pred_string.strip().split(' ')
            num_predictions = len(predictions) // 6  # 每6個元素組成一個預測
            bbox_counts.append(num_predictions)
            
            image_confidences = []
            image_areas = []
            
            for i in range(num_predictions):
                try:
                    score = float(predictions[i * 6])
                    width = float(predictions[i * 6 + 3])
                    height = float(predictions[i * 6 + 4])
                    
                    confidence_scores.append(score)
                    image_confidences.append(score)
                    
                    area = width * height
                    bbox_sizes.append(area)
                    image_areas.append(area)
                except (ValueError, IndexError):
                    continue
            
            detection_stats.append({
                'image_id': row['Image_ID'],
                'bbox_count': num_predictions,
                'max_conf': max(image_confidences) if image_confidences else 0,
                'avg_conf': np.mean(image_confidences) if image_confidences else 0,
                'avg_bbox_area': np.mean(image_areas) if image_areas else 0
            })
    
    # 基本統計
    images_with_detections = sum(1 for count in bbox_counts if count > 0)
    images_without_detections = total_images - images_with_detections
    total_bboxes = sum(bbox_counts)
    
    print(f"🎯 檢測概覽：")
    print(f"   總圖片數量: {total_images:,}")
    print(f"   有檢測結果: {images_with_detections:,} ({images_with_detections/total_images*100:.1f}%)")
    print(f"   無檢測結果: {images_without_detections:,} ({images_without_detections/total_images*100:.1f}%)")
    print(f"   總檢測框數: {total_bboxes:,}")
    
    # Bbox數量統計
    print(f"\n📦 邊界框統計：")
    print(f"   平均每張圖片bbox數: {np.mean(bbox_counts):.2f}")
    print(f"   bbox數量中位數: {np.median(bbox_counts):.1f}")
    print(f"   最大bbox數(單張圖): {max(bbox_counts) if bbox_counts else 0}")
    print(f"   bbox數量標準差: {np.std(bbox_counts):.2f}")
    
    # Bbox數量分布
    if bbox_counts:
        bbox_distribution = {}
        for count in bbox_counts:
            bbox_distribution[count] = bbox_distribution.get(count, 0) + 1
        
        print(f"\n📈 Bbox數量分布（前10名）：")
        sorted_dist = sorted(bbox_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        for bbox_num, freq in sorted_dist:
            percentage = freq / total_images * 100
            print(f"   {bbox_num:2d}個bbox: {freq:4d}張圖片 ({percentage:5.1f}%)")
    
    # 置信度統計
    if confidence_scores:
        print(f"\n🎯 置信度統計：")
        print(f"   平均置信度: {np.mean(confidence_scores):.4f}")
        print(f"   置信度中位數: {np.median(confidence_scores):.4f}")
        print(f"   最高置信度: {max(confidence_scores):.4f}")
        print(f"   最低置信度: {min(confidence_scores):.4f}")
        print(f"   置信度標準差: {np.std(confidence_scores):.4f}")
        
        # 置信度區間分布
        conf_ranges = {
            '0.9-1.0': sum(1 for c in confidence_scores if 0.9 <= c <= 1.0),
            '0.7-0.9': sum(1 for c in confidence_scores if 0.7 <= c < 0.9),
            '0.5-0.7': sum(1 for c in confidence_scores if 0.5 <= c < 0.7),
            '0.3-0.5': sum(1 for c in confidence_scores if 0.3 <= c < 0.5),
            '0.1-0.3': sum(1 for c in confidence_scores if 0.1 <= c < 0.3),
            '0.0-0.1': sum(1 for c in confidence_scores if 0.0 <= c < 0.1),
        }
        
        print(f"\n📊 置信度區間分布：")
        for range_name, count in conf_ranges.items():
            percentage = count / len(confidence_scores) * 100
            print(f"   {range_name}: {count:4d} ({percentage:5.1f}%)")
    
    # 邊界框大小統計
    if bbox_sizes:
        print(f"\n📏 邊界框尺寸統計：")
        print(f"   平均面積: {np.mean(bbox_sizes):.1f} 像素²")
        print(f"   面積中位數: {np.median(bbox_sizes):.1f} 像素²")
        print(f"   最大面積: {max(bbox_sizes):.1f} 像素²")
        print(f"   最小面積: {min(bbox_sizes):.1f} 像素²")
        
        # 尺寸分布
        small_boxes = sum(1 for size in bbox_sizes if size < 1000)
        medium_boxes = sum(1 for size in bbox_sizes if 1000 <= size < 10000)
        large_boxes = sum(1 for size in bbox_sizes if size >= 10000)
        
        print(f"\n📐 尺寸分布：")
        print(f"   小框 (<1000px²): {small_boxes} ({small_boxes/len(bbox_sizes)*100:.1f}%)")
        print(f"   中框 (1000-10000px²): {medium_boxes} ({medium_boxes/len(bbox_sizes)*100:.1f}%)")
        print(f"   大框 (≥10000px²): {large_boxes} ({large_boxes/len(bbox_sizes)*100:.1f}%)")
    
    # 高檢測數量的圖片
    high_detection_images = [(stats['image_id'], stats['bbox_count']) 
                           for stats in detection_stats if stats['bbox_count'] > 10]
    if high_detection_images:
        print(f"\n🔥 高檢測數量圖片 (>10個bbox)：")
        high_detection_images.sort(key=lambda x: x[1], reverse=True)
        for img_id, count in high_detection_images[:5]:  # 顯示前5張
            print(f"   圖片 {img_id:08d}: {count} 個bbox")
    
    # 高置信度檢測
    if confidence_scores:
        high_conf_detections = sum(1 for c in confidence_scores if c > 0.8)
        print(f"\n⭐ 高置信度檢測 (>0.8): {high_conf_detections} 個 ({high_conf_detections/len(confidence_scores)*100:.1f}%)")
    
    print("=" * 60)
    
    return {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'total_bboxes': total_bboxes,
        'avg_bboxes_per_image': np.mean(bbox_counts),
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'bbox_counts': bbox_counts,
        'confidence_scores': confidence_scores,
        'detection_stats': detection_stats
    }
    
analyze_submission_results(submission_df)

# 在你的預測完成後調用這個函數
# print(f"使用 {custom_weight_path} 的預測已儲存至 {custom_submission_path}")

# 分析結果
# stats = analyze_submission_results(submission_df)

# 顯示DataFrame前幾行
# custom_submission_df.head()
