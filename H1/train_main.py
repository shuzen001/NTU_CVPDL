from pathlib import Path

import torch

from src import config, data_prep, train_pipeline


def main() -> None:
    print(f"Torch版本: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    yolo_cfg_path = config.resolve_yolo_cfg_path()
    print(f"使用YOLO配置: {yolo_cfg_path}")

    config.set_global_seed(config.SEED)

    print(f"訓練圖片: {len(list(config.TRAIN_IMG_DIR.glob('*.jpg')))}")
    print(f"測試圖片: {len(list(config.TEST_IMG_DIR.glob('*.jpg')))}")
    print(
        f"使用配置{yolo_cfg_path}進行訓練，圖片大小{config.YOLO_IMAGE_SIZE}，"
        f"訓練週期{config.YOLO_EPOCHS}，批次大小{config.YOLO_BATCH}"
    )

    yolo_data_yaml = data_prep.prepare_yolo_dataset(
        train_img_dir=config.TRAIN_IMG_DIR,
        train_gt_path=config.TRAIN_GT_PATH,
        test_img_dir=config.TEST_IMG_DIR,
        yolo_data_root=config.YOLO_DATA_ROOT,
        val_split=config.VAL_SPLIT,
    )

    model, best_model, save_dir = train_pipeline.train_model(
        data_yaml=yolo_data_yaml,
        epochs=config.YOLO_EPOCHS,
        image_size=config.YOLO_IMAGE_SIZE,
        batch_size=config.YOLO_BATCH,
        device=config.default_device(),
        lr0=config.YOLO_LR0,
        momentum=config.YOLO_MOMENTUM,
        weight_decay=config.YOLO_WEIGHT_DECAY,
        warmup_epochs=config.YOLO_WARMUP_EPOCHS,
        patience=config.YOLO_PATIENCE,
        output_dir=config.YOLO_OUTPUT_DIR,
    )

    val_metrics = train_pipeline.evaluate_model(
        model=best_model,
        data_yaml=yolo_data_yaml,
        image_size=config.YOLO_IMAGE_SIZE,
        batch_size=config.YOLO_BATCH,
        device=config.default_device(),
        verbose=True,
    )
    print(val_metrics)
    print("=== 詳細mAP指標 ===")
    print(f"mAP50: {val_metrics.box.map50:.4f}")
    print(f"mAP50-95: {val_metrics.box.map:.4f}")
    print(f"mAP75: {val_metrics.box.map75:.4f}")
    if hasattr(val_metrics.box, "maps"):
        print(f"Per-class mAP50-95: {val_metrics.box.maps}")

    threshold_df, best_conf_threshold, best_iou_threshold = train_pipeline.sweep_thresholds(
        model=best_model,
        data_yaml=yolo_data_yaml,
        image_size=config.YOLO_IMAGE_SIZE,
        batch_size=config.YOLO_BATCH,
        device=config.default_device(),
    )
    print(threshold_df.head())

    # 手動調整IoU閾值以便提交
    best_iou_threshold = 0.68

    print(f"最佳閾值 -> 置信度: {best_conf_threshold:.3f}, IoU: {best_iou_threshold:.2f}")
    print(f"訓練輸出目錄: {Path(save_dir)}")


if __name__ == "__main__":
    main()
