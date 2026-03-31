from ultralytics import YOLO

from src import analysis, config, prediction, train_pipeline


SELECTED_WEIGHT_INDEX = 0


def main() -> None:
    available_weight_paths = prediction.list_available_weight_paths(
        config.YOLO_OUTPUT_DIR / "yolo11x_pig_from_yaml2" / "weights"
    )
    if not available_weight_paths:
        raise FileNotFoundError("在 runs/ 內找不到任何 .pt 權重檔案，請確認已將權重檔案保存於此。")

    print("可用權重檔案：")
    for idx, path in enumerate(available_weight_paths):
        print(f"[{idx}] {path}")

    if SELECTED_WEIGHT_INDEX < 0 or SELECTED_WEIGHT_INDEX >= len(available_weight_paths):
        raise IndexError(f"SELECTED_WEIGHT_INDEX 必須介於 0 和 {len(available_weight_paths) - 1} 之間")

    custom_weight_path = available_weight_paths[SELECTED_WEIGHT_INDEX]
    print(f"使用權重: {custom_weight_path}")

    device = config.default_device()
    model = YOLO(str(custom_weight_path))

    yolo_data_yaml = config.YOLO_DATA_ROOT / "pig_detection.yaml"
    if not yolo_data_yaml.exists():
        raise FileNotFoundError("找不到 pig_detection.yaml，請先執行訓練流程以建立資料集配置。")

    _, best_conf_threshold, best_iou_threshold = train_pipeline.sweep_thresholds(
        model=model,
        data_yaml=yolo_data_yaml,
        image_size=config.YOLO_IMAGE_SIZE,
        batch_size=config.YOLO_BATCH,
        device=device,
    )

    best_iou_threshold = 0.68

    submission_path = config.SUBMISSION_PATH.with_name(f"{custom_weight_path.stem}_submission.csv")
    submission_df = prediction.create_submission(
        model=model,
        sample_path=config.SAMPLE_SUB_PATH,
        test_dir=config.TEST_IMG_DIR,
        output_path=submission_path,
        image_size=config.YOLO_IMAGE_SIZE,
        device=device,
        conf_threshold=best_conf_threshold,
        iou_threshold=best_iou_threshold,
    )
    print(f"使用 {custom_weight_path} 的預測已儲存至 {submission_path}")

    analysis.analyze_submission_results(submission_df)


if __name__ == "__main__":
    main()
