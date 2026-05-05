"""Run a full VAME demo pipeline on the Irish dance video.

This script converts MediaPipe output to a DeepLabCut-like CSV, initializes
an Irish dance VAME project, and runs preprocessing/training/segmentation.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import pandas as pd
import vame


KEYPOINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_video_info(video_path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return width, height, fps


def convert_mediapipe_to_dlc(
    input_csv: Path,
    output_csv: Path,
    video_width: int,
    video_height: int,
    scorer_name: str = "MediaPipePose",
) -> None:
    """Convert flat MediaPipe CSV to DeepLabCut-like 3-row header CSV."""
    df = pd.read_csv(input_csv)

    for kp in KEYPOINTS:
        if f"{kp}_x" not in df.columns or f"{kp}_y" not in df.columns:
            raise ValueError(f"Missing keypoint columns for {kp}")

    header_scorer = ["scorer"]
    header_bodyparts = ["bodyparts"]
    header_coords = ["coords"]

    for kp in KEYPOINTS:
        header_scorer.extend([scorer_name, scorer_name, scorer_name])
        header_bodyparts.extend([kp, kp, kp])
        header_coords.extend(["x", "y", "likelihood"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_scorer)
        writer.writerow(header_bodyparts)
        writer.writerow(header_coords)

        for idx, row in df.iterrows():
            out_row = [idx]
            for kp in KEYPOINTS:
                x = float(row[f"{kp}_x"]) if pd.notna(row[f"{kp}_x"]) else float("nan")
                y = float(row[f"{kp}_y"]) if pd.notna(row[f"{kp}_y"]) else float("nan")
                conf_col = f"{kp}_conf"
                likelihood = float(row[conf_col]) if conf_col in df.columns and pd.notna(row[conf_col]) else 1.0

                # MediaPipe coordinates are normalized [0, 1], VAME expects image-space pixels.
                out_row.extend([x * video_width, y * video_height, likelihood])
            writer.writerow(out_row)


def run_pipeline(project_root: Path, run_train: bool = True) -> None:
    video_path = project_root / "Images" / "Irish-dance-video-15sec.mp4"
    mediapipe_csv = project_root / "results_dance_pose" / "irish_dance_pose.csv"
    dlc_csv = project_root / "results_dance_pose" / "irish_dance_pose_dlc.csv"

    width, height, fps = get_video_info(video_path)
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps:.3f} fps")

    print("Converting MediaPipe CSV to DeepLabCut-like CSV...")
    convert_mediapipe_to_dlc(mediapipe_csv, dlc_csv, width, height)
    print(f"Converted CSV: {dlc_csv}")

    project_name = "irish-dance-vame"
    project_path = project_root / project_name

    # If a previous initialization failed before config creation, reset the folder.
    if project_path.exists() and not (project_path / "config.yaml").exists():
        shutil.rmtree(project_path)

    print(f"Initializing VAME project: {project_name}")

    config_path, config = vame.init_new_project(
        project_name=project_name,
        poses_estimations=[str(dlc_csv)],
        source_software="DeepLabCut",
        working_directory=str(project_root),
        videos=[str(video_path)],
        video_type=".mp4",
        fps=fps,
        copy_videos=True,
        config_kwargs={
            "project_random_state": 42,
            "max_epochs": 25,
            "model_snapshot": 5,
            "model_convergence": 10,
            "batch_size": 64,
            "zdims": 10,
            "time_window": 20,
            "prediction_steps": 5,
            "n_clusters": 8,
            "hmm_n_iter": 50,
            "pose_confidence": 0.2,
            "savgol_length": 5,
            "savgol_order": 2,
        },
    )
    print(f"Config: {config_path}")

    print("Running preprocessing...")
    latest_var = vame.preprocessing(
        config=config,
        centered_reference_keypoint="left_hip",
        orientation_reference_keypoint="right_hip",
        run_lowconf_cleaning=True,
        run_egocentric_alignment=True,
        run_outlier_cleaning=True,
        run_savgol_filtering=True,
        run_rescaling=True,
    )
    print(f"Preprocessing output variable: {latest_var}")

    print("Creating training set...")
    vame.create_trainset(
        config=config,
        test_fraction=0.15,
        split_mode="mode_2",
        read_from_variable=latest_var,
    )

    if not run_train:
        print("Skipping train/evaluate/segment by request.")
        return

    print("Training model...")
    vame.train_model(config=config)

    print("Evaluating model...")
    vame.evaluate_model(config=config)

    print("Segmenting motifs...")
    vame.segment_session(config=config)

    print("Running community analysis...")
    vame.community(config=config, cut_tree=2)

    print("Pipeline finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VAME pipeline on Irish dance data.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to VAME project root",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only run conversion/init/preprocess/trainset",
    )
    args = parser.parse_args()

    run_pipeline(args.project_root, run_train=not args.no_train)


if __name__ == "__main__":
    main()
