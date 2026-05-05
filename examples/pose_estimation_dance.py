"""
Pose Estimation for Dance Video using MediaPipe
Extracts human body keypoints from video and saves to CSV format compatible with VAME

This script uses MediaPipe Pose to extract body landmarks from a video
and saves them to CSV format similar to DeepLabCut output.
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path


class PoseEstimator:
    """Extract pose landmarks from video using MediaPipe"""
    
    def __init__(self, video_path: str, output_csv: str):
        """
        Initialize pose estimator
        
        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
        """
        self.video_path = video_path
        self.output_csv = output_csv
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Get body part names from MediaPipe
        self.landmarks_list = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
    
    def extract_poses(self):
        """
        Extract poses from video and save to CSV
        
        Returns:
            DataFrame with pose landmarks
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video info: {frame_count} frames at {fps} fps")
        
        # Storage for all landmarks
        all_landmarks = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks
            frame_data = {'frame': frame_idx}
            
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    body_part = self.landmarks_list[idx]
                    frame_data[f'{body_part}_x'] = landmark.x
                    frame_data[f'{body_part}_y'] = landmark.y
                    frame_data[f'{body_part}_conf'] = landmark.visibility
            else:
                # No landmarks detected, fill with NaN
                for body_part in self.landmarks_list:
                    frame_data[f'{body_part}_x'] = np.nan
                    frame_data[f'{body_part}_y'] = np.nan
                    frame_data[f'{body_part}_conf'] = np.nan
            
            all_landmarks.append(frame_data)
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Processed frame {frame_idx}/{frame_count}")
        
        cap.release()
        
        # Create DataFrame
        df = pd.DataFrame(all_landmarks)
        
        # Save to CSV
        df.to_csv(self.output_csv, index=False)
        print(f"\nPose data saved to: {self.output_csv}")
        print(f"Total frames: {len(df)}")
        print(f"Landmarks extracted: {len(self.landmarks_list)}")
        
        return df
    
    def get_selected_landmarks(self, landmark_names: list = None):
        """
        Get subset of landmarks for VAME analysis
        
        Args:
            landmark_names: List of landmark names to extract
            
        Returns:
            DataFrame with selected landmarks
        """
        if landmark_names is None:
            # Default selection: key joints for dance analysis
            landmark_names = [
                'nose',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle'
            ]
        
        df = pd.read_csv(self.output_csv)
        
        # Select frame and chosen landmarks
        selected_cols = ['frame']
        for landmark in landmark_names:
            selected_cols.extend([f'{landmark}_x', f'{landmark}_y'])
        
        df_selected = df[selected_cols].copy()
        
        # Save selected landmarks
        output_selected = self.output_csv.replace('.csv', '_selected.csv')
        df_selected.to_csv(output_selected, index=False)
        print(f"Selected landmarks saved to: {output_selected}")
        
        return df_selected


def main():
    """Main execution"""
    import sys
    
    # Define paths
    # Define paths - use absolute paths from project root
    project_root = Path(__file__).parent.parent
    video_path = project_root / "Images" / "Irish-dance-video-15sec.mp4"
    output_dir = project_root / "results_dance_pose"
    output_dir.mkdir(exist_ok=True)
    
    csv_output = output_dir / "irish_dance_pose.csv"
    
    print("=" * 70)
    print("Irish Dance Pose Estimation with MediaPipe")
    print("=" * 70)
    
    print(f"Video path: {video_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Initialize and run pose estimation
        estimator = PoseEstimator(str(video_path), str(csv_output))
        df = estimator.extract_poses()
        
        # Get selected landmarks for VAME
        print("\n" + "=" * 70)
        print("Extracting selected landmarks for VAME analysis...")
        print("=" * 70)
        df_selected = estimator.get_selected_landmarks()
        
        print("\n" + "=" * 70)
        print("SUCCESS: Pose estimation complete!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. The CSV file is ready for VAME processing")
        print(f"2. Initialize VAME project with the output CSV")
        print(f"3. Preprocess and align the pose data")
        print(f"4. Train VAME model to discover dance motifs")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
