import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp


skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],  # legs and center
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],  # body and arms
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],  # arms, face
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],  # face to shoulders to arms
]

# Define colors for different body parts
pose_palette = np.array(
    [
        [51, 153, 255],
        [51, 153, 255],
        [51, 153, 255],
        [51, 153, 255],
        [51, 153, 255],  # legs
        [255, 51, 51],
        [255, 51, 51],
        [255, 51, 51],
        [255, 102, 66],
        [255, 102, 66],  # body and arms
        [255, 102, 66],
        [255, 102, 66],
        [51, 153, 51],
        [51, 153, 51],
        [51, 153, 51],  # arms and face
        [51, 153, 51],
        [51, 153, 51],
        [51, 153, 51],
        [51, 153, 51],  # face to shoulders
    ],
    dtype=np.uint8,
).tolist()


def draw_pose_estimation(frame, results):
    """
    Draw pose estimation results on the frame including keypoints and bones.

    Args:
        frame: Input image/frame to draw on
        results: YOLO results object containing pose estimation data

    Returns:
        Frame with pose estimation drawn on it
    """
    # Define the pose skeleton connections for drawing bones
    # These are the connections between keypoints for human pose estimation

    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Process results
    for r in results:
        # Plot boxes and poses on the frame
        annotated_frame = r.plot()

        # Draw skeleton if keypoints are available
        if hasattr(r, "keypoints") and r.keypoints is not None:
            # Get the keypoints data
            keypoints = (
                r.keypoints.data
            )  # Shape: (num_persons, num_keypoints, 3) -> (x, y, conf)

            # Iterate through each person detected
            for person_kpts in keypoints:
                if person_kpts is not None:
                    # Draw connections (bones) between keypoints
                    for i, sk in enumerate(skeleton):
                        pos1 = (
                            int(person_kpts[sk[0] - 1][0]),
                            int(person_kpts[sk[0] - 1][1]),
                        )
                        pos2 = (
                            int(person_kpts[sk[1] - 1][0]),
                            int(person_kpts[sk[1] - 1][1]),
                        )

                        # Check if both points have high confidence
                        conf1 = person_kpts[sk[0] - 1][2]
                        conf2 = person_kpts[sk[1] - 1][2]

                        if (
                            conf1 > 0.5
                            and conf2 > 0.5
                            and pos1[0] > 0
                            and pos1[1] > 0
                            and pos2[0] > 0
                            and pos2[1] > 0
                        ):
                            # Draw the bone (line between keypoints)
                            color = pose_palette[i]
                            cv2.line(
                                annotated_frame,
                                pos1,
                                pos2,
                                color,
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )

    return annotated_frame


def draw_mediapipe_pose_estimation(frame, pose_results):
    """
    Draw MediaPipe pose estimation results on the frame.

    Args:
        frame: Input image/frame to draw on
        pose_results: MediaPipe pose results object

    Returns:
        Frame with pose estimation drawn on it
    """
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()

    if pose_results.pose_landmarks:
        # Draw pose landmarks and connections
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0), thickness=2
            )
        )

    return annotated_frame
