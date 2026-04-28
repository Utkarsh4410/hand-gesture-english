"""
Shared feature extraction for hand gesture recognition.
Normalizes landmarks and computes engineered features for better accuracy.
"""
import numpy as np


def extract_features_from_raw(raw_coords):
    """
    Takes a flat list of 63 values (21 landmarks × 3: x, y, z)
    and returns an engineered feature vector.
    """
    coords = np.array(raw_coords).reshape(21, 3)
    return _compute_features(coords)


def extract_features_from_landmarks(hand_landmarks):
    """
    Takes a mediapipe hand_landmarks object and returns an engineered feature vector.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    return _compute_features(coords)


def _compute_features(coords):
    """
    Compute normalized + engineered features from 21×3 landmark array.

    Features:
    1. Wrist-centered, scale-normalized coordinates (21×3 = 63)
    2. Distances from each fingertip to wrist (5)
    3. Distances between adjacent fingertips (4)
    4. Distances from each fingertip to palm center (5)
    5. Finger curl ratios — how bent each finger is (5)
    6. Angles between finger directions (4)
    Total: 63 + 5 + 4 + 5 + 5 + 4 = 86 features
    """
    features = []

    # --- 1. Normalize: wrist-centered, scale by palm size ---
    wrist = coords[0]
    centered = coords - wrist  # translate to wrist origin

    # Scale by distance from wrist to middle finger MCP (landmark 9)
    palm_size = np.linalg.norm(centered[9])
    if palm_size < 1e-6:
        palm_size = 1e-6  # avoid division by zero
    normalized = centered / palm_size

    features.extend(normalized.flatten())  # 63 features

    # --- 2. Fingertip to wrist distances ---
    fingertip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    for tip_id in fingertip_ids:
        dist = np.linalg.norm(normalized[tip_id])
        features.append(dist)  # 5 features

    # --- 3. Adjacent fingertip distances ---
    for i in range(len(fingertip_ids) - 1):
        dist = np.linalg.norm(normalized[fingertip_ids[i]] - normalized[fingertip_ids[i + 1]])
        features.append(dist)  # 4 features

    # --- 4. Fingertip to palm center distances ---
    palm_landmarks = [0, 1, 5, 9, 13, 17]  # wrist + finger MCPs
    palm_center = np.mean(normalized[palm_landmarks], axis=0)
    for tip_id in fingertip_ids:
        dist = np.linalg.norm(normalized[tip_id] - palm_center)
        features.append(dist)  # 5 features

    # --- 5. Finger curl ratios ---
    # Each finger: ratio of (tip-to-mcp distance) / (sum of bone lengths)
    finger_joints = [
        [1, 2, 3, 4],     # thumb
        [5, 6, 7, 8],     # index
        [9, 10, 11, 12],  # middle
        [13, 14, 15, 16], # ring
        [17, 18, 19, 20], # pinky
    ]
    for joints in finger_joints:
        # Direct distance from MCP to tip
        tip_dist = np.linalg.norm(normalized[joints[-1]] - normalized[joints[0]])
        # Sum of bone segment lengths
        bone_length = sum(
            np.linalg.norm(normalized[joints[i + 1]] - normalized[joints[i]])
            for i in range(len(joints) - 1)
        )
        if bone_length < 1e-6:
            bone_length = 1e-6
        curl = tip_dist / bone_length  # 1.0 = straight, 0.0 = fully curled
        features.append(curl)  # 5 features

    # --- 6. Angles between adjacent finger directions ---
    finger_directions = []
    for joints in finger_joints:
        direction = normalized[joints[-1]] - normalized[joints[0]]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.zeros(3)
        else:
            direction = direction / norm
        finger_directions.append(direction)

    for i in range(len(finger_directions) - 1):
        dot = np.clip(np.dot(finger_directions[i], finger_directions[i + 1]), -1.0, 1.0)
        angle = np.arccos(dot) / np.pi  # normalize to [0, 1]
        features.append(angle)  # 4 features

    return np.array(features)
