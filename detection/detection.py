from ultralytics import YOLO
import os
import pickle
from get_keypoint import GetKeypoint
import pandas as pd


class Detection():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.get_keypoint = GetKeypoint()

    def detect(self, frames, read_from_stub=False, stub_path=None):
        # Read from stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Detect keypoints
        detections = []
        for frame in frames:
            # Get keypoints
            results = self.model.predict(frame)
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]

            # Assign keypoints to variables
            nose = keypoints[self.get_keypoint.NOSE]
            left_shoulder = keypoints[self.get_keypoint.LEFT_SHOULDER]
            right_shoulder = keypoints[self.get_keypoint.RIGHT_SHOULDER]
            left_elbow = keypoints[self.get_keypoint.LEFT_ELBOW]
            right_elbow = keypoints[self.get_keypoint.RIGHT_ELBOW]
            left_wrist = keypoints[self.get_keypoint.LEFT_WRIST]
            right_wrist = keypoints[self.get_keypoint.RIGHT_WRIST]
            left_hip = keypoints[self.get_keypoint.LEFT_HIP]
            right_hip = keypoints[self.get_keypoint.RIGHT_HIP]
            left_knee = keypoints[self.get_keypoint.LEFT_KNEE]
            right_knee = keypoints[self.get_keypoint.RIGHT_KNEE]
            left_ankle = keypoints[self.get_keypoint.LEFT_ANKLE]
            right_ankle = keypoints[self.get_keypoint.RIGHT_ANKLE]

            # Create dictionary
            keypoints_dict = {
                'nose': nose,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder,
                'left_elbow': left_elbow,
                'right_elbow': right_elbow,
                'left_wrist': left_wrist,
                'right_wrist': right_wrist,
                'left_hip': left_hip,
                'right_hip': right_hip,
                'left_knee': left_knee,
                'right_knee': right_knee,
                'left_ankle': left_ankle,
                'right_ankle': right_ankle,
            }

            # Append to detections
            detections.append(keypoints_dict)

        # Save to stub
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)

        return detections

    def interpolate(self, detections):
        '''Some values are missing in the nose, due to subject turning away from the camera.'''

        # Convert to DataFrame
        df = pd.DataFrame(detections)

        # Get nose keypoints
        nose_x = df['nose'].apply(lambda x: x[0])
        nose_y = df['nose'].apply(lambda x: x[1])

        # Replace 0 with NaN
        nose_x = nose_x.replace(0, float('nan'))
        nose_y = nose_y.replace(0, float('nan'))

        # Interpolate missing keypoints
        nose_x = nose_x.interpolate(method='linear')
        nose_y = nose_y.interpolate(method='linear')

        # Convert to list
        nose = list(zip(nose_x, nose_y))

        # Replace nose keypoints
        df['nose'] = nose

        # Convert to dictionary
        interpolated_detections = df.to_dict(orient='records')

        return interpolated_detections
