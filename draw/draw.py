from get_keypoint import GetKeypoint
import cv2
import numpy as np


class Draw():
    def __init__(self):
        self.get_keypoint = GetKeypoint()
        self.frame_num = 0

    def create_masked_overlay(self, path_to_overlay_image):
        # Read overlay image
        overlay = cv2.imread(path_to_overlay_image)

        # Resize overlay image
        overlay = cv2.resize(overlay, (140, 140))

        # Create 4D mask (for transparency)
        mask = np.zeros_like(overlay[:, :, 0])

        # Create a circle mask
        cv2.circle(mask, (70, 70), 70, 255, -1)

        # Create a boolean mask
        mask_bool = mask.astype(bool)

        # Create a masked overlay
        masked_overlay = np.zeros(
            overlay.shape[:2] + (4,), dtype=overlay.dtype)

        # Apply the mask
        masked_overlay[:, :, :3] = overlay

        # Apply the transparency
        masked_overlay[:, :, 3] = np.where(mask_bool, 255, 0)

        return masked_overlay

    def draw_donald_duck_face(self, frames, detections, masked_overlay):
        output_frames = []

        # Loop through frames and detections
        for frame, detection in zip(frames, detections):
            nose = detection['nose']
            left_shoulder = detection['left_shoulder']
            right_shoulder = detection['right_shoulder']
            left_elbow = detection['left_elbow']
            right_elbow = detection['right_elbow']
            left_wrist = detection['left_wrist']
            right_wrist = detection['right_wrist']
            left_hip = detection['left_hip']
            right_hip = detection['right_hip']
            left_knee = detection['left_knee']
            right_knee = detection['right_knee']
            left_ankle = detection['left_ankle']
            right_ankle = detection['right_ankle']

            left_body_parts = [left_shoulder, left_elbow,
                               left_wrist, left_hip, left_knee, left_ankle]
            right_body_parts = [right_shoulder, right_elbow,
                                right_wrist, right_hip, right_knee, right_ankle]

            # If nose is visible
            if np.any(nose):
                # Position overlay image
                y_offset = int(nose[1]) - masked_overlay.shape[0] // 2 - 30
                x_offset = int(nose[0]) - masked_overlay.shape[1] // 2

                # Overlay image
                for y in range(masked_overlay.shape[0]):
                    for x in range(masked_overlay.shape[1]):
                        if masked_overlay[y, x, 3] > 0:
                            alpha = masked_overlay[y, x, 3] / 255.0
                            frame[y + y_offset, x + x_offset] = alpha * masked_overlay[y,
                                                                                       x, :3] + (1 - alpha) * frame[y + y_offset, x + x_offset]

            # Draw circles around left body parts
            for body_part in left_body_parts:
                if np.any(body_part):
                    # Draw a circle around body parts
                    cv2.circle(frame, (int(body_part[0]), int(body_part[1])),
                               5, (0, 0, 255), -1)

            # Draw circles around right body parts
            for body_part in right_body_parts:
                if np.any(body_part):
                    # Draw a circle around body parts
                    cv2.circle(frame, (int(body_part[0]), int(body_part[1])),
                               5, (0, 255, 0), -1)

            # Draw lines between body parts
            if np.any(left_shoulder) and np.any(right_shoulder):
                cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                         (int(right_shoulder[0]), int(right_shoulder[1]),), (200, 200, 200), 2)

            if np.any(left_hip) and np.any(right_hip):
                cv2.line(frame, (int(left_hip[0]), int(left_hip[1])),
                         (int(right_hip[0]), int(right_hip[1]),), (200, 200, 200), 2)

            if np.any(left_shoulder) and np.any(left_hip):
                cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                         (int(left_hip[0]), int(left_hip[1])), (200, 200, 200), 2)

            if np.any(right_shoulder) and np.any(right_hip):
                cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])),
                         (int(right_hip[0]), int(right_hip[1])), (200, 200, 200), 2)

            if np.any(left_shoulder) and np.any(left_elbow):
                cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                         (int(left_elbow[0]), int(left_elbow[1])), (0, 0, 255), 2)

            if np.any(left_elbow) and np.any(left_wrist):
                cv2.line(frame, (int(left_elbow[0]), int(left_elbow[1])),
                         (int(left_wrist[0]), int(left_wrist[1])), (0, 0, 255), 2)

            if np.any(left_hip) and np.any(left_knee):
                cv2.line(frame, (int(left_hip[0]), int(left_hip[1])),
                         (int(left_knee[0]), int(left_knee[1])), (0, 0, 255), 2)

            if np.any(left_knee) and np.any(left_ankle):
                cv2.line(frame, (int(left_knee[0]), int(left_knee[1])),
                         (int(left_ankle[0]), int(left_ankle[1])), (0, 0, 255), 2)

            if np.any(right_shoulder) and np.any(right_elbow):
                cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])),
                         (int(right_elbow[0]), int(right_elbow[1])), (0, 255, 0), 2)

            if np.any(right_elbow) and np.any(right_wrist):
                cv2.line(frame, (int(right_elbow[0]), int(right_elbow[1])),
                         (int(right_wrist[0]), int(right_wrist[1])), (0, 255, 0), 2)

            if np.any(right_hip) and np.any(right_knee):
                cv2.line(frame, (int(right_hip[0]), int(right_hip[1])),
                         (int(right_knee[0]), int(right_knee[1])), (0, 255, 0), 2)

            if np.any(right_knee) and np.any(right_ankle):
                cv2.line(frame, (int(right_knee[0]), int(right_knee[1])),
                         (int(right_ankle[0]), int(right_ankle[1])), (0, 255, 0), 2)

            output_frames.append(frame)
            self.frame_num += 1

        return output_frames
