from detection import Detection
from draw import Draw
from utils import read_video, save_video


# Read video
frames = read_video('data/original_video_no_audio_square.mp4')

# Detect keypoints
detection = Detection('models/yolov8n-pose.pt')
detections = detection.detect(
    frames, read_from_stub=True, stub_path='stubs/detection_stubs.pkl')

# Interpolate missing keypoints
interpolated_detections = detection.interpolate(detections)

# Draw on frame
draw = Draw()
masked_overlay = draw.create_masked_overlay('data/images/donald_duck.jpg')
output_frames = draw.draw_donald_duck_face(
    frames, interpolated_detections, masked_overlay)


# Save video
save_video(output_frames, 'output/output_video.mp4')
