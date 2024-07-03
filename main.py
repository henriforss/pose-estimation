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
wireframe = True
draw = Draw()

if wireframe:
    blank_frames = draw.create_blank_frames(frames)
    output_frames = draw.draw_wireframe(blank_frames, interpolated_detections)
else:
    output_frames = draw.draw_wireframe(frames, interpolated_detections)
    masked_overlay = draw.create_masked_overlay('data/images/donald_duck.jpg')
    output_frames = draw.draw_donald_duck_face(
        output_frames, interpolated_detections, masked_overlay)

# Save video
save_video(output_frames, 'output/output_video_wireframe.mp4')
