from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from transformer import *
from collections import defaultdict, deque

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


source_video_path = "vehicle-counting.mp4"


model = YOLO("yolov8x.pt")


# Open the video file to get its dimensions
cap = cv2.VideoCapture(source_video_path)
"""
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
"""

# Create a window with the size matching the video resolution
"""
cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotated Frame", frame_width, frame_height)
"""

video_info = sv.VideoInfo.from_video_path(source_video_path)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

# Initialize the VideoWriter
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define the codec
output_video = cv2.VideoWriter(
    output_video_path, fourcc, video_info.fps, (video_info.width, video_info.height)
)

byte_track = sv.ByteTrack(frame_rate=video_info.fps)

bounding_box_annotator = sv.BoxAnnotator(
    thickness=thickness,
    color_lookup=sv.ColorLookup.TRACK,
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER,
    color_lookup=sv.ColorLookup.TRACK,
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER,
    color_lookup=sv.ColorLookup.TRACK,
)

frame_generator = sv.get_video_frames_generator(source_video_path)
polygon_zone = sv.PolygonZone(SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

for frame in frame_generator:

    result = model(frame)[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = detections[polygon_zone.trigger(detections)]
    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points).astype(int)

    # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    # labels = [f"#x:{x},y: {y}" for x, y in points]

    for tracker_id, [_, y] in zip(
        detections.tracker_id, points
    ):  # Loop through both detections and points simultaneously
        coordinates[tracker_id].append(y)

    labels = []

    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            coordinates_start = coordinates[tracker_id][-1]
            coordinates_end = coordinates[tracker_id][0]
            distance = abs(coordinates_end - coordinates_start)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    annotated_frame = frame.copy()
    """annotated_frame = sv.draw_polygon(
        scene=frame,
        polygon=SOURCE,
        color=sv.Color.RED,
    )"""

    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
    )

    # Write the annotated frame to the output video
    output_video.write(annotated_frame)

    # cv2.imshow("Annotated Frame", annotated_frame)

    # Wait for 1 ms and check if 'q' is pressed to exit
    # Wait time in milliseconds
    # if cv2.waitKey(int(100 / fps)) == ord("q"):
    # break

# Release all resources and close OpenCV windows
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as {output_video_path}")
