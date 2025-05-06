import argparse
from ultralytics import YOLO
import supervision as sv
import cv2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle speed estimation using interference and supervision"
    )
    parser.add_argument(
        "--Source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()


# source_video_path = "vehicle-counting.mp4"


if __name__ == "__main__":
    args = parse_arguments()
    model = YOLO("yolov8x.pt")
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    frame_generator = sv.get_video_frames_generator(args.Source_video_path)

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
