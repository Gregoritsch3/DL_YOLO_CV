import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

ZONE_POLYGON = np.array([
    [0.5, 0],
    [1, 0],
    [1, 1],
    [0.5, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
        )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(thickness=2)
    labels_annotator = sv.LabelAnnotator()

    zone_polygon_real = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon_real)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

    while True:
        ret, frame = cap.read()

        #Carrying out inference with included Agnostic Non-Max Supression
        result = model(frame, agnostic_nms = True)[0]
        detections = sv.Detections.from_ultralytics(result) 

        labels = [
            f"{model.model.names[class_id]}{confidence:0.2f}"
            for _, _, confidence, class_id, _, _,
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = labels_annotator.annotate(scene=frame, detections=detections, labels=labels)

        #Filtering out 'person' class in zone
        zone.trigger(detections=detections[detections.class_id != 0])
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)
        
        if (cv2.waitKey(30)==27):
            break

if __name__=="__main__":
    main()      