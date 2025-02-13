# DL_YOLO_CV

The repository contains two files containing code:

- The first is the notebook that is concerned with carrying out object detection inference on images and videos using the ```Ultralytics YOLOv8``` Python library.

- The second is the main.py script that carries out object detection in real time through a webcam to detect whether objects are present inside a zone - the right hand side of the screen dynamically defined by a red rectangle. The zone does not show the presence of the class 'person' amongst its detections. Additionally, an agnostic Non-Max-Suppression (a-NMS) algorithm is included so as to prevent superfluous bounding boxes being applied to the same or very near objects. The bounding boxes are realised through the ```supervision``` library, while the dynamic command-line level window specification is brought about through the ```argparse``` library.
