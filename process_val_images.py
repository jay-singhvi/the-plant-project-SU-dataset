# Version 2
import os
import cv2
import numpy as np
from PIL import Image


def load_existing_labels(label_path):
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            return [line.strip().split() for line in f.readlines()]
    return []


def convert_yolo_to_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = map(float, yolo_bbox)
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]


def convert_bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    return [x_center, y_center, width, height]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)


def process_image(image_path, label_path, model, iou_threshold=0.5):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    existing_labels = load_existing_labels(label_path)
    existing_bboxes = [
        convert_yolo_to_bbox(label[1:], img_width, img_height)
        for label in existing_labels
    ]
    results = model(img)
    new_labels = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        for det in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if cls == 0:  # Assuming 0 is the class for plants
                new_bbox = [int(x1), int(y1), int(x2), int(y2)]
                is_new = all(
                    iou(new_bbox, existing_bbox) < iou_threshold
                    for existing_bbox in existing_bboxes
                )
                if is_new:
                    yolo_bbox = convert_bbox_to_yolo(new_bbox, img_width, img_height)
                    new_labels.append([0] + yolo_bbox)
                    existing_bboxes.append(new_bbox)
    return new_labels


def process_val_images(model):
    val_images_path = "data/images/val"
    val_labels_path = "data/labels/val"
    os.makedirs(val_labels_path, exist_ok=True)

    for image_file in os.listdir(val_images_path):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(val_images_path, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(val_labels_path, label_file)
            new_labels = process_image(image_path, label_path, model)
            with open(label_path, "a") as f:
                for label in new_labels:
                    f.write(" ".join(map(str, label)) + "\n")
            print(
                f"Processed {image_file}: {len(new_labels)} new plants detected and labeled"
            )


if __name__ == "__main__":
    model = torch.hub.load("ultralytics/yolov10", "yolov10x")
    process_val_images(model)
