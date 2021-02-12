import cv2
import torch

from torching.vision.structures.box_list import BoxList


# def draw_detections(canvas, boxes, labels, scores, color=(255, 0, 0), thickness=2):
#     for box, label, score in zip(boxes, labels, scores):
#         box = list(map(int, box))
#         cv2.rectangle(canvas, tuple([box[0], box[1]]), tuple([box[2], box[3]]), color, thickness=thickness)
#         text_x, text_y = box[0], box[3]
#         cv2.putText(canvas, '{}, {:.4f}'.format(int(label), score), (text_x, text_y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def draw_boxlist(canvas, box_list, color=(255, 0, 0), thickness=2):
    if isinstance(box_list, BoxList):
        boxes = box_list.xyxy().to_tensor().int().numpy()

        for i in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[i, ...]
            print(xmin, ymin, xmax, ymax)
            cv2.rectangle(canvas, tuple([xmin, ymin]), tuple([xmax, ymax]), color, thickness=thickness)
