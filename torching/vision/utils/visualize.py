import cv2
import time
import torch
import numpy as np

from torching.vision.structures.box import Box
from torching.vision.structures.box import BoxList
from torching.vision.structures.box import BoxTarget


def _run_video(video_url, callback, frame_interval, fps, win_name, win_size):
    cap = cv2.VideoCapture(video_url)

    cv2.namedWindow(win_name, 0)
    cv2.resizeWindow(win_name, win_size[0], win_size[1])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue
        
        try:
            callback(frame)
        except:
            pass

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1000 // fps)
        if key == ord('q'):
            break


def _show_image(image, win_name, win_size):
    cv2.imshow(win_name, image)
    cv2.waitKey()
    cv2.destroyWindow(win_name)



class AbcDisplay:
    def show(self):
        raise NotImplementedError


class ImageDisplay(AbcDisplay):
    def __init__(self,
                 image,
                 win_name=None,
                 win_size=None):
        self.image = image

        self.win_name = win_name or 'ImageDisplay'
        self.win_size = win_size if win_size is not None else (1024, 768)

    def show(self):
        _show_image(self.image, self.win_name, self.win_size)

    def show_grayscale(self):
        pass


class VideoDisplay(AbcDisplay):
    def __init__(self,
                 video_url, 
                 callback,
                 frame_interval=1,
                 fps=30,
                 win_name=None,
                 win_size=None):

        self.video_url = video_url
        self.callback = callback
        self.frame_interval = frame_interval
        self.fps = fps

        self.win_name = win_name or 'VideoDisplay'
        self.win_size = win_size if win_size is not None else (1024, 768)

    def show(self):
        _run_video(self.video_url, self.callback, self.frame_interval, self.fps, self.win_name, self.win_size)


class BoxListDisplay(AbcDisplay):
    def __init__(self,
                 box_list,
                 image,
                 box_color=None, 
                 box_thickness=2, 
                 text_color=None, 
                 text_size=0.5,
                 win_name=None,
                 win_size=None):

        assert isinstance(box_list, BoxList)
        self.box_list = box_list

        self.image = image

        if box_color is None:
            if self.image.ndim == 2:
                self.box_color = 255
            elif self.image.ndim == 3:
                self.box_color = (255, 0, 0)

        self.box_thickness = box_thickness

        if text_color is None:
            if self.image.ndim == 2:
                self.text_color = 255
            elif self.image.ndim == 3:
                self.text_color = (0, 0, 255)

        self.text_size = text_size

        self.win_name = win_name or 'BoxListDisplay'
        self.win_size = win_size if win_size is not None else (1024, 768)

    def show(self):
        canvas = self.image.copy()
        boxes = self.box_list.xyxy().to_tensor().int().tolist()

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(canvas, tuple([xmin, ymin]), tuple([xmax, ymax]), 
                          self.box_color, thickness=self.box_thickness)

        _show_image(canvas, self.win_name, self.win_size)


class BoxTargetDisplay:
    def __init__(self,
                 target, 
                 image,
                 box_color=None, 
                 box_thickness=2, 
                 text_color=None, 
                 text_size=0.5,
                 win_name=None,
                 win_size=None):

        assert isinstance(target, BoxTargetDisplay)
        self.target = target
        self.image = image

        if box_color is None:
            if self.image.ndim == 2:
                self.box_color = 255
            elif self.image.ndim == 3:
                self.box_color = (255, 0, 0)

        self.box_thickness = box_thickness

        if text_color is None:
            if self.image.ndim == 2:
                self.text_color = 255
            elif self.image.ndim == 3:
                self.text_color = (0, 0, 255)

        self.text_size = text_size

        self.win_name = win_name or 'BoxTargetDisplay'
        self.win_size = win_size if win_size is not None else (1024, 768)

    def show(self):
        canvas = self.image.copy()

        boxes = self.target.box_list.xyxy().to_tensor().int().tolist()
        labels = self.target.labels.flatten().int().tolist()

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(canvas, tuple([xmin, ymin]), tuple([xmax, ymax]), 
                          self.box_color, thickness=self.box_thickness)

            text_loc = (xmin, ymin - 20)
            text = 'LABEL: {}'.format(int(label))
            cv2.putText(canvas, text, text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, 1)

        _show_image(canvas, self.win_name, self.win_size)


# class BoxesDisplay(AbcDisplay):
#     def __init__(self,
#                  box_objs,
#                  image,
#                  box_color=None, 
#                  box_thickness=2, 
#                  text_color=None, 
#                  text_size=0.5,
#                  win_name=None,
#                  win_size=None):

#         for box_obj in box_objs:
#             assert isinstance(box_obj, Box)
#         self.box_objs = box_objs

#         self.image = image

#         if box_color is None:
#             if self.image.ndim == 2:
#                 self.box_color = 255
#             elif self.image.ndim == 3:
#                 self.box_color = (255, 0, 0)

#         self.box_thickness = box_thickness

#         if text_color is None:
#             if self.image.ndim == 2:
#                 self.text_color = 255
#             elif self.image.ndim == 3:
#                 self.text_color = (0, 0, 255)

#         self.text_size = text_size

#         self.win_name = win_name or 'BoxesDisplay'
#         self.win_size = win_size if win_size is not None else (1024, 768)

#     def show(self):
#         canvas = self.image.copy()
#         for box_obj in self.box_objs:
#             loc = box_obj.loc
#             label = box_obj.label
#             score = box_obj.score

#             if loc is not None:
#                 xmin, ymin, xmax, ymax = loc
#                 cv2.rectangle(canvas, tuple([xmin, ymin]), tuple([xmax, ymax]), 
#                             self.box_color, thickness=self.box_thickness)

#             if label is not None:
#                 text_loc = (xmin, ymin - 20)
#                 text = 'LABEL: {}'.format(int(label))
#                 cv2.putText(canvas, text, text_loc,
#                             cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, 1)

#             if score is not None:
#                 text_loc = (xmin, ymin - 5)
#                 text = 'SCORE: {:.2f}'.format(score)
#                 cv2.putText(canvas, text, text_loc,
#                             cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, 1)

#         _show_image(canvas, self.win_name, self.win_size)