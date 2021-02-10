import torch
import numpy as np
import math


class BoxList:
    def __init__(self,
                 data,
                 img_size,
                 mode='xyxy',
                 absolute=True,
                 device=None):

        if device is None:
            if isinstance(data, torch.Tensor):
                device = data.device
            else:
                device = torch.device('cpu')

        self.data = torch.as_tensor(data, dtype=torch.float32, device=device)
        self.img_size = img_size  # (image_width, image_height)
        self.mode = mode
        self.absolute = absolute
    
    @property
    def device(self):
        return self.data.device

    @classmethod
    def create_like(cls, other, data=None):
        return cls(data if data is not None else other.data.clone(),
                   other.img_size,
                   other.mode,
                   other.absolute,
                   other.device)

    def xyxy(self):
        if self.mode == 'xywh':
            x, y, w, h = self.data.split(1, dim=-1)
            data = torch.cat([x, y, x + w, y + h], dim=0)
            return BoxList(data, self.img_size, mode='xywh')
        return self.clone()

    def xywh(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.data.split(1, dim=-1)
            data = torch.cat([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)
            return BoxList(data, self.img_size, mode='xywh')
        return self.clone()

    def absolute_coords(self):
        if self.mode == 'xyxy' and not self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] *= self.img_size[0]
            data[:, [2, 3]] *= self.img_size[1]
            return BoxList(data, self.img_size, mode='xyxy', absolute=True)
        elif self.mode == 'xywh' and not self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] *= self.img_size[0]
            data[:, [2, 3]] *= self.img_size[1]
            return BoxList(data, self.img_size, mode='xywh', absolute=True)

    def percent_coords(self):
        if self.mode == 'xyxy' and self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] /= self.img_size[0]
            data[:, [2, 3]] /= self.img_size[1]
            return BoxList(data, self.img_size, mode='xyxy', absolute=False)
        elif self.mode == 'xywh' and self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] /= self.img_size[0]
            data[:, [2, 3]] /= self.img_size[1]
            return BoxList(data, self.img_size, mode='xywh', absolute=False)

    def clone(self):
        return self.create_like(self)

    def to(self, device=None):
        if device != self.data.device:
            return self.data.to(device)
        return self.data

    def resize(self, size):
        data = self.xyxy().data

        width_ratio = size[0] / self.img_size[0]
        height_ratio = size[1] / self.img_size[1]
        data[:, [0, 2]] *= width_ratio
        data[:, [1, 3]] *= height_ratio
        
        obj = BoxList(data, size, absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    def crop(self, region):
        origin_xmin, origin_ymin, origin_xmax, origin_ymax = self.xyxy().data.split(1, dim=-1)

        if self.mode == 'xyxy':
            region_xmin, region_ymin, region_xmax, region_ymax = region
            region_w, region_h = region_xmax - region_xmin, region_ymax - region_ymin
        elif self.mode == 'xywh':
            region_xmin, region_ymin = region[:2]
            region_xmax, region_ymax = region[:2] - region[2:]
            region_w, region_h = region[2:]

        data = torch.cat([(origin_xmin - region_xmin).clamp(min=0, max=region_w),
                          (origin_ymin - region_ymin).clamp(min=0, max=region_h),
                          (region_xmax - origin_xmax).clamp(min=0, max=region_w),
                          (region_ymax - origin_ymax).clamp(min=0, max=region_h)], dim=-1)

        obj = BoxList(data, (w, h), absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    def expand(self, ratios):
        ones = [1.0] * 4
        assert len(ratios) <= 4
        
        ratios = ratios + ones[len(ones):]
        
        expaned_img_l = self.img_size[0] // 2 * (1 - ratios[0])
        expaned_img_t = self.img_size[1] // 2 * (1 - ratios[1])
        expaned_img_r = self.img_size[0] + self.img_size[0] // 2 * (ratios[2] - 1)
        expaned_img_b = self.img_size[1] + self.img_size[1] // 2 * (ratios[3] - 1)
        expaned_img_w = expaned_img_r - expaned_img_l
        expaned_img_h = expaned_img_b - expaned_img_t

        origin_xmin, origin_ymin, origin_xmax, origin_ymax = self.xyxy().data.split(1, dim=-1)

        data = torch.cat([(origin_xmin - expaned_img_l).clamp(min=0, max=expaned_img_w),
                          (origin_ymin - expaned_img_t).clamp(min=0, max=expaned_img_h),
                          (expaned_img_r - origin_xmax).clamp(min=0, max=expaned_img_w),
                          (expaned_img_b - origin_ymax).clamp(min=0, max=expaned_img_h)], dim=-1)

        obj = BoxList(data, (expaned_img_w, expaned_img_h), absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    def translate(self, x, y):
        data = self.xyxy().data

        data[:, [0, 2]].add_(x).clamp_(0, self.img_size[0])
        data[:, [1, 3]].add_(y).clamp_(0, self.img_size[1])

        obj = BoxList(data, self.img_size, absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    def vflip(self):
        data = self.xyxy().data
        data[:, [1, 3]] = self.img_size[1] - data[:, [3, 1]]
        obj = BoxList(data, self.img_size, absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    def hflip(self):
        data = self.xyxy().data
        data[:, [0, 2]] = self.img_size[0] - data[:, [2, 0]]
        obj = BoxList(data, self.img_size, absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    # def rotate90(self):
    #     def gen_rotate_mat(theta):
    #         M = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    #         return torch.as_tensor(M, dtype=torch.float32, device=self.device)

    #     img_w, img_h = self.img_size
    #     data = self.xyxy().data

    #     ratate_mat = gen_rotate_mat(np.pi / 2.0)
    #     data[:, 1::2] = img_h - data[:, 1::2]
    #     data = torch.cat([ratate_mat.mm(data[:, :2].T),
    #                       ratate_mat.mm(data[:, 2:].T)], dim=0).T
    #     data[:, 1::2] *= -1

    #     obj = BoxList(data, (img_h, img_w), absolute=self.absolute)
    #     if self.mode == 'xywh':
    #         obj = obj.xywh()
    #     return obj

    def rotate90(self, center=None):
        return self.rotate(90, center)

    def rotate(self, angle, center=None):
        return self.affine(angle)

    def affine(self, angle, translate=None, center=None):
        angle = angle % 360
        angle = math.radians(angle)

        def get_affine_mat_in_pixel_grid(theta):
            """
            Notes: In the image pixel coordinate, the transformation is:
                x_ = cos(theta) * x - sin(theta) * y
                y_ = sin(theta) * x + cos(theta) * y
            """
            M = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]]
            return torch.as_tensor(M, dtype=torch.float32, device=self.device)

        def transform_points(points, M):
            if points.ndim == 1:
                points = points.unsqueeze(dim=0)

            return points.mm(M[:2, :2].T) + M[:2, -1].T

        def transform_boxes(boxes, M):
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(dim=0)

            boxes = torch.cat([transform_points(boxes[:, :2], M),
                               transform_points(boxes[:, 2:], M)], dim=-1)
            return boxes

        def get_out_box(points):
            xmin = points[:, 0].min()
            xmax = points[:, 0].max()
            ymin = points[:, 1].min()
            ymax = points[:, 1].max()
            box = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=points.device)
            return box

        img_w, img_h = self.img_size
        boxes = self.xyxy().data

        if center is None:
            center = torch.as_tensor([0, 0], dtype=torch.float32, device=self.device)

        M = get_affine_mat_in_pixel_grid(angle)
        affined_center = transform_points(-center, M)
        M[0, 2] = affined_center[0, 0]
        M[1, 2] = affined_center[0, 1]

        affined_boxes = transform_boxes(boxes, M)
        
        if self.absolute:
            img_corners = torch.as_tensor([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]],
                                          dtype=torch.float32, device=self.device)
        else:
            img_corners = torch.as_tensor([[0, 0], [1, 0], [1, 1], [0, 1]],
                                          dtype=torch.float32, device=self.device)
        
        print('\naffined_boxes:\n', affined_boxes)
        print('\nimg_corners:\n', img_corners)
        affined_img_corners = transform_points(img_corners, M)
        print('\naffined_img_corners:\n', affined_img_corners)
        affined_img_grid = get_out_box(affined_img_corners)
        affined_boxes[:, [0, 2]] -= affined_img_grid[0]
        affined_boxes[:, [1, 3]] -= affined_img_grid[1]
        print('\naffined_img_grid:\n', affined_img_grid)

        print('\naffined_boxes:\n', affined_boxes)

        affined_img_w = (affined_img_grid[2] - affined_img_grid[0]).int().item()
        affined_img_h = (affined_img_grid[3] - affined_img_grid[1]).int().item()

        obj = BoxList(affined_boxes, (affined_img_w, affined_img_h), absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj
    
    def iou(self, other):
        boxes1 = self.xyxy().data
        boxes2 = other.xyxy().data.to(self.device)

        area1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(dim=-1)
        area2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(dim=-1)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        area_union = area1[:, None] + area2
        area_inter = wh[:, :, 0] * wh[:, :, 1]  # (M, 2)
        iou = area_inter / (area_union - area_inter)

        return iou

    def area(self):
        data = self.xywh().data
        return data[:, 2] * data[:, 3]

    def __str__(self):
        return 'BoxList({}, {}, mode={}, absolute={}, device={})'.format(
            self.data, self.img_size, self.mode, self.absolute, self.device
        )

    def allclose(self, other):
        assert isinstance(other, BoxList)
        assert self.data.shape == other.data.shape
        return torch.allclose(self.to(), other.to(self.device))