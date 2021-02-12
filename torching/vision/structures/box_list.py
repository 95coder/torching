import torch
import numpy as np
import math

# from torching.vision.utils.img_process import get_affine_matrix


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

    def clone(self):
        return self.create_like(self)

    def to(self, device=None):
        if device != self.data.device:
            return self.data.to(device=device)
        return self.data

    def to_tensor(self, dtype=None):
        if dtype is None:
            return self.data
        return self.data.type(dtype=dtype)

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

    def split_into_pts(self):
        data = self.xyxy().data
        xmin, ymin, xmax, ymax = data.split(1, dim=-1)
        return (torch.cat([xmin, ymin], dim=-1), 
                torch.cat([xmax, ymin], dim=-1), 
                torch.cat([xmax, ymax], dim=-1), 
                torch.cat([xmin, ymax], dim=-1))

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

    def rotate90(self, center=None):
        return self.rotate(90, center)

    def rotate(self, angle, center=None):
        return self.affine(angle)

    def affine(self, angle, center=None, translate=None, scale=1, shear=None):
        def transform_points(points, M):
            if points.ndim == 1:
                points = points.unsqueeze(dim=0)
            return points.mm(M[:2, :2].T) + M[:2, -1].T

        def get_affine_matrix(center, angle, translate, scale, shear):
            """
            Notes: In the image pixel coordinate, the transformation is:
                x_ = cos(theta) * x - sin(theta) * y
                y_ = sin(theta) * x + cos(theta) * y
            """
            angle = angle % 360
            angle = math.radians(angle)
            M = [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0]]
            M = torch.as_tensor(M, dtype=torch.float32, device=self.device)
            M.mul_(scale)

            affined_center = transform_points(center + translate, M)
            M[0, 2] = center[0] - affined_center[0, 0]
            M[1, 2] = center[1] - affined_center[0, 1]
            return M

        def find_bounding_box_for_contour_pts(points):
            xmin = points[:, 0].min()
            xmax = points[:, 0].max()
            ymin = points[:, 1].min()
            ymax = points[:, 1].max()
            box = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=points.device)
            return box

        def find_bounding_boxes_for_boxes_in_pts(boxes_in_pts):
            xmin, _ = boxes_in_pts[:, 0::2].min(dim=-1, keepdim=True)
            xmax, _ = boxes_in_pts[:, 0::2].max(dim=-1, keepdim=True)
            ymin, _ = boxes_in_pts[:, 1::2].min(dim=-1, keepdim=True)
            ymax, _ = boxes_in_pts[:, 1::2].max(dim=-1, keepdim=True)

            boxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
            return boxes
        
        img_w, img_h = self.img_size

        # Get the image grid corners.
        if self.absolute:
            img_corners = torch.as_tensor([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]],
                                          dtype=torch.float32, device=self.device)
        else:
            img_corners = torch.as_tensor([[0, 0], [1, 0], [1, 1], [0, 1]],
                                          dtype=torch.float32, device=self.device)

        if center is None:
            center = (0.5, 0.5)
        center = tuple(center)
        center = torch.as_tensor([center[0] * img_w, center[1] * img_h], dtype=torch.float32, device=self.device)

        if translate is None:
            translate = (0, 0)
        translate = tuple(translate)
        translate = torch.as_tensor([translate[0] * img_w, translate[1] * img_h], dtype=torch.float32, device=self.device)

        # Make a affine transformation matrix.
        M = get_affine_matrix(center, angle, translate, scale, shear)

        # Get the transformed img grid.
        # print('\nimg_corners:\n', img_corners)
        affined_img_corners = transform_points(img_corners, M)
        # print('\naffined_img_corners:\n', affined_img_corners)
        affined_img_grid = find_bounding_box_for_contour_pts(affined_img_corners)
        # print('\naffined_img_grid:\n', affined_img_grid)

        # Do transformation for box corner points, then find the bounding boxes of the transformed points.
        boxes_in_pts = self.split_into_pts()
        affined_boxes_in_pts = []
        for boxes_pts in boxes_in_pts:
            affined_boxes_in_pts.append(transform_points(boxes_pts, M))
        affined_boxes_in_pts = torch.cat(affined_boxes_in_pts, dim=-1)
        affined_boxes = find_bounding_boxes_for_boxes_in_pts(affined_boxes_in_pts)

        # Translate the boxes to new image grid.
        # print('\naffined_boxes:\n', affined_boxes)
        # affined_boxes[:, [0, 2]] -= affined_img_grid[0]
        # affined_boxes[:, [1, 3]] -= affined_img_grid[1]
        # print('\naffined_boxes:\n', affined_boxes)
        affined_img_w = (affined_img_grid[2] - affined_img_grid[0]).int().item()
        affined_img_h = (affined_img_grid[3] - affined_img_grid[1]).int().item()

        obj = BoxList(affined_boxes, (affined_img_w, affined_img_h), absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj
    
    def iou(self, other):
        boxes1 = self.xyxy().data
        boxes2 = other.xyxy().to(self.device).data

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
        return torch.allclose(self.data, other.to(self.device).data)