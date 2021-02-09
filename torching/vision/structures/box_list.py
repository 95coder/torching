import torch


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
            return BoxList(data, self.img_size, 'xywh')
        return self.clone()

    def xywh(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.data.split(1, dim=-1)
            data = torch.cat([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)
            return BoxList(data, self.img_size, 'xywh')
        return self.clone()

    def absolute_coords(self):
        if self.mode == 'xyxy' and not self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] *= self.img_size[0]
            data[:, [2, 3]] *= self.img_size[1]
            return BoxList(data, self.img_size, absolute=True, device=self.device)
        elif self.mode == 'xywh' and not self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] *= self.img_size[0]
            data[:, [2, 3]] *= self.img_size[1]
            return BoxList(data, self.img_size, absolute=True, device=self.device)

    def percent_coords(self):
        if self.mode == 'xyxy' and self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] /= self.img_size[0]
            data[:, [2, 3]] /= self.img_size[1]
            return BoxList(data, self.img_size, absolute=False, device=self.device)
        elif self.mode == 'xywh' and self.absolute:
            data = self.data.clone()
            data[:, [0, 1]] /= self.img_size[0]
            data[:, [2, 3]] /= self.img_size[1]
            return BoxList(data, self.img_size, absolute=False, device=self.device)

    def clone(self):
        return self.create_like(self)

    def to(self, device=None):
        if device != self.data.device:
            return self.data.to(device)
        return self.data

    def crop(self, region):
        origin_xmin, origin_ymin, origin_xmax, origin_ymax = self.xyxy().data.split(1, dim=-1)

        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = region
            w, h = xmax - xmin, ymax - ymin
        else:
            xmin, ymin = region[:2]
            xmax, ymax = region[:2] - region[2:]
            w, h = region[2:]

        data = torch.cat([(origin_xmin - xmin).clamp(min=0, max=w),
                          (origin_ymin - ymin).clamp(min=0, max=h),
                          (xmax - origin_xmax).clamp(min=0, max=w),
                          (ymax - origin_ymax).clamp(min=0, max=h)], dim=-1)

        obj = BoxList(data, (w, h), absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

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

    def rotate(self):
        pass

    def translate(self, x, y):
        data = self.xyxy().data

        data[:, [0, 1]].add_(x).clamp_(0, self.img_size[0])
        data[:, [2, 3]].add_(y).clamp_(0, self.img_size[1])

        obj = BoxList(data, self.img_size, absolute=self.absolute)
        if self.mode == 'xywh':
            obj = obj.xywh()
        return obj

    def affine(self):
        pass

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
        return 'BoxList({}, {}, mode={}, absolute={} device={})'.format(
            self.data, self.img_size, self.mode, self.absolute, self.device
        )

    def allclose(self, other):
        assert self.data.shape == other.data.shape
        return torch.allclose(self.to(), other.to(self.device))
