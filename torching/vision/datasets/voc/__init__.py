import torch
import os
from xml.etree import ElementTree as ET
from PIL import Image

from torchvision import datasets
from torching.vision.structures import BoxList
from torching.vision.structures import BoxTarget


class VOC2007DetectionDataset(torch.utils.data.Dataset):
    CLASSES = (
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    )

    def __init__(self, data_dir, img_set, transfrom):
        self._imgs_dir = os.path.join(data_dir, 'JPEGImages')
        self._annos_dir = os.path.join(data_dir, 'Annotations')
        self._imgsetidx_file = os.path.join(data_dir, 'ImageSets', 'Main', '%s.txt' % img_set)

        self.transfrom = transfrom

        with open(self._imgsetidx_file, 'r') as f:
            self._idxs = [line.strip() for line in f]

    def __getitem__(self, index):
        img_idx = self._idxs[index]
        img_file = os.path.join(self._imgs_dir, '%s.jpg' % img_idx)
        anno_file = os.path.join(self._annos_dir, '%s.xml' % img_idx)

        img = Image.open(img_file).convert('RGB')  # pytroch的图像输入为chw的RGB图像

        anno = ET.parse(anno_file).getroot()
        img_width = int(anno.find('size').find('width').text)
        img_height = int(anno.find('size').find('height').text)
        img_size = (img_width, img_height)

        boxes = []
        labels = []
        for obj in anno.iter('object'):
            name = obj.find('name').text.strip().lower()
            bndbox = obj.find('bndbox')
            xmin = (int(bndbox.find('xmin').text) - 1) / img_width
            ymin = (int(bndbox.find('ymin').text) - 1) / img_height
            xmax = (int(bndbox.find('xmax').text) - 1) / img_width
            ymax = (int(bndbox.find('ymax').text) - 1) / img_height
            box = [xmin, ymin, xmax, ymax]  # in xyxy format
            label = self.CLASSES.index(name)
            boxes.append(box)
            labels.append(label)

        box_list = BoxList(boxes, img_size)
        img, box_list = self.transfrom(img, box_list)
        target = BoxTarget(box_list, labels).to_tensor()
        return img, target

    def __len__(self):
        return len(self._idxs)

    @staticmethod
    def collate_fn(batch):
        imgs = []
        targets = []
        for sample in batch:
            img, target = sample
            targets.append(target)
            imgs.append(img)
        return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    dataset = VOCDetectionDataset('/data/datasets/images/VOC/VOCdevkit/VOC2007', 'train', transform)
    img, target = dataset[0]
    print(img.shape)
    print(target.shape)