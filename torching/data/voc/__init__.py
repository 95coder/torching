import torch
import os
from xml.etree import ElementTree as ET
from PIL import Image
from torchvision import datasets, transforms


class VOCDetectionDataset(torch.utils.data.Dataset):
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
        img_height = int(anno.find('size').find('height').text)
        img_width = int(anno.find('size').find('width').text)

        targets = []
        for obj in anno.iter('object'):
            name = obj.find('name').text.strip().lower()
            bndbox = obj.find('bndbox')
            xmin = (int(bndbox.find('xmin').text) - 1) / img_width
            ymin = (int(bndbox.find('ymin').text) - 1) / img_height
            xmax = (int(bndbox.find('xmax').text) - 1) / img_width
            ymax = (int(bndbox.find('ymax').text) - 1) / img_height
            bbox = [xmin, ymin, xmax, ymax]  # in xyxy format
            label = self.CLASSES.index(name)
            targets.append(bbox + [label])
        
        img = self.transfrom(img)
        target = torch.tensor(targets)
        return img, target

    def __len__(self):
        return len(self._idxs)


def data_collate_fn(batch):
    imgs = []
    targets = []
    for sample in batch:
        img, target = sample
        targets.append(target)
        imgs.append(img)
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    dataset = VOCDetectionDataset('/data/datasets/images/VOC/VOCdevkit/VOC2007', 'train', transform)
    img, target = dataset[0]
    print(img.shape)
    print(target.shape)