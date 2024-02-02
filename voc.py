"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import random
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os

try:
    from .data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment
except:
    from data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment

VOC_CLASSES = [ 'open_eye','closed_eye','closed_mouth','open_mouth']
# 疲劳驾驶的VOC数据集特征有4类：睁眼、闭眼、张嘴、闭嘴
class VOCAnnotationTransform(object):
    """
    读取数据类别，并根据类别计算出voc_classes中的类别序号，序号从0开始
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        # zip函数将这两个序列组合成一个元组序列，其中每个元组包含一个类名和相应的索引
        # self.class_to_ind成为一个字典，可以通过类名来查找它在VOC_CLASSES中的索引
        self.keep_difficult = keep_difficult

    def __call__(self, target):

        res = []
        for obj in target.iter('object'):# 遍历target中的所有object元素。在VOC数据集的XML标注文件中，每个object元素代表图像中的一个对象

            difficult = int(obj.find('difficult').text) == 1
            # 检查对象是否被标记为“难以识别”（difficult）。在VOC数据集中，如果一个对象被认为很难识别，它的difficult标签会被设置为1

            if not self.keep_difficult and difficult:
                continue
            # 如果这个对象被标记为难以识别，并且类的属性keep_difficult为False，则跳过这个对象，不处理

            name = obj.find('name').text.lower().strip()
            # 获取对象的类名，并将其转换为小写并去除两端的空格

            bbox = obj.find('bndbox')
            # 提取对象的边界框（bounding box）信息，在VOC数据集中，边界框用四个坐标点表示：xmin, ymin, xmax, ymax
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            bndbox = []
            for i, pt in enumerate(pts):  #遍历边界框的四个坐标点
                cur_pt = int(bbox.find(pt).text) - 1
                # 提取并转换每个坐标点的值。坐标点在XML文件中是从1开始的，但在许多处理中需要从0开始，因此这里减去1

                bndbox.append(cur_pt)
                # 将处理后的坐标点添加到bndbox列表中
            label_idx = self.class_to_ind[name]
            # 将类名转换为其对应的索引值

            bndbox.append(label_idx)
            # 将类的索引值添加到边界框列表的末尾

            res += [bndbox]  # [x1, y1, x2, y2, label_ind]
            # 将完整的边界框（包括坐标和类索引）添加到结果列表中

        return res


class VOCDetection(data.Dataset):
    """
    self.root: 数据集路径
    self.image_set: 数据集的划分（设置为trainval时，读取VOC的trainval集的图像和标签）
    self.transformer: 数据预处理函数
    """

    def __init__(self, 
                 img_size=640,
                 data_dir=None,
                 image_sets=[('trainval')],
                 trans_config=None,
                 transform=None,
                 is_train=False,
                 load_cache=False
                 ):
        self.root = data_dir
        self.img_size = img_size
        #self.image_set = image_sets
        self.target_transform = VOCAnnotationTransform()
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.is_train = is_train
        self.load_cache = load_cache
        for name in image_sets:
            '''
            从VOC数据集的ImageSets/Main/目录下的文本文件中读取图像ID，
            然后将这些ID连同数据集的根路径一起保存在self.ids列表中。
            这样的操作通常是为了后续能够方便地根据图像ID找到对应的图像和标注文件
            '''
            rootpath=self.root
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        # augmentation
        self.transform = transform
        self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
        self.mixup_prob = trans_config['mixup_prob'] if trans_config else 0.0
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')

        # load cache data
        if load_cache:
            self._load_cache()


    def __getitem__(self, index):
        image, target, deltas = self.pull_item(index)
        return image, target, deltas


    def __len__(self):
        return len(self.ids)


    def _load_cache(self):
        # load image cache
        self.cached_images = []
        self.cached_targets = []
        dataset_size = len(self.ids) # 获取数据集的大小，即图像的数量

        print('loading data into memory ...')
        for i in range(dataset_size):
            if i % 5000 == 0:
                print("[{} / {}]".format(i, dataset_size)) # 每处理5000张图像时，打印当前处理进度
            # load an image
            image, image_id = self.pull_image(i)  # 读取图像及其id
            orig_h, orig_w, _ = image.shape  # 获取图像的size

            # 调整图像大小
            r = self.img_size / max(orig_h, orig_w) # 计算调整比例r，以保持图像的宽度或高度与self.img_size相同
            if r != 1:
                # 如果r不等于1（即需要调整大小），使用OpenCV的cv2.resize函数调整图像大小
                interp = cv2.INTER_LINEAR
                new_size = (int(orig_w * r), int(orig_h * r))
                image = cv2.resize(image, new_size, interpolation=interp)

            img_h, img_w = image.shape[:2]
            # 记录调整后的图像尺寸
            self.cached_images.append(image)

            '''
            解析XML文件。self._annopath % image_id是用于格式化字符串的表达式，
            用于生成特定图像的标注文件（annotation file）的路径，image_id是该图像的唯一标识符
            '''
            anno = ET.parse(self._annopath % image_id).getroot()
            anno = self.target_transform(anno)
            # target_transform用于数据增强

            anno = np.array(anno).reshape(-1, 5)
            boxes = anno[:, :4] # 从anno数组中提取前边界框的坐标（xmin, ymin, xmax, ymax）
            labels = anno[:, 4] # 提取anno数组中的第5个元素，这通常是与边界框相关联的类别标签

            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h
            # 两行代码用于调整边界框的坐标，以适应图像尺寸的变化。如果图像被缩放，则边界框的坐标也需要相应地缩放。

            self.cached_targets.append({"boxes": boxes, "labels": labels})
        

    def load_image_target(self, index): # 读取图像和标签
        if self.load_cache:
            image = self.cached_images[index]
            target = self.cached_targets[index]
            height, width, channels = image.shape
            target["orig_size"] = [height, width]
        else:
            # load an image
            img_id = self.ids[index]
            image = cv2.imread(self._imgpath % img_id)
            height, width, channels = image.shape

            # laod an annotation
            anno = ET.parse(self._annopath % img_id).getroot()
            if self.target_transform is not None:
                anno = self.target_transform(anno)

            # guard against no boxes via resizing
            anno = np.array(anno).reshape(-1, 5)
            target = {
                "boxes": anno[:, :4],
                "labels": anno[:, 4],
                "orig_size": [height, width]
            }
        
        return image, target


    def load_mosaic(self, index):
        # load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        # load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # Mosaic
        if self.trans_config['mosaic_type'] == 'yolov5_mosaic':
            image, target = yolov5_mosaic_augment(
                image_list, target_list, self.img_size, self.trans_config, self.is_train)

        return image, target


    def load_mixup(self, origin_image, origin_target):
        # YOLOv5 type Mixup
        if self.trans_config['mixup_type'] == 'yolov5_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
            image, target = yolov5_mixup_augment(
                origin_image, origin_target, new_image, new_target)
        return image, target
    

    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        # augment
        image, target, deltas = self.transform(image, target, mosaic)

        return image, target, deltas


    def pull_image(self, index):
        '''
        使用OpenCV的 imread 函数加载图像
        self._imgpath % img_id 表示图像的路径，
        这里使用字符串格式化操作 (%) 来替换路径模板中的占位符（如 %s 或 %d）为实际的图像ID或文件名，
        cv2.IMREAD_COLOR 指定以彩色模式读取图像
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    import argparse
    from build import build_transform
    
    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    parser.add_argument('--root', default='./dataset/Drowsy-driving',
                        help='data root')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='load cached data.')
    
    args = parser.parse_args()

    trans_config = {
        'aug_type': 'yolov5',            # 或者改为'ssd'来使用SSD风格的数据增强
        # Basic Augment
        'degrees': 0.0,                  # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的10.0
        'translate': 0.2,                # 可以修改数值来决定平移图片的程度，
        'scale': [0.1, 2.0],             # 图片尺寸扰动的比例范围
        'shear': 0.0,                    # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的2.0
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Mosaic & Mixup
        'mosaic_prob': 1.0,              # 使用马赛克增强的概率：0～1
        'mixup_prob': 1.0,               # 使用混合增强的概率：0～1
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolox_mixup',     # 或者改为'yolov5_mixup'，使用yolov5风格的混合增强
        'mixup_scale': [0.5, 1.5]
    }
    transform, trans_cfg = build_transform(args, trans_config, 32, args.is_train)

    dataset = VOCDetection(
        img_size=args.img_size,
        data_dir=args.root,
        trans_config=trans_config,
        transform=transform,
        is_train=args.is_train,
        load_cache=args.load_cache
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target, deltas = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 - x1 > 1 and y2 - y1 > 1:
                cls_id = int(label)
                color = class_colors[cls_id]
                # class name
                label = VOC_CLASSES[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        #cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)