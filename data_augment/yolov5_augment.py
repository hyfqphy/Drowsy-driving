import random
import cv2
import math
import numpy as np
import torch


# ------------------------- 基础数据增强 -------------------------
# 空间变换
def random_perspective(image,
                       targets=(),
                       degrees=10, # 旋转的最大角度
                       translate=.1, # 平移的最大比例
                       scale=[0.1, 2.0], # 缩放比例的范围
                       shear=10, # 剪切的最大角度
                       perspective=0.0, # 透视变换的强度
                       border=(0, 0)): # 图像边缘增加的像素数
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    # 对图像执行随机透视变换，并相应地调整图像中目标的坐标

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # 初始化中心矩阵 C 用于后续的变换
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # 创建一个透视变换矩阵，用于图像的透视变换
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # 随机生成旋转角度和缩放比例
    # 创建一个旋转和缩放矩阵
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # 旋转变换
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale) # 缩放变换
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    # 这是OpenCV库中的一个函数，用于生成2D旋转的仿射变换矩阵。a是一个变量，代表随机选择的旋转角度
    # center = (0, 0): 旋转的中心点，在这里被设置为原点(0, 0)。这意味着图像将围绕其左上角旋转
    # scale=s: 缩放因子，用于在旋转的同时进行缩放
    # cv2.getRotationMatrix2D 返回一个2x3的矩阵，它可以被用来与图像坐标进行乘法运算，从而实现旋转和缩放效果

    # 剪切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # 平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # 组合变换矩阵M，由前几个变换相乘得到
    M = T @ S @ R @ P @ C  # 这里的@为矩阵乘法
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        '''
        这个条件判断用来检查是否需要对图像进行变换
        如果边框大小 (border) 不是零，或者变换矩阵 M 不是单位矩阵（意味着至少有一种变换需要应用），那么就需要进行图像变换
        '''
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
            # 通过 M 矩阵对图像进行透视变换，dsize 指定了输出图像的大小，borderValue 指定了边框的颜色（这里是灰色）
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            # 里使用了 M 矩阵的前两行来进行仿射变换，同样，dsize 和 borderValue 指定了输出图像的大小和边框颜色

    '''
    下面这段代码是对图像中目标的边界框（targets）进行变换的过程
    当对图像应用仿射或透视变换时，图像中的目标（例如物体检测中的边界框）也需要相应地进行变换
    '''
    n = len(targets) #获取目标（边界框）的数量
    if n:
        xy = np.ones((n * 4, 3))
        # 初始化一个坐标数组，每个目标（边界框）有四个角点，每个点是一个(x, y, 1)格式的坐标

        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        #将目标边界框的坐标填充到 xy 数组中，坐标被重新排列以形成四个角点：左上、右下、左下、右上
        xy = xy @ M.T  # 使用组合变换矩阵 M 对每个点进行变换
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)
        # 如果应用了透视变换，则对坐标进行相应的归一化处理；如果是仿射变换，则保持原样

        # 计算新的边界框坐标:
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        # 对于每个边界框，找到变换后四个角点的最小x、最小y、最大x、最大y坐标，构成变换后的新边界框
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # 剪切边界框
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        targets[:, 1:5] = new
        #将计算出的新边界框坐标更新到原始 targets 数组中

    return image, targets

# 用于通过随机改变图像的色调（Hue）、饱和度（Saturation）和亮度（Value）来增强图像
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    '''
     为色调（H）、饱和度（S）和亮度（V）生成三个随机增益
     这些增益值是在 [-1, 1] 范围内随机生成的，然后分别乘以给定的 hgain、sgain 和 vgain，最后加上1
     这意味着增益可以在 [0, 2] 的范围内变化
    '''
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # 将图像从BGR颜色空间转换到HSV颜色空间，并将色调（H）、饱和度（S）和亮度（V）通道分离开来
    dtype = img.dtype  # uint8

    # 创建查找表
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    # 创建色调的查找表，由于HSV色调的范围是[0, 180)，因此将结果取模180，并转换回原始数据类型
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    # 为饱和度和亮度创建查找表，通过乘以相应的增益，并确保值在0到255之间

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    # 使用查找表更新每个通道的值，并将它们合并回一个单一的HSV图像
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    # 将增强后的HSV图像转换回BGR颜色空间，并将结果存储在原始图像变量img中


# ------------------------- Strong augmentations -------------------------
## YOLOv5-Mosaic
def yolov5_mosaic_augment(image_list, target_list, img_size, affine_params, is_train=False):
    # 马赛克增强，将四张图像组合成一张大图像的技术，同时调整各图像中目标的标注

    assert len(image_list) == 4
    # 确保传入的图像列表 image_list 中有四张图

    mosaic_img = np.ones([img_size*2, img_size*2, image_list[0].shape[2]], dtype=np.uint8) * 114
    # 创建一个初始化为灰色（114）的空白图像，其大小是单张输入图像大小的两倍

    yc, xc = [int(random.uniform(-x, 2*img_size + x)) for x in [-img_size // 2, -img_size // 2]]
    # 随机确定四张图像组合后的中心点

    mosaic_bboxes = []
    mosaic_labels = []
    # 分别用于存储组合图像中的边界框和标签

    for i in range(4):
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]
        # 遍历四张图像，并根据其在马赛克图像中的位置（左上、右上、左下、右下）调整大小并放置

        orig_h, orig_w, _ = img_i.shape

        # resize
        r = img_size / max(orig_h, orig_w)
        # 计算图像缩放比例，以确保图像适应预定的单图像尺寸 img_size
        if r != 1: 
            interp = cv2.INTER_LINEAR if (is_train or r > 1) else cv2.INTER_AREA
            # 选择合适的插值方法。如果是训练阶段或者需要放大图像（r > 1），则使用线性插值；否则，使用面积插值
            img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)), interpolation=interp)
            #根据计算出的比例调整图像大小
        h, w, _ = img_i.shape

        # 根据图像的索引 i（0到3），计算图像在马赛克图像中的位置。四个位置分别是左上角、右上角、左下角和右下角
        if i == 0:  # 左上
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (大图像坐标)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (小图像坐标)
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
        # 将调整后的图像 img_i 的相应部分放置到马赛克图像的计算位置上
        padw = x1a - x1b
        padh = y1a - y1b
        # 计算水平和垂直方向上的偏移量

        # labels
        bboxes_i_ = bboxes_i.copy()
        # 创建当前图像边界框的副本，以便修改而不影响原始数据

        if len(bboxes_i) > 0:
            # 如果当前图像存在有效的边界框，则进行处理
            # 边界框坐标调整的公式基本上是：(缩放后的尺寸 * 原始坐标 / 原始尺寸 + 偏移量)
            # 原始坐标指的是在原始图像中的边界框坐标，而调整后的坐标是在马赛克图像中的坐标
            bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padw)
            # 调整边界框的x轴坐标xmin
            bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + padh)
            # 调整边界框的y轴坐标ymin
            bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padw)
            # 调整边界框的x轴坐标xmax
            bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + padh)
            # 调整边界框的y轴坐标ymax
            mosaic_bboxes.append(bboxes_i_)
            mosaic_labels.append(labels_i)
            # 将调整后的边界框和标签追加到 mosaic_bboxes 和 mosaic_labels

    if len(mosaic_bboxes) == 0: # 合并所有图像的边界框和标签
        mosaic_bboxes = np.array([]).reshape(-1, 4)
        mosaic_labels = np.array([]).reshape(-1)
        # 如果 mosaic_bboxes 中没有边界框，则创建一个空的边界框数组
    else:
        mosaic_bboxes = np.concatenate(mosaic_bboxes)
        mosaic_labels = np.concatenate(mosaic_labels)
        # 否则，使用 np.concatenate 将所有图像的边界框和标签合并成一个数组

    # clip
    mosaic_bboxes = mosaic_bboxes.clip(0, img_size * 2)
    # 确保边界框坐标不超出马赛克图像的范围，这里 clip 函数用于将坐标限制在[0, 2 * img_size]的范围内

    mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
    # 将mosaic_labels和mosaic_bboxes沿着最后一个轴（axis = -1）连接起来
    # mosaic_labels[..., None] 将标签从一维数组转换为二维数组，使其可以与二维的 mosaic_bboxes 数组合并
    # 最终，mosaic_targets 数组的每一行将包含一个目标的标签和对应的边界框坐标

    mosaic_img, mosaic_targets = random_perspective(
        mosaic_img,
        mosaic_targets,
        affine_params['degrees'],
        translate=affine_params['translate'],
        scale=affine_params['scale'],
        shear=affine_params['shear'],
        perspective=affine_params['perspective'],
        border=[-img_size//2, -img_size//2]
        )
    # 使用 random_perspective 函数对马赛克图像和目标进行随机透视变换
    # 这个函数利用了提供的 affine_params 参数（包括旋转角度、平移、缩放比例、剪切和透视变换的强度）以及边框参数来执行变换

    # target
    mosaic_target = {   # 创建一个字典，包含处理后的目标信息
        "boxes": mosaic_targets[..., 1:],   # 提取变换后的边界框坐标
        "labels": mosaic_targets[..., 0],   # 提取对应的标签
        "orig_size": [img_size, img_size]   # 记录原始图像的尺寸
    }

    return mosaic_img, mosaic_target

## YOLOv5-Mixup
def yolov5_mixup_augment(origin_image, origin_target, new_image, new_target):
# Mixup增强，将两幅图像结合成一幅新的训练样本
    if origin_image.shape[:2] != new_image.shape[:2]:
        # 检查origin_image和new_image的高度和宽度（前两个维度）是否不同，如果它们不同，需要调整图像的大小以匹配
        img_size = max(new_image.shape[:2])
        # 找出new_image的较大维度（高度或宽度），这个尺寸将用于缩放origin_image
        orig_h, orig_w = origin_image.shape[:2]
        # 提取origin_image的高度（orig_h）和宽度（orig_w）
        scale_ratio = img_size / max(orig_h, orig_w)
        # 计算需要缩放origin_image的比例，通过将new_image的较大维度除以origin_image的较大维度得出

        if scale_ratio != 1: # 检查是否需要缩放
            interp = cv2.INTER_LINEAR if scale_ratio > 1 else cv2.INTER_AREA
            # 当需要放大图像（scale_ratio > 1）时使用cv2.INTER_LINEAR插值，而需要缩小图像时使用cv2.INTER_AREA插值
            resize_size = (int(orig_w * scale_ratio), int(orig_h * scale_ratio))
            origin_image = cv2.resize(origin_image, resize_size, interpolation=interp)
            # 通过将origin_image的原始宽度和高度乘以scale_ratio来计算新的大小
            # 然后，使用所选的插值方法通过cv2.resize将origin_image调整为这个新的大小

        # pad new image
        pad_origin_image = np.ones([img_size, img_size, origin_image.shape[2]], dtype=np.uint8) * 114
        pad_origin_image[:resize_size[1], :resize_size[0]] = origin_image
        origin_image = pad_origin_image.copy()
        del pad_origin_image

    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    mixup_image = r * origin_image.astype(np.float32) + \
                  (1.0 - r)* new_image.astype(np.float32)
    mixup_image = mixup_image.astype(np.uint8)
    
    cls_labels = new_target["labels"].copy()
    box_labels = new_target["boxes"].copy()

    mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
    mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)

    mixup_target = {
        "boxes": mixup_bboxes,
        "labels": mixup_labels,
        'orig_size': mixup_image.shape[:2]
    }
    
    return mixup_image, mixup_target

# ------------------------- Preprocessers -------------------------
## YOLOv5-style Transform for Train
class YOLOv5Augmentation(object):
    def __init__(self, 
                 img_size=640,
                 trans_config=None):
        self.trans_config = trans_config
        self.img_size = img_size


    def __call__(self, image, target, mosaic=False):
        # resize
        img_h0, img_w0 = image.shape[:2]

        r = self.img_size / max(img_h0, img_w0)
        if r != 1: 
            interp = cv2.INTER_LINEAR
            new_shape = (int(round(img_w0 * r)), int(round(img_h0 * r)))
            img = cv2.resize(image, new_shape, interpolation=interp)
        else:
            img = image

        img_h, img_w = img.shape[:2]

        # hsv augment
        augment_hsv(img, hgain=self.trans_config['hsv_h'], 
                    sgain=self.trans_config['hsv_s'], 
                    vgain=self.trans_config['hsv_v'])
        
        if not mosaic:
            # rescale bbox
            boxes_ = target["boxes"].copy()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # spatial augment
            target_ = np.concatenate(
                (target['labels'][..., None], target['boxes']), axis=-1)
            img, target_ = random_perspective(
                img, target_,
                degrees=self.trans_config['degrees'],
                translate=self.trans_config['translate'],
                scale=self.trans_config['scale'],
                shear=self.trans_config['shear'],
                perspective=self.trans_config['perspective']
                )
            target['boxes'] = target_[..., 1:]
            target['labels'] = target_[..., 0]
        
        # random flip
        if random.random() < 0.5:
            w = img.shape[1]
            img = np.fliplr(img).copy()
            boxes = target['boxes'].copy()
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            target["boxes"] = boxes

        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        assert max(img_h0, img_w0) <= self.img_size

        pad_image = torch.ones([img_tensor.size(0), self.img_size, self.img_size]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor
        dh = self.img_size - img_h0
        dw = self.img_size - img_w0

        return pad_image, target, [dw, dh]

## YOLOv5-style Transform for Eval
class YOLOv5BaseTransform(object):
    def __init__(self, img_size=640, max_stride=32):
        self.img_size = img_size
        self.max_stride = max_stride


    def __call__(self, image, target=None, mosaic=False):
        # resize
        img_h0, img_w0 = image.shape[:2]

        r = self.img_size / max(img_h0, img_w0)
        # r = min(r, 1.0) # only scale down, do not scale up (for better val mAP)
        if r != 1: 
            new_shape = (int(round(img_w0 * r)), int(round(img_h0 * r)))
            img = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        else:
            img = image

        img_h, img_w = img.shape[:2]

        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        # rescale bboxes
        if target is not None:
            # rescale bbox
            boxes_ = target["boxes"].copy()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # to tensor
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        dh = img_h0 % self.max_stride
        dw = img_w0 % self.max_stride
        dh = dh if dh == 0 else self.max_stride - dh
        dw = dw if dw == 0 else self.max_stride - dw
        
        pad_img_h = img_h0 + dh
        pad_img_w = img_w0 + dw
        pad_image = torch.ones([img_tensor.size(0), pad_img_h, pad_img_w]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor

        return pad_image, target, [dw, dh]
