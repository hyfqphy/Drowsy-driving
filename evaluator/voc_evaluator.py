
from voc import VOCDetection, VOC_CLASSES
import os
import time
import numpy as np
import pickle
import xml.etree.ElementTree as ET

from utils.box_ops import rescale_bboxes


class VOCAPIEvaluator():
    """ VOC AP Evaluation class """
    def __init__(self, 
                 data_dir, 
                 device,
                 transform,
                 set_type='val',
                 display=False):
        # basic config
        self.data_dir = data_dir
        self.device = device
        self.labelmap = VOC_CLASSES
        self.set_type = set_type
        self.display = display
        self.map = 0.

        # transform
        self.transform = transform

        # path
        self.devkit_path = os.path.join(data_dir)
        self.annopath = os.path.join(data_dir,'Annotations', '%s.xml')
        self.imgpath = os.path.join(data_dir,'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(data_dir,'ImageSets', 'Main', set_type+'.txt')
        self.output_dir = self.get_output_dir('det_results/eval/Drowsy-driving_eval/', self.set_type)

        # dataset
        self.dataset = VOCDetection(
            data_dir=data_dir, 
            image_sets=[('val')],
            is_train=False)
        

    def evaluate(self, net):
        net.eval()
        num_images = len(self.dataset)

        self.all_boxes = [[[] for _ in range(num_images)]
                        for _ in range(len(self.labelmap))]
        # 建了一个复杂的列表结构all_boxes三维列表
        # 第一维对应于类别（基于labelmap的长度），第二维对应于每个类别中的图像（基于图像数量num_images）
        # 每个元素是一个空列表，用于存储检测结果

        det_file = os.path.join(self.output_dir, 'detections.pkl')
        # 定义一个检测结果文件的路径，用于保存检测结果

        for i in range(num_images): # 对数据集中的每个图像进行处理
            img, _ = self.dataset.pull_image(i)
            # 从数据集中提取第i个图像
            orig_h, orig_w = img.shape[:2]
            # 获取图像的原始高度和宽度

            x, _, deltas = self.transform(img)
            # 对图像img应用变换函数transform
            x = x.unsqueeze(0).to(self.device) / 255.
            # 对处理后的图像x进行额外的处理，增加一个批处理维度、转移到GPU，并将像素值标准化到[0,1]范围内

            # forward
            t0 = time.time()
            # 记录前向传播开始的时间
            bboxes, scores, labels = net(x)
            # 将处理过的图像x通过神经网络net进行前向传播，得到边界框（bboxes）和标签（labels）
            detect_time = time.time() - t0
            # 计算检测所用时间

            origin_img_size = [orig_h, orig_w]
            #  原始图像尺寸
            cur_img_size = [*x.shape[-2:]]
            # 当前处理后的图像尺寸
            bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)
            # 用rescale_bboxes函数调整边界框的尺寸，以匹配原始图像的尺寸

            # 处理每个类别的检测结果
            for j in range(len(self.labelmap)): # 遍历每个类别
                inds = np.where(labels == j)[0]
                # 找出属于当前类别j的检测结果的索引
                if len(inds) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    # 如果当前类别没有检测到任何对象，则为该类别的当前图像分配一个空的边界框数组
                    continue
                c_bboxes = bboxes[inds]
                # 从检测结果中提取当前类别j的边界框
                c_scores = scores[inds]
                # 提取对应的得分
                c_dets = np.hstack((c_bboxes,c_scores[:, np.newaxis])).astype(np.float32,copy=False)
                # 将边界框和得分水平堆叠，形成一个检测结果数组，每个检测结果包含边界框坐标和对应得分
                self.all_boxes[j][i] = c_dets
                # 将当前图像的当前类别的检测结果存储在all_boxes中

        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)
            # 将检测结果all_boxes保存到先前打开的文件中
            # 使用pickle.HIGHEST_PROTOCOL以最高协议版本进行序列化，以提高存储效率

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes)

        print('Mean AP: ', self.map)
  

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        # 使用ElementTree模块解析XML文件
        objects = []
        # 初始化一个列表来存储解析出的对象
        for obj in tree.findall('object'):
            obj_struct = {} # 创建一个字典来存储对象的信息
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            # 从bndbox元素中提取边界框坐标，这包括xmin、ymin、xmax、ymax，并将它们转换为整数格式
            objects.append(obj_struct)
            # 将解析出的对象信息添加到objects列表中

        return objects


    def get_output_dir(self, name, phase):
        # 用于获取或创建用于存放output的目录
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir


    def get_voc_results_file_template(self, cls):
        # 用于获取VOC结果文件的路径模板
        filename = 'det_' + self.set_type + '_%s.txt' % (cls)
        filedir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes):
        # 用于将检测结果写入文件
        for cls_ind, cls in enumerate(self.labelmap):
            # 遍历所有类别
            if self.display:
                print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids):
                    # 遍历数据集中的所有图像
                    dets = all_boxes[cls_ind][im_ind]

                    # 获取当前类别cls_ind和当前图像im_ind的检测结果
                    if dets == []:
                        continue
                    # 如果没有检测到当前类别的任何对象，则继续处理下一张图像

                    for k in range(dets.shape[0]):
                        # 遍历当前图像中的所有检测结果
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))
                        # 将检测结果写入文件，格式包括图像索引、检测得分和边界框坐标


    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        #  设置一个用于存储注释缓存的目录
        aps = []
        # 初始化一个列表，用于存储每个类别的平均精度
        use_07_metric = use_07
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        for i, cls in enumerate(self.labelmap): # 遍历每个类别
            filename = self.get_voc_results_file_template(cls)
            #  获取特定类别的结果文件的路径
            rec, prec, ap = self.voc_eval(detpath=filename, 
                                          classname=cls, 
                                          cachedir=cachedir, 
                                          ovthresh=0.5, 
                                          use_07_metric=use_07_metric
                                        )
            # 调用voc_eval方法计算该类别的召回率（recall）、精确率（precision）和平均精度（AP）
            aps += [ap]
            # 将计算出的AP值添加到aps列表中
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            # 打开一个文件用于写入，文件名包含类别名和'_pr.pkl'，表示这是一个包含精确率和召回率的Pickle文件

        if self.display:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')
        else:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))


    def voc_ap(self, rec, prec, use_07_metric=True):
        """
        定义一个名为voc_ap的方法，用于计算平均精度（AP）
        这个方法接受召回率（rec）、精确率（prec）作为输入
        """
        ap = 0. # 初始化AP值
        for t in np.arange(0., 1.1, 0.1):
            # 使用11点插值方法计算AP，遍历召回率的11个点
            if np.sum(rec >= t) == 0:
                p = 0
                # 如果没有召回率大于或等于t的点，精确率p设置为0
            else:
                p = np.max(prec[rec >= t])
                # 在召回率大于或等于t的点上找到最大精确率
            ap = ap + p / 11.
        return ap


    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')

        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
            # 打开由self.imgsetpath指定的文件，这个文件包含了数据集中所有图像的列表
        imagenames = [x.strip() for x in lines]
        # 从文件中读取每一行，去除两端的空白字符，并将每个图像的名称存储在列表imagenames中
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            # 创建一个空字典recs来存储注释
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
                # 遍历每个图像名称，使用self.parse_rec方法读取对应图像的注释，并将结果存储在recs中
                if i % 100 == 0 and self.display:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            if self.display:
                print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        class_recs = {}
        # 初始化字典class_recs，存储每个图像中特定类别的真实对象信息
        npos = 0
        # 初始化计数器npos用于记录非难识别（non-difficult）对象的数量
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            # 从recs（包含所有图像的注释信息）中筛选出当前图像中属于特定类别（classname）的对象，存储在列表R中
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
            # 从R中提取边界框（bbox）和难易程度（difficult）信息
            det = [False] * len(R)
            # 创建一个与R等长的列表det，初始值均为False，用于标记每个对象是否已被检测到
            npos = npos + sum(~difficult)
            # 更新npos，增加当前图像中非难识别对象的数量
            class_recs[imagename] = {'bbox': bbox,'difficult': difficult,'det': det}
            # 在class_recs字典中为当前图像创建一个记录，包含边界框、难易程度和检测状态

        detfile = detpath.format(classname)
        # 构建检测结果文件的路径detfile
        with open(detfile, 'r') as f:
            lines = f.readlines()
            # 从detfile中读取检测结果，每行代表一个检测
        if any(lines) == 1: # 如果文件中有数据
            splitlines = [x.strip().split(' ') for x in lines]
            # 将每行分割并提取图像ID、置信度和边界框坐标
            image_ids = [x[0] for x in splitlines]

            confidence = np.array([float(x[1]) for x in splitlines])
            # 按置信度对检测结果进行排序
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
            # 更新排序后的边界框（BB）和图像ID

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            # 按置信度降序排序检测结果的索引
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]
            # 根据排序后的索引重新排列边界框和图像ID
            #print(image_ids)
            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            # 初始化两个数组tp（真阳性）和fp（假阳性），长度为检测结果数量
            for d,k in enumerate(image_ids):
                R = class_recs[k]
                # 获取该检测对应图像的真实标注数据R
                bb = BB[d, :].astype(float)
                # 计算当前检测（边界框bb）与真实边界框的重叠（IoU）
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()

if __name__ == '__main__':
    pass