# -*- coding: utf-8 -*-
import sys
import os
import glob
import cv2
import numpy as np
import json
#---below---imgaug module
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage

import io
from labelme import utils
from labelme import PY2
from labelme import QT4
import base64
import os.path as osp

# 新增
import copy
import threading
from threading import Lock,Thread
import time,os

from xml.etree.ElementTree import ElementTree as ET, Element

'''
ticks:
1) picture type : jpg;
2) while augumenting, mask not to go out image shape;
3) maybe some error because data type not correct.
'''
import PIL
import PIL.Image
# from labelme.logger import logger
# 新增
def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        # logger.error("Failed opening image file: {}".format(filename))
        print("Failed opening image file: {}".format(filename))
        return

    # apply orientation to image according to exif
    image_pil = utils.apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if PY2 and QT4:
            format = "PNG"
        elif ext in [".jpg", ".jpeg"]:
            format = "JPEG"
        else:
            format = "PNG"
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('====================')
        print('create path : ', path)
        print('====================')
    return 0


def find_bndbox_bndnodes(tree, path='object/bndbox'):
    return tree.findall(path)

def check_json_file(path):
    for i in path:
        json_path = i[:-3] + 'json'
        if not os.path.exists(json_path):
            print('error')
            print(json_path, ' not exist !!!')
            sys.exit(1)

def check_xml_file(xmlpath, jpgpath_list):
    # now_path = os.path.dirname(path[0])
    for jpgpath in jpgpath_list:
        jpgname = os.path.split(jpgpath)[-1]

        xml_name = jpgname[:-3] + 'xml'
        xml_pathname = os.path.join(xmlpath, xml_name)
        if not os.path.exists(xml_pathname):
            print('error')
            print(xml_pathname, ' not exist !!!')
            sys.exit(1)
 
def read_jsonfile(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
 
 
def save_jsonfile(object, save_path):
    json.dump(object, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
 
 
def get_points_from_json(json_file):
    point_list = []
    shapes = json_file['shapes']
    for i in range(len(shapes)):
        for j in range(len(shapes[i]["points"])):
            point_list.append(shapes[i]["points"][j])
    return point_list

def get_ET_from_xml(xml_file):
    tree = ET()
    tree.parse(xml_file)
    return tree

def get_bndbox_from_ET(tree):
    #return [[x_min, y_min, x_max, y_max],[x_min, y_min, x_max, y_max] ...]
    # and x_min, y_min, x_max, y_max： (type = int)
    root = tree.getroot()
    findstr_list = ['xmin', 'ymin', 'xmax', 'ymax']
    tree.getroot()
    bndbox_list = [bndbox for bndbox in root.iter('bndbox')]
    ls = [[float(bndbox.find(name).text) for name in findstr_list] for bndbox in bndbox_list]
    print(ls)
    return ls

def write_points_to_json(json_file, aug_points):
    k = 0
    new_json = json_file
    shapes = new_json['shapes']
    # new_point = []
    for i in range(len(shapes)):
        for j in range(len(shapes[i]["points"])):
            # 废弃
            # new_point = [aug_points.keypoints[k].x, aug_points.keypoints[k].y]
            # 新增替换
            # point_list = [float(aug_points.keypoints[k].x), float(aug_points.keypoints[k].y)]
            new_point = [float(aug_points.keypoints[k].x), float(aug_points.keypoints[k].y)]
            # new_point.extend(point_list)
            new_json['shapes'][i]["points"][j] = new_point
            k = k + 1
    return new_json


def write_xml(tree, output_path):
    tree.write(output_path, encoding='utf-8', xml_declaration = True)

def check_and_alter_bndbox(etree ,unnormalizedBatch,img_shape, id):
    '''
    Args:
        unnormalizedBatch:存放 images=images, bounding_boxes_aug
        # bounding_boxes_aug 存放着 list:[BoundingBoxOnImage, BoundingBoxOnImage, BoundingBoxOnImage]
        # BoundingBoxOnImage: 两个变量
        # BoundingBoxOnImage: [BoundingBox(x1=p[0],y1=p[1],x2=p[2],y2=p[3]) for p in box_list]
        # shape: img_content.shape
        img_shape:
        id:

    Returns:
        total_flag
        # 判断总体其中是否至少有1个被修改，total_flag = 1时代表有，total_flag = 0便无，-1代表该图片已经没有 boundingbox
    '''


    # img_shape 存放 (长, 宽, 通道数)
    '''
    BoundingBoxOnImage.BoundingBox = BoundingBox
    BoundingBoxOnImage.shape = normalize_shape(shape)

    BoundingBoxes 存放 [BoundingBox, BoundingBox, BoundingBox ...]
    BoundingBox 存放  BoundingBox.x1 与 BoundingBox.y1 、 BoundingBox.x2 与 BoundingBox.y2
    '''

    name_list = ['xmin', 'ymin', 'xmax', 'ymax']
    # bndbox_obj_node_list = find_bndbox_bndnodes(etree)
    # print('bndbox_obj_node_list：',len(bndbox_obj_node_list))
    root = etree.getroot()
    # print(f'type root:{type(root)}')
    bdx_object_list = find_bndbox_bndnodes(etree, 'object')
    print('bdx_object_list：',len(bdx_object_list))
    # 存放图片 width, height, channel 参数
    img_shape_x = img_shape[1]
    img_shape_y = img_shape[0]
    img_shape_channel = img_shape[2]

    BoundingBoxes_list = unnormalizedBatch.bounding_boxes_aug[id].bounding_boxes
    print('BoundingBoxes_list：',len(BoundingBoxes_list))
    # 判断总体其中是否至少有1个被修改，total_flag=1时代表有，total_flag=0便无，-1代表该图片已经没有boundingbox
    total_flag = 0
    no_box_flag = 0
    idx = 0
    for i in range(len(BoundingBoxes_list)):
        # 如果有修改，boundingBox
        box_fix_flag = 0
        box_delete_flag = 0
        # KeypointsOnImage_list[i]：这是keypoint对象，keypoint对象有两个变量keypoint.x与keypoint.y
        boundingBox_x1 = BoundingBoxes_list[idx].x1
        boundingBox_y1 = BoundingBoxes_list[idx].y1
        boundingBox_x2 = BoundingBoxes_list[idx].x2
        boundingBox_y2 = BoundingBoxes_list[idx].y2


        bdx_list = [str(boundingBox_x1), str(boundingBox_y1), str(boundingBox_x2), str(boundingBox_y2)]
        for idxx, name in enumerate(name_list):
            pt = bdx_list[idxx]
            bdx_object_list[idx].find('bndbox').find(name).text = pt

        # 如果 boundingBox_x2 > img_shape_x，那么令 boundingBox_x2 = img_shape_x
        if boundingBox_x2 > img_shape_x:
            boundingBox_x2 = img_shape_x
            box_fix_flag = 1
        # 如果 boundingBox_y2 > img_shape_y，那么令 boundingBox_y2 = img_shape_y
        if boundingBox_y2 > img_shape_y:
            boundingBox_y2 = img_shape_y
            box_fix_flag = 1

        # 如果 boundingBox_x1 < 0，那么令 boundingBox_x1 = 0
        if boundingBox_x1 < 0:
            boundingBox_x1 = 0
            box_fix_flag = 1

        # 如果 boundingBox_y1 < 0，那么令 boundingBox_y1 = 0
        if boundingBox_y1 < 0:
            boundingBox_y1 = 0
            box_fix_flag = 1

        # print(f'x1 = {(boundingBox_x1 <= boundingBox_x2)}')
        # print(f'x2 = {(boundingBox_y1 <= boundingBox_y2)}')
        # print(((boundingBox_x1 = boundingBox_x2) or (boundingBox_y1 = boundingBox_y2)))
        if (boundingBox_x1 >= boundingBox_x2) or (boundingBox_y1 >= boundingBox_y2) == 1:
            print("bbbbbbbbbbbx",boundingBox_x1 , boundingBox_x2 , boundingBox_y1 , boundingBox_y2)
            del BoundingBoxes_list[idx]
            # print(bdx_object_list[idx])
            # print(f'type root:{type(root)}')
            root.remove(bdx_object_list[idx])
            print('检测到增强图片的boundingbox完全飞出图片之外，删除')
            box_delete_flag = -1
            if idx != 0:
                idx -= 1
            # 如果最后一个框被删除，表示已经没有框, 把no_box_flag打到1
            if (i == len(BoundingBoxes_list)) and idx == 0:
                no_box_flag == 1
            continue

        if box_fix_flag == 1 and box_delete_flag != -1:
            total_flag = 1
            BoundingBoxes_list[idx].x1, BoundingBoxes_list[idx].y1 = boundingBox_x1, boundingBox_y1
            BoundingBoxes_list[idx].x2, BoundingBoxes_list[idx].y2 = boundingBox_x2, boundingBox_y2
            bdx_list = [str(BoundingBoxes_list[idx].x1), str(BoundingBoxes_list[idx].y1), str(BoundingBoxes_list[idx].x2), str(BoundingBoxes_list[idx].y2)]
            for idxx, name in enumerate(name_list):
                pt = bdx_list[idxx]
                bdx_object_list[idx].find('bndbox').find(name).text = pt
        print(idx)
        idx += 1
    # total_flag判断总体其中是否至少有1个被修改，total_flag=1时代表有，total_flag=0便无

    if(len(BoundingBoxes_list) == 0): print("该图片已经没有BoundingBox")
    return total_flag


#-----------------------------Sequential-augument choose here-----
ia.seed(1)
 
# Define our augmentation pipeline.
sometimes = lambda aug : iaa.Sometimes(0.3, aug)
seq = iaa.Sequential([
    # weather
    iaa.Sometimes(0.3, iaa.Affine(translate_percent={"x": 0.1, "y": 0.1})),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=10)),
    iaa.Sometimes(0.3, iaa.Fliplr(0.5)),
    # iaa.Sometimes(0.3, iaa.PerspectiveTransform((0.01, 0.1))),
    iaa.Sometimes(0.3, iaa.AddToHueAndSaturation((-18, 18))),
    iaa.Sometimes(0.3, iaa.LinearContrast((0.8, 1.2), per_channel=0.5)),
    iaa.Sometimes(0.3, iaa.FastSnowyLandscape(lightness_threshold=40, lightness_multiplier=2)),
    iaa.Sometimes(0.3, iaa.Clouds()),
    iaa.Sometimes(0.4, iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03))),
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0))),
    # iaa.Sometimes(0.3, iaa.Affine(rotate=(-3, 3))),    # rotate by -3 to 3 degrees (affects segmaps)
], random_order=True)


# for 多线程, 一个batch多张图片
def save_to_file(BATCH_SIZE, batches_aug_old, idx_jpg_path, out_dir_old, xml_et_old,id):
    print(f"thread{id} start")
    # unnormalizedBatch存放 images=images, bounding_boxes_aug
    # bounding_boxes_aug 存放着 list:[BoundingBoxOnImage, BoundingBoxOnImage, BoundingBoxOnImage]
    # BoundingBoxOnImage: 两个变量
    # list: [BoundingBox(x1=p[0],y1=p[1],x2=p[2],y2=p[3]) for p in box_list]
    # shape: img_content.shape

    batch_aug = copy.deepcopy(batches_aug_old)
    # print(f'batch_aug: type : {type(batch_aug)}')
    # idx_json = copy.deepcopy(idx_json_old)
    xml_et = copy.deepcopy(xml_et_old)
    out_dir = copy.deepcopy(out_dir_old)
    where_dir = os.path.dirname(idx_jpg_path)

    out_img_dir = os.path.join(out_dir, 'JPEGImages')
    print("out_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dirout_img_dir:", out_img_dir)
    out_Anno_dir = os.path.join(out_dir, 'Annotations')

    # 随便找一个kps都有相同图片的shape
    # img_shape = batch_aug.keypoints_aug[0].shape
    img_shape = batch_aug.bounding_boxes_aug[0].shape
    # print(img_shape)
    for i in range(BATCH_SIZE):
        elementTree = copy.deepcopy(xml_et)
        # print(f'ETTTT:{elementTree}')
        batch_aug.images_aug[i].astype(np.uint8)
        # write aug_points in json file
        # print(f'batches_aug_old length: {len(batches_aug_old)}')
        # print(f'batches_aug_old length: {len(batches_aug_old)}')
        # kps_aug_clip = batch_aug.keypoints_aug[i].clip_out_of_image()

        # batches_aug_old：UnnormalizedBatch(images=images, keypoints=keypoints)
        # batches_aug_old:check_and_alter_bndbox object：UnnormalizedBatch
        flag = check_and_alter_bndbox(elementTree, batch_aug, img_shape, i)
        print(f"{id}.bdxaug--是否有修改不符合的bndbox:{flag}")
        if flag == -1:
            print("跳过这张图片")
            continue
        # print(f"{id}.kpsaug_clip:{kps_aug_clip}")
        # print(f"{id}.kpsaug_clip:{len(kps_aug_clip)}")
        print(f"{id}.bndboxaug--{batch_aug.bounding_boxes_aug[i]}")
        # print(f"{id}.bndboxaug--{len(batch_aug.bounding_boxes_aug[i])}")

        # 写入新的xml以及jpg名字
        # idx_new_json = write_points_to_json(idx_json, batch_aug.keypoints_aug[i])
        # idx_new_json = write_points_to_json(idx_json, kps_aug_clip)
        new_imagePath_name = idx_jpg_path.split(os.sep)[-1][:-4] + '_at_' +str(i) + '.jpg'
        # print('bbbbbbbbbbbbbbbbbbbbbbbbbb',new_imagePath_name)
        new_xmlPath_name = idx_jpg_path.split(os.sep)[-1][:-4]+ '_at_' + str(i) + '.xml'


        # print(f"idx_jpg_path:{idx_jpg_path}")
        # idx_new_json["imagePath"] = idx_jpg_path.split(os.sep)[-1][:-4] + str(i) + '.jpg'
        # idx_new_json["imageData"] = str(utils.img_arr_to_b64(batch_aug.images_aug[i]),
        #                                 encoding='utf-8')

        # save jpg
        print('saving to file')
        new_img_path = os.path.join(out_img_dir, new_imagePath_name)
        new_Anno_path = os.path.join(out_Anno_dir, new_xmlPath_name)

        root = elementTree.getroot()
        root.find('filename').text = new_imagePath_name

        try:
            root.find('path').text = new_img_path
        except AttributeError as att:
            print(att)
            print("该xml文件没有'path'属性，跳过")

        # print('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC',root.findall())
        # print(batch_aug.images_aug[i])
        cv2.imwrite(new_img_path, batch_aug.images_aug[i])
        write_xml(elementTree, new_Anno_path)
        # new_json_path = new_img_path[:-3] + 'json'

        # 废弃
        # save_jsonfile(idx_new_json, new_json_path)
        # 新增替换
        # imageData = load_image_file(new_img_path)
        # imageData = base64.b64encode(imageData).decode("utf-8")
        # idx_new_json["imageData"] = imageData
        # with open(new_json_path, "w") as f:
        #     json.dump(idx_new_json, f, ensure_ascii=False, indent=2)

        print(f'save: {idx_jpg_path.split(os.sep)[-1][:-4]}{str(id)}{str(i)}.jpg , done')
    print(f"thread{id} finished")


from tqdm import tqdm
from imgaug import multicore
 
if __name__ == '__main__':
    # TO-DO-BELOW
    # aug_times = 2
    in_dir = r"D:\萱草-白色背景\一片叶\锈病"
    in_img_dir = os.path.join(in_dir, 'JPEGImages')
    in_xml_dir = os.path.join(in_dir, 'Annotations')
    # ######## 不能有中文名!!!!!!!!!!!!!!!!!!!
    out_dir = r"D:\xc_new_aug"
    out_img_dir = os.path.join(out_dir, 'JPEGImages')
    out_xml_dir = os.path.join(out_dir, 'Annotations')
    #---check-------------
    mkdir(out_dir)
    mkdir(out_img_dir)
    mkdir(out_xml_dir)
    imgs_dir_list = glob.glob(os.path.join(in_img_dir, '*.jpg'))
    # check_json_file(imgs_dir_list)
    check_xml_file(in_xml_dir, imgs_dir_list)

    # 新增
    img_number = len(imgs_dir_list)
    BATCH_SIZE = 3
    NB_BATCHES = 3

    # 循环次数 = int(图片数量 / 每次需要增强的原始图片数量) + 1
    # 如 3 = int(8 / 3) + 1, 顺序依次放3张，3张，2张
    cycle_times = int(img_number/NB_BATCHES) + 1


    import time
    from imgaug.augmentables.batches import UnnormalizedBatch
    init_index = 0
    time_start = time.time()
    for cycle_time in tqdm(range(cycle_times)):
        # 初始化list
        jpg_paths = []
        img_list = []
        # kps_list = []
        # json_list = []

        xml_et_list = []
        bbxes_list = []

        cyc_t = NB_BATCHES
        # 例子：如果是最后一个cycle_time，这个batch只有 2张图片，而不是3张
        # 比如 images_number = 8 ，而NB_BATCHES = 3 那么最后一个batch只有2张图片需要增强
        # 那么2 = images_number % NB_BATCHES 即 2 = 8 % 3
        if cycle_time == (cycle_times-1) :
            cyc_t = (cycle_times % NB_BATCHES)
        for id in range(cyc_t):
            img_list_index = cycle_time * NB_BATCHES + id

            # 添加图片路径进list
            jpg_paths.append(imgs_dir_list[img_list_index])
            jpg_name = os.path.split(imgs_dir_list[img_list_index])[-1]

            # 生成json的path
            # json_path = imgs_dir_list[img_list_index][:-3] + 'json'
            # 生成xml的path
            xml_name = jpg_name[:-3] + 'xml'
            xml_path = os.path.join(in_xml_dir, xml_name)

            # 读取图片内容
            img_content = cv2.imdecode(np.fromfile(imgs_dir_list[img_list_index], dtype=np.uint8), 1)
            # print(img_content)
            # 读取json内容
            # idx_json = read_jsonfile(json_path)
            # print(type(idx_json))
            # print(idx_json)
            # json_list.append(idx_json)

            # 从json内容里获取keypoint信息
            # points_list = get_points_from_json(idx_json)
            # 读取keypoint放进KeypointsOnImage
            # kps = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in points_list], shape=img_content.shape)

            # 从xml文件读取bounding_box信息
            etree = get_ET_from_xml(xml_path)
            xml_et_list.append(etree)
            box_list = get_bndbox_from_ET(etree)
            # print(id ,' ',box_list)
            # 读取bbx进BoundingBoxesOnImage
            # print(img_content.shape)
            bndboxes = BoundingBoxesOnImage([BoundingBox(x1=p[0],y1=p[1],x2=p[2],y2=p[3]) for p in box_list], shape=img_content.shape)

            # 一张图片中
            # kps_list.append([kps.deepcopy() for _ in range(BATCH_SIZE)])
            img_list.append([np.copy(img_content) for _ in range(BATCH_SIZE)])

            # 复制 bnd_boxes
            # 出错
            # bbxes_list.append(bndboxes.deepcopy() for _ in range(BATCH_SIZE))
            # 正确
            bbxes_list.append([bndboxes.deepcopy() for _ in range(BATCH_SIZE)])
            # print('bbxes_list:',bbxes_list)
            init_index += 1

        # polygons
        # batches = [UnnormalizedBatch(images=images, keypoints=keypoints) for images,keypoints in zip(img_list,kps_list)]

        # bndbox
        batches = [UnnormalizedBatch(images=images, bounding_boxes=bndboxes) for images,bndboxes in zip(img_list,bbxes_list)]
        # print(f'batches:{batches}')
        # 方法1.使用多线程及多batch 测试结果: 处理速度大概为 82s/3张图片
        # batches_aug = list(seq.augment_batches(batches, background=True))  # list() converts generator to list

        # 方法2.调用Pool函数，processes可以调节，-1表示调到最大
        # maxtasksperchild = 20 时，处理速度大概为 78.84s/生成3张图片 后面 85s/生成3张
        # maxtasksperchild = 10 时，处理速度大概为 72.53s/生成3张图片 后面74.58s/生成3张
        # 把NB_BATCHES从3改成1后，maxtasksperchild = 10时，处理速度大概稳定在 54s/生成3张图片
        # 使用多线程保存jpg以及json后，处理速度提高到 56s/生成9张图片
        with multicore.Pool(seq, processes=-1, maxtasksperchild=3, seed=1) as pool:
            batches_aug = pool.map_batches(batches)

        # ia.imshow(batches_aug[0].images_aug[0])

        # print(batches_aug)
        # print(len(batches_aug))

        # 多线程尝试
        # 用多线程之后,处理速度大概稳定在 25s/生成3张图片
        # batches_aug_old = copy.deepcopy(batches_aug)
        # print('aaaaaaaaaaaaaaaaaaaaaaaa',jpg_paths[0])


        t1 = threading.Thread(target=save_to_file, args=(BATCH_SIZE, batches_aug[0], jpg_paths[0], out_dir, xml_et_list[0],0))
        t1.daemon = False
        t1.start()
        t2 = threading.Thread(target=save_to_file, args=(BATCH_SIZE, batches_aug[1], jpg_paths[1], out_dir, xml_et_list[1],1))
        t2.daemon = False
        t2.start()
        t3 = threading.Thread(target=save_to_file, args=(BATCH_SIZE, batches_aug[2], jpg_paths[2], out_dir, xml_et_list[2],2))
        t3.daemon = False
        t3.start()
        # time.sleep(7)

    time_end = time.time()
    print("计时结束，KeypointsOnImage process done in %.2fs" % (time_end - time_start,))
