import os
import json

# 0.从字典里面提取

# 一.json文件的操作
# 1.open json文件
# 2.解析json文件的内容，生成一个cache
# 3.使用二的操作
# 4.用cache的内容保存一个新的json文件

def save_json(output_path, js):
    with open(output_path, 'w') as file:
        json.dump(js,file)

def fix_coco_anno_json_imgid(input_path, start_id=0):

    with open(input_path, 'r', encoding='utf-8') as file:
        # 读成字典
        js = json.load(file)
        img_list = js['images']

    convert_table, oldName_newId_dict = generate_and_repair_image_id(js, img_list, start_id)

    output_path = input_path.split('.')[0] + "_new.json"
    print(js['images'])
    save_json(output_path, js)

    return oldName_newId_dict



# 二.给每个文件名字赋予一个新的纯数字的image_id，然后返回字典
# 1.读取json里面原始的image_id和image文件的名称
# 2.给每个image新建一个id，并且做成一个字典供查询
# 3.返回id字典
# 4.保存id字典成一个xls供参考(optional)
def generate_and_repair_image_id(js, old_id_list, start_id=0):
    # old_id_list = js['images']
    # 所以直接在old_id_list修改里面的内容
    img_list_len = len(old_id_list)

    # print("old_id_list", old_id_list)

    convert_table = []
    oldName_newId_dict = {}

    for i, img_dict in enumerate(old_id_list):
        # new_name new_id old_name old_id
        i = i + start_id
        new_id, new_name = str(i), str(i)+'.jpg'
        convert_table.append(zip(new_name, new_id, img_dict['file_name'], img_dict['id']))
        oldName_newId_dict[str(img_dict['file_name'].split('.')[0])] = new_id

        # inplace操作
        repair_image_name_and_id_in_json_images(img_dict, new_id, new_name)

    # 全部完成后，再根据convert_table修改annotations部分的image_id
    fix_image_id_in_json_annotations(js, oldName_newId_dict)
    return convert_table, oldName_newId_dict

# 三.通过id字典更改json文件里面的内容
# 更改image_id和文件名
def repair_image_name_and_id_in_json_images(js, new_id, new_name):
    js["id"] = new_id
    js["file_name"] = new_name


# 根据 converted table 修改
def fix_image_id_in_json_annotations(js, oldName_newId_dict):
    annotations = js['annotations']
    annotations_len = len(annotations)

    for i, ann_dict in enumerate(annotations):
        try:
            ann_dict["image_id"] = oldName_newId_dict[ann_dict["image_id"]]
        except Exception as e:
            print(e)
            print("In oldName_newId_dict, we can't found key:", ann_dict["image_id"])

# annotation json标注文件的格式主要如下所示
# {
#    "images":[
#                {
#                     "file_name": "1094.jpg",     !!!!!需要修改
#                     "height": 1024,
#                     "width": 1024,
#                     "id": "1094"                 !!!!!需要修改
#                },
#                .........
#    ],
#
#    "annotations": [
#         {
#             "area": 1272,
#             "iscrowd": 0,
#             "image_id": "1094", !!!!!需要修改
#             "bbox": [
#                 867,
#                 707,
#                 53,
#                 24
#             ],
#             "category_id": 2,
#             "id": 1,
#             "ignore": 0,
#             "segmentation": []
#         },
#         ...
#     ]
# }
#

# def fix_image_id_in_json(json_type, image_id_dict):

# 四.通过字典cache修改文件(jpg/xml)名称
# 1.参数读入字典
# 2.用os.listdir查看该文件夹里面的jpg/xml文件列表
# 3.通过字典

import shutil
def fix_image_jpg_xml_name_by_id_dict(xml_path=None, jpg_path=None, image_id_dict=None):

    # image_id_dict : {oldname: new_id} (str, str)
    if jpg_path is not None:
        # print(os.listdir(jpg_path))
        jpg_name_list = os.listdir(jpg_path)
        jpg_path_new = jpg_path + '_new'

        #先创建一个新的文件夹
        if os.path.exists(jpg_path_new) != True:
            os.makedirs(jpg_path_new)

        #### 将所有jpg复制一份到新文件夹
        for jn in jpg_name_list:
            shutil.copy(os.path.join(jpg_path,jn), os.path.join(jpg_path_new,jn))

        # 将新文件夹里面的所有jpg文件更名
        jpg_path_new_list = os.listdir(jpg_path_new)
        for jpg_name in jpg_path_new_list:
            jpg_name_path = os.path.join(jpg_path_new, jpg_name)
            new_path_name = os.path.join(jpg_path_new, image_id_dict[jpg_name.split('.')[0]] + '.jpg')
            os.rename(jpg_name_path, new_path_name)

    if xml_path is not None:

        print(os.listdir(xml_path))
        xml_path_new = xml_path + '_new'
        xml_name_list = os.listdir(xml_path)

        #先创建一个新的文件夹
        if os.path.exists(xml_path_new) != True:
            os.makedirs(xml_path_new)

        #### 将所有xml复制一份到新文件夹
        for xn in xml_name_list:
            shutil.copy(os.path.join(xml_path,xn), os.path.join(xml_path_new,xn))

        # 将新文件夹里面的所有xml文件更名
        xml_name_new_list = os.listdir(xml_path_new)
        for xml_name in xml_name_new_list:
            xml_name_path = os.path.join(xml_path_new, xml_name)
            new_path_name = os.path.join(xml_path_new, image_id_dict[xml_name.split('.')[0]] + '.xml')
            os.rename(xml_name_path, new_path_name)


if __name__ == '__main__':
    # COCO的json文件已经在这一步完成更名

    # parameter：(原始json文件路径(str))
    oldName_newId_dict = fix_coco_anno_json_imgid('coco.json')

    # Annotations是VOC的Annotations: 里面存放着XML文件
    # 写成'./Annotations' 和 './JPEGImages'
    # 不要写成'./Annotations/' 和 './JPEGImages/'

    # parameter：(原始VOC标注所在的文件夹(str), 原始jpg所在的文件夹(str),
    # 用旧名称查找对应新id、新名称的字典: dict {old_name(str): new_id(str)})
    fix_image_jpg_xml_name_by_id_dict('./Annotations','./JPEGImages', oldName_newId_dict)
    # 假如 ”没有Annotations文件夹(没有VOC的标注文件)“ 或者 ”不想转换VOC的标注“：注释上面的，使用下面这行代码
    # fix_image_jpg_xml_name_by_id_dict(None, './JPEGImages', oldName_newId_dict)
