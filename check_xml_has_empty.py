import xml

import os

path = 'D:/xb_aug/slice/annotations/'

def split_list_by_extension_name(li, ext_name_list):
    li2 = []
    for name in li:
        if str(name.split('.')[-1]) in ext_name_list:
            li2.append(name)
    return li2

def find_xml_name_list(path):
    lis_all = os.listdir(path)
    li2 = split_list_by_extension_name(lis_all, ['xml'])
    return li2

# find_xml_list(path)

from xml.dom.minidom import parse

def read_xml(root_path, xml_path):
    xml_tree = parse(os.path.join(root_path, xml_path))
    # print(xml_tree.getElementsByTagName('object'))
    # 返回一个list
    return xml_tree.getElementsByTagName('object')

def check_xml_on_this_path_if_have_empty(root_path):
    li_xml_name = find_xml_name_list(root_path)
    empty_xml_name_list = []
    for i in range(len(li_xml_name)):
        object_list = read_xml(root_path, li_xml_name[i])

        # print(f"第{i}个xml，名字为{li_xml_name[i]}，object_list \n {object_list}")

        # if len(object_list) == 0:
        # if bool(object_list) == False:
        if not object_list:
            empty_xml_name_list.append(li_xml_name[i])
            # print("空！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1")

    if not empty_xml_name_list:
        return False

    return True

have_empty_xml = check_xml_on_this_path_if_have_empty(path)
print(have_empty_xml)