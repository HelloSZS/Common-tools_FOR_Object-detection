# count object class

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

def read_xml_return_obj_list(root_path, xml_path):
    xml_tree = parse(os.path.join(root_path, xml_path))
    # 因为getElementsByTagName('name')返回列表只有1个元素，所以getElementsByTagName('name')[0]，来直接获取DOM Element
    # 然后又因为只有一个childNodes，所以:childNodes[0]
    # print(xml_tree.getElementsByTagName('object')[0].getElementsByTagName('name')[0].childNodes[0].data)
    # 返回一个list
    return xml_tree.getElementsByTagName('object')

# count_dict: {'class1': 15, 'class2': 33} ....
def count_object_in_classes(root_path):
    li_xml_name = find_xml_name_list(root_path)
    dict_num_object_on_classes = {}

    for i in range(len(li_xml_name)):
        object_list = read_xml_return_obj_list(root_path, li_xml_name[i])
        for i in range(len(object_list)):
            if str(object_list[i].getElementsByTagName('name')[0].childNodes[0].data) not in dict_num_object_on_classes:
                dict_num_object_on_classes[str(object_list[i].getElementsByTagName('name')[0].childNodes[0].data)] = 1
            else:
                dict_num_object_on_classes[str(object_list[i].getElementsByTagName('name')[0].childNodes[0].data)] += 1

    print(dict_num_object_on_classes)

# count numbers of object on every classes
count_object_in_classes(path)