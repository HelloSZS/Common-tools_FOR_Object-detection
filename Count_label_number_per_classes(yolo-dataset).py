import os


def get_train_val_label_path_and_list(path = None):

    if path == None:
        # file: '.../VOC_v5/'
        # this file have 'images' and 'labels' file
        this_path = os.path.dirname(__file__)
    else:
        try:
            this_path = os.path.dirname(path)
        except Exception as e:
            print(e)

    img_path = this_path + '/images/'
    label_path = this_path + '/labels/'

    spilt = ['train', 'val']

    # train_val_img_path = [img_path + f'/{i}/' for i in spilt]
    train_val_label_path = [label_path + f'/{i}/' for i in spilt]
    print('...,',train_val_label_path)

    # print(os.listdir(train_val_label_path[0]))

    train_label_list = os.listdir(train_val_label_path[0])
    val_label_list = os.listdir(train_val_label_path[1])

    # return str, str, list[str, ...], list[str, ...]
    return train_val_label_path[0], train_val_label_path[1], train_label_list, val_label_list

def strip_str(str, strip_by = '\n'):
    return str.strip(strip_by)


def statistic_how_many_object_in_this_classes(labels_path, labels_list):
    class_num = {}

    for label_name in labels_list:
        label_path = os.path.join(labels_path, label_name)


        with open(label_path, 'r') as f:
            data = f.readlines()

            data_ = map(strip_str, data)

            for i in data_:
                # print(i.split(' ')[0])
                classes = int(i.split(' ')[0])

                if classes not in class_num:
                    class_num[classes] = 1
                else:
                    class_num[classes] = class_num[classes] + 1

    return class_num

def get_counted_class(labels_path, labels_list):
    d = statistic_how_many_object_in_this_classes(labels_path, labels_list)
    class_num = sorted(d.items(), key=lambda x:x[0])
    # print(train_class_num)
    return class_num


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

# print(train_class_num, '\n', val_class_num)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_num_bar(class_num_dict):
    classes, classes_num = list(zip(*class_num_dict))
    print(classes, '\n', classes_num)

    sns.barplot(list(classes), list(classes_num))

    for a,b in zip(classes,classes_num):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)

    plt.xlabel('classes')
    plt.ylabel("numbers of object")
    plt.title("numbers_of_object_per_classes")

    plt.show()


# 首先获得 train/val的path还有name list

train_labels_path, val_labels_path, train_labels_list, val_labels_list = get_train_val_label_path_and_list()
# print("get_train_val_label_path_and_list()",len(train_labels_list), len(val_labels_list))
# 然后通过get_counted_class，输入labels_path和labels_name_list获得对应train/val的每个类别的label的数量
train_class_num_dict = get_counted_class(train_labels_path, train_labels_list)

val_class_num_dict = get_counted_class(val_labels_path, val_labels_list)

plot_class_num_bar(train_class_num_dict)
plot_class_num_bar(val_class_num_dict)
