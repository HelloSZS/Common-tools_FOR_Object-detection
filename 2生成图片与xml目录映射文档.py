import os
import os.path as osp
import re
import random

devkit_dir = './'


def get_dir(devkit_dir,  type):
    return osp.join(devkit_dir, type)


def walk_dir(devkit_dir):

    annotation_dir = get_dir(devkit_dir, 'Annotations')


    img_dir = get_dir(devkit_dir, 'JPEGImages')
    trainval_list = []
    test_list = []
    added = set()
    ii = 1
    #...................
    img_ann_list = []

    for i in range(2):
        if (ii == 1):
            img_ann_list = trainval_list
            fpath = "train.txt"#这里写train.txt的路径,如果不在同一个文件夹下,记得把路径写对
            ii = ii + 1
        elif (ii == 2):
            img_ann_list = test_list
            fpath = "val.txt"#这里写train.txt的路径,如果不在同一个文件夹下,要记得把路径写对记得用"\\"表示文件夹分割 \v会被认作转义字符的
        else:
            print("error")

        for line in open(fpath):
            name_prefix = line.strip().split()[0]
            if name_prefix in added:  # 这里的这个可以防止错误,检测到重复的则跳出本次循环
                continue
            added.add(name_prefix)
            ann_path1 = osp.join(name_prefix + '.xml')
            # print(ann_path1)

            ann_path = osp.join(annotation_dir + "/" + ann_path1)

            # print("begin")
            # print(ann_path)

            img_path1 = osp.join(name_prefix + '.jpg')
            img_path = osp.join(img_dir + "/" + img_path1)
            # print("begin2")
            # print(img_path)

            # assert os.path.isfile(ann_path), 'file %s not found.' % ann_path
            # assert os.path.isfile(img_path), 'file %s not found.' % img_path
            img_ann_list.append((img_path, ann_path))

            # print("begin3")
            # print(trainval_list)
            # print(test_list)

        # print("trainval_list:")
        # print(trainval_list)
        #
        # print("test_list")
        # print(test_list)


    return trainval_list, test_list


def prepare_filelist(devkit_dir, output_dir):
    trainval_list = []
    test_list = []
    trainval, test = walk_dir(devkit_dir)
    trainval_list.extend(trainval)
    test_list.extend(test)
    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'm_train.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')

    with open(osp.join(output_dir, 'm_val.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')


if __name__ == '__main__':
    prepare_filelist(devkit_dir, '.')