import os
import shutil

def copyfile(txt_path, postfix = '.jpg'):

    this_path = os.path.dirname(__file__) + '/JPEGImages/'
    new_path = os.path.dirname(__file__) + '/new_path/'
    # print(this_path)

    if os.path.exists(new_path) != True:
        os.makedirs(new_path)

    with open(txt_path, 'r') as t:
        # print(t.readlines())
        intxt_file_list = t.readlines()

        new_list = map(lambda x: x.strip() + postfix, intxt_file_list)
        print(list(new_list))

    for nle in new_list:
        file_path = this_path + nle
        try:
            shutil.copy(file_path, new_path)
        except Exception as e:
            print(e)


copyfile('train.txt')
