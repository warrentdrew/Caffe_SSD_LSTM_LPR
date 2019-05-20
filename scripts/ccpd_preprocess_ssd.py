import os
from os import environ
from cv2 import imread
ROOT_CCPD = "/home/zhuyipin/DATASET/lpr_v1/ssd/val"

#the test_name_size.txt file is used in caffe ssd when doing the testing
#create the test_name_size.txt from test.txt
def prepare_test_name_size(path, out):
    with open(path, 'r') as fr:
        for line in fr:
            subpath = str(line.split(' ')[0])
            fullname = os.path.split(subpath)[-1]
            name = os.path.splitext(fullname)[0]
            imgpath = os.path.join(ROOT_CCPD, fullname)
            print(imgpath)

            h, w, _ = imread(imgpath).shape
            output = "{} {} {}\n".format(name, h, w)
            with open(out, 'a') as fw:
                fw.write(output)


if __name__ == '__main__':
    prepare_test_name_size('../data/ccpd/val.txt', out = '../data/ccpd/test_name_size.txt')
