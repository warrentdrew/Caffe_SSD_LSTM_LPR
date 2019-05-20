#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from multiprocessing import Process
import caffe
import h5py

CAFFE_ROOT = os.getcwd()   # assume you are in $CAFFE_ROOT$ dir

img_path = '/home/zhuyipin/DATASET/lpr_v1/bbox_trainval_aug'
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 32
LABEL_SEQ_LEN = 7

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}



label_count=len(index)+1 #66
image_size = (128, 32)

def get_label(name):    #name example : 0_浙B116VT_1234.jpg
    lpr_str = str(os.path.splitext(name)[0].split('_')[1])
    label = []
    for s in lpr_str:
        label.append(index[s])
    return np.array(label)

def write_image_info_into_file(file_name, images):
    with open(file_name, 'w') as f:
        for image in images:
            img_name = os.path.splitext(image)[0].split('_')[1]
            #numbers = img_name[img_name.find('-')+1:]
            f.write(os.path.join(img_path, image) + "|" + ','.join(img_name) + "\n")


def write_image_info_into_hdf5(file_name, images, phase):
    total_size = len(images)
    print('[+] total image for {0} is {1}'.format(file_name, len(images)))
    single_size = 2500
    groups = total_size // single_size
    if total_size % single_size:
        groups += 1
    def process(file_name, images):
        img_data = np.zeros((len(images), 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype = np.float32)
        label_seq = label_count * np.ones((len(images), LABEL_SEQ_LEN), dtype = np.float32)
        for i, image in enumerate(images): #image here is the name of the image

            numbers = get_label(image)
            #print('numbers:', numbers)
            label_seq[i, :len(numbers)] = numbers
            img = caffe.io.load_image(os.path.join(img_path, image))
            img = caffe.io.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            img = np.transpose(img, (2, 0, 1))
            img_data[i] = img

        
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('data', data = img_data)
            f.create_dataset('label', data = label_seq)

    with open(file_name, 'w') as f:
        workspace = os.path.split(file_name)[0]
        process_pool = []
        for g in range(groups):
            h5_file_name = os.path.join(workspace, '%s_%d.h5' %(phase, g))
            print('t1:', h5_file_name)
            f.write(h5_file_name + '\n')
            start_idx = g*single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(images)
            p = Process(target = process, args = (h5_file_name, images[start_idx:end_idx]))
            p.start()
            process_pool.append(p)
        for p in process_pool:
            p.join()

if __name__ == "__main__":
    images = list(filter(lambda x: os.path.splitext(x)[1] == '.jpg', os.listdir(img_path)))  # image list
    print('total image number: {}'.format(len(images)))
    np.random.shuffle(images)

    training_size = 25000   # number of images for training
    training_images = images[:training_size]
    testing_images = images[training_size:]
    save_path = "/home/zhuyipin/DATASET/lpr_v1/caffe_hdf5"

    write_image_info_into_hdf5(os.path.join(save_path, 'training.list'), training_images, 'train')
    write_image_info_into_hdf5(os.path.join(save_path, 'testing.list'), testing_images, 'test')
    write_image_info_into_file(os.path.join(save_path, 'testing_images.list'), testing_images)

