#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]

def net_result_to_string(res):
    str = ''

    for x in res[0]:
        index = int(x[0][0])
        #print('idx:', index)
        if index != -1:
            str += chars[index]

    return str






class CaffeRecog:
    def __init__(self, gpu_id, model_def, model_weights, image_height, image_width):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_height = image_height
        self.image_width = image_width
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))

        #self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        #self.transformer.set_raw_scale('data', 255)

        # the reference model has channels in BGR order instead of RGB
        #self.transformer.set_channel_swap('data', (2, 1, 0))

    # def predict(self, image_file):
    #     # set net to batch size of 1
    #     # image_resize = 300
    #     self.net.blobs['data'].reshape(1, 3, self.image_height, self.image_width)
    #     image = caffe.io.load_image(image_file) #shape h, w, 3
    #
    #     #print("shape!", image.shape)
    #
    #
    #     transformed_image = self.transformer.preprocess('data', image)
    #
    #     self.net.blobs['data'].data[...] = transformed_image
    #
    #     # Forward pass.
    #     result = self.net.forward()['result']
    #
    #     ret_str = net_result_to_string(result)
    #     return result, ret_str
    def predict(self, image):
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_height, self.image_width)
        #image = caffe.io.load_image(image_file) #shape h, w, 3

        #print(image[:10, :10, 0])


        transformed_image = self.transformer.preprocess('data', image)

        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        result = self.net.forward()['result']

        ret_str = net_result_to_string(result)
        return result, ret_str

def main(args):
    '''main '''
    error_count = 0
    recognet = CaffeRecog(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_height, args.image_width)
    #detection on all the images from path
    for i, imgname in enumerate(os.listdir(args.image_dir)):
        imgpath = os.path.join(args.image_dir, imgname)
        img = caffe.io.load_image(imgpath)
        result, _ = recognet.predict(img) #[10, 20 ,14 ,15, 25 , 63, 43]

        str_out = net_result_to_string(result)
        str_label = str(os.path.splitext(imgname)[0].split('_')[1])


        print('test image:', imgname)
        print(str_label, str_out)

        if str_label != str_out:
            error_count += 1
            print('This is an error image---------------------------:', error_count)


    sample_num = i + 1
    print("Acc:", (sample_num - error_count) / sample_num)




def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='examples/LPR/basic_lstm/deploy_revised.prototxt')
    parser.add_argument('--image_height', default=32, type=int)
    parser.add_argument('--image_width', default=128, type=int)
    parser.add_argument('--model_weights',
                        default= 'models/LPR/lpr_resnet_lstm_iter_60000.caffemodel')
    parser.add_argument('--image_dir', default='/home/zhuyipin/DATASET/ccpd/bbox_test')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())

