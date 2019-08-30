#encoding=utf8
import os, sys

#to run this file at any place, import caffe root path
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

from caffe.ssd_detect_ccpd import main, CaffeDetection      #detection caffe
from caffe.recog_eval import CaffeRecog                     #recognition caffe
import argparse
import cv2
import time

import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def drawRectBox(image,rect,addText, fontC):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode("utf-8").decode("utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

def parse_args():
    '''parse args'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default= '{}data/ccpd/labelmap.prototxt'.format(caffe_root))
    parser.add_argument('--ssd_model_def',
                        default= '{}models/SSD_300x300/deploy.prototxt'.format(caffe_root))
    parser.add_argument('--ssd_image_resize', default=300, type=int)
    parser.add_argument('--ssd_model_weights',
                        default= '{}models/SSD_300x300/lpr_detection.caffemodel'.format(caffe_root))


    parser.add_argument('--image_dir', default='{}sample_images/'.format(caffe_root))
    parser.add_argument('--result_dir', default='{}results/'.format(caffe_root))
    parser.add_argument('--intermediate_dir', default='{}intermediate'.format(caffe_root))

    parser.add_argument('--recog_model_def', default='{}models/ResNet_CTC/deploy.prototxt'.format(caffe_root))
    parser.add_argument('--recog_image_width', default=128, type=int)
    parser.add_argument('--recog_image_height', default=32, type=int)
    parser.add_argument('--recog_model_weights', default='{}/models/ResNet_CTC/lpr_resnet_ctc.caffemodel'.format(caffe_root))

    parser.add_argument('--font_file', default='{}font/platech.ttf'.format(caffe_root))

    return parser.parse_args()

if __name__ == '__main__':
    write_intermediate = True
    args = parse_args()
    #initialize detection net

    detection = CaffeDetection(args.gpu_id, args.ssd_model_def, args.ssd_model_weights,
                               args.ssd_image_resize, args.labelmap_file)

    recognet = CaffeRecog(args.gpu_id,
                          args.recog_model_def, args.recog_model_weights,
                          args.recog_image_height, args.recog_image_width)

    fontC = ImageFont.truetype(args.font_file, 16, 0)

    for imagename in os.listdir(args.image_dir):
        imagepath = os.path.join(args.image_dir, imagename)
        t0 = time.time()

        rets, img = main(args, imagepath, detection)
        h, w, _ = img.shape

        for item in rets:
            xmin, ymin, xmax, ymax, confidence = item
            if xmin < 0 or xmax > w:
                print("wrong detection!")
                continue
            if ymin < 0 or ymax > h:
                print("wrong detection!")
                continue

            imgcrop = img[ymin: ymax, xmin: xmax, :]

            if write_intermediate:
                writepath = os.path.join(args.intermediate_dir, imagename)
                interm_img = cv2.imread(imagepath)
                interm_img = interm_img[ymin: ymax, xmin: xmax, :]
                cv2.imwrite(writepath, interm_img)

            _, str_pred = recognet.predict(imgcrop)
            print('{}  time: {:.3f}s'.format(str_pred, (time.time() - t0)))
            rect = [xmin, ymin, xmax - xmin, ymax - ymin]

            imgtodraw = cv2.imread(imagepath)
            imagesave = drawRectBox(imgtodraw, rect, str_pred, fontC)

        cv2.imwrite(os.path.join(args.result_dir, imagename), imagesave)
