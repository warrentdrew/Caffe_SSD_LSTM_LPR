import os, sys

#to import caffe root path
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from caffe.ssd_detect_ccpd import main, CaffeDetection      #detection net output
from caffe.recog_eval import CaffeRecog                     #recognition net output
import argparse
import cv2
import time


def parse_args():
    '''parse args'''

    #dataset_root = "/home/zhuyipin/DATASET"

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default= '{}data/ccpd/labelmap.prototxt'.format(caffe_root))
    parser.add_argument('--ssd_model_def',
                        default= '{}models/SSD_300x300/deploy.prototxt'.format(caffe_root))
    parser.add_argument('--ssd_image_resize', default=300, type=int)
    parser.add_argument('--ssd_model_weights',
                        default= '{}models/SSD_300x300/lpr_detection.caffemodel'.format(caffe_root))

    parser.add_argument('--image_dir', default='test_images/')
    parser.add_argument('--intermediate_dir', default='intermediate/acc_test/')

    parser.add_argument('--recog_model_def', default='{}models/ResNet_CTC/deploy.prototxt'.format(caffe_root))
    parser.add_argument('--recog_image_width', default=128, type=int)
    parser.add_argument('--recog_image_height', default=32, type=int)
    parser.add_argument('--recog_model_weights', default='{}models/ResNet_CTC/lpr_resnet_ctc.caffemodel'.format(caffe_root))

    return parser.parse_args()

if __name__ == '__main__':
    write_intermediate = False
    args = parse_args()
    #initialize detection net
    detection = CaffeDetection(args.gpu_id,
                               args.ssd_model_def, args.ssd_model_weights,
                               args.ssd_image_resize, args.labelmap_file)

    recognet = CaffeRecog(args.gpu_id,
                          args.recog_model_def, args.recog_model_weights,
                          args.recog_image_height, args.recog_image_width)

    error_count = 0

    for i, imgname in enumerate(os.listdir(args.image_dir)):
        t0 = time.time()
        imgpath = os.path.join(args.image_dir,imgname)
        print("Test image: {}   id: {}".format(imgname, i + 1))
        ret, img = main(args, imgpath, detection)

        if not len(ret) > 0:
            error_count += 1
            t = time.time() - t0
            print('This image has no detection, total error: {}, time: {}s'.format(error_count, t))


        elif len(ret) > 1:
            error_count += 1
            t = time.time() - t0
            print('This image has extra detections, total error: {}, time: {}s'.format(error_count, t))


        else:
            xmin, ymin, xmax, ymax, confidence = ret[0]
            h, w, _ = img.shape

            if xmin < 0 or xmax > w:        #detection out of range
                print("wrong detection!")
                continue
            if ymin < 0 or ymax > h:
                print("wrong detection!")
                continue

            imgcrop = img[ymin: ymax, xmin: xmax, :]

            if write_intermediate:
                writepath = os.path.join(args.intermediate_dir, imgname)
                cv2img = cv2.imread(imgpath)
                imgwrite = cv2img[ymin: ymax, xmin: xmax, :]
                cv2.imwrite(writepath, imgwrite)

            _, str_pred = recognet.predict(imgcrop)
            str_label = imgname.split('.')[0].split('_')[-1]
            print("label: {}, pred: {}".format(str_label, str_pred))
            t = time.time() - t0
            if str_pred != str_label:
                error_count += 1
                print('This image has wrong recognition, total error: {}, time: {:.3f}s'.format(error_count, t))
            print('correct!!, time: {:.3f}s'.format(t))


    sample_num = i + 1

    print("Acc:", (sample_num - error_count) / sample_num)
