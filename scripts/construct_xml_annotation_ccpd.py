import json
from os import path, environ, mkdir
import argparse
from shutil import move, copyfile
from cv2 import imread
from lxml import etree, objectify
import pandas as pd
import numpy as np
import os

# Will create annotations/val or annotations/train

ROOT_CCPD = "%s/DATASET/lpr_v1/" % (environ['HOME'])
#PATH_TO_IMAGES = "/home/zhuyipin/DATASET/ccpd/ssd_test"
PATH_TO_IMAGES = "%s/trainval/" % ROOT_CCPD
ROUND_COORDS = True
IMAGE_NAMES = []
TYPEPREFIX = 'trainval'
ANN_DIR = '%sssd/annotation/%s/' % (ROOT_CCPD, TYPEPREFIX)
TRAIN_SPLIT = 0.95
COPY_FILES = True

if not path.isdir(ANN_DIR):
    mkdir(ANN_DIR)

def parse_img_info(path):
    coord_dict = {}
    vertex_dict = {}
    info = os.path.split(path)[-1].split('-')
    coord = info[2]
    vertex = info[3].split("_")
    coord_ul = (int(coord.split('_')[0].split('&')[1]), int(coord.split('_')[0].split('&')[0]))  #bounding box upper left y,x
    coord_br = (int(coord.split('_')[1].split('&')[1]), int(coord.split('_')[1].split('&')[0]))  #bounding box bottom right y,x
    #print("ul:",coord_ul)
    #print("br:", coord_br)
    vertex_br = (int(vertex[0].split('&')[1]), int(vertex[0].split('&')[0]))
    vertex_bl = (int(vertex[1].split('&')[1]), int(vertex[1].split('&')[0]))
    vertex_ul = (int(vertex[2].split('&')[1]), int(vertex[2].split('&')[0]))
    vertex_ur = (int(vertex[3].split('&')[1]), int(vertex[3].split('&')[0]))
    # print("br:", vertex_br)
    # print("bl:", vertex_bl)
    # print("ul:", vertex_ul)
    # print("ur:", vertex_ur)
    coord_dict['ul'] = coord_ul
    coord_dict['br'] = coord_br
    vertex_dict['br'] = vertex_br
    vertex_dict['bl'] = vertex_bl
    vertex_dict['ul'] = vertex_ul
    vertex_dict['ur'] = vertex_ur
    return coord_dict, vertex_dict

def create_xml(datapath):
    for image in os.listdir(datapath):
        imagepath = os.path.join(datapath, image)
        imagename = os.path.splitext(image)[0]
        IMAGE_NAMES.append(imagename)
        if path.isfile(ANN_DIR + imagename + ".xml"):
            continue  # Don't overwrite, unless we force it to
        image_file = imread(PATH_TO_IMAGES + imagename + '.jpg')
        E = objectify.ElementMaker(annotate=False)
        img_annotation = E.annotation(
            E.folder(TYPEPREFIX),
            E.filename(imagename),
            E.source(
                E.database('CCPD'),
            ),
            E.size(
                E.width(image_file.shape[1]),
                E.height(image_file.shape[0]),
                E.depth(3),
            ),
            E.segmented(0)
        )

        coord_dict, vertex_dict = parse_img_info(imagepath)

        objectNode = E.object(
            E.name("Plate"),
            E.pose("Unspecified"),
            E.truncated("0"),
            E.difficult("0"),
            E.bndbox(
                E.xmin(str(coord_dict['ul'][1])),
                E.ymin(str(coord_dict['ul'][0])),
                E.xmax(str(coord_dict['br'][1])),
                E.ymax(str(coord_dict['br'][0])),
            ),
        )
        img_annotation.append(objectNode)
        xml_pretty = etree.tostring(img_annotation, pretty_print=True)
        with open(ANN_DIR + imagename + ".xml", 'wb') as ann_file:
            ann_file.write(xml_pretty)

    #print("finished with xmls, now moving or copying")
#path corresponding to xml
def write_pathtoxml_txt():
    if TYPEPREFIX == 'trainval': #if we want to have val set
        if TRAIN_SPLIT and TRAIN_SPLIT < 1.0:
            TRAIN_DIR = 'ssd/train/'
            TRAIN_FULL_DIR = '%s%s' % (ROOT_CCPD, TRAIN_DIR)
            TRAIN_ANN_DIR = 'ssd/annotation/train/'
            TRAIN_FULL_ANN_DIR = '%s%s' % (ROOT_CCPD, TRAIN_ANN_DIR)
            MINIVAL_DIR = 'ssd/val/'
            MINIVAL_FULL_DIR = '%s%s' % (ROOT_CCPD, MINIVAL_DIR)
            MINIVAL_ANN_DIR = 'ssd/annotation/val/'
            MINIVAL_FULL_ANN_DIR = '%s%s' % (ROOT_CCPD, MINIVAL_ANN_DIR)

            SELECTED_FOR_MINIVAL = []
            while len(SELECTED_FOR_MINIVAL) < (1.0 - TRAIN_SPLIT) * len(IMAGE_NAMES):
                RANDOM_IDX = np.random.randint(0, high=len(IMAGE_NAMES), size=1)
                while RANDOM_IDX in SELECTED_FOR_MINIVAL:
                    RANDOM_IDX = np.random.randint(
                        0, high=len(IMAGE_NAMES), size=1)
                SELECTED_FOR_MINIVAL.append(int(RANDOM_IDX))
            SELECTED_FOR_TRAIN = sorted(
                list(set(list(range(len(IMAGE_NAMES)))).difference(SELECTED_FOR_MINIVAL)))
            with open('train.txt', 'w') as train_file:
                for idx in SELECTED_FOR_TRAIN:
                    if COPY_FILES: #copy jpg and
                        copyfile(
                            PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                            TRAIN_FULL_DIR + IMAGE_NAMES[idx] + ".jpg")
                        copyfile(
                            ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                            TRAIN_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml")
                    else:
                        move(
                            PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                            TRAIN_FULL_DIR + IMAGE_NAMES[idx] + ".jpg")
                        move(
                            ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                            TRAIN_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml")
                    train_file.write(
                        "/" + TRAIN_DIR + IMAGE_NAMES[idx] + ".jpg /" + TRAIN_ANN_DIR + IMAGE_NAMES[idx] + ".xml\n")
            with open('val.txt', 'w')as minival_file:
                for idx in SELECTED_FOR_MINIVAL:
                    if COPY_FILES:
                        copyfile(
                            PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                            MINIVAL_FULL_DIR + IMAGE_NAMES[idx] + ".jpg"
                        )
                        copyfile(
                            ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                            MINIVAL_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml"
                        )
                    else:
                        move(
                            PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                            MINIVAL_FULL_DIR + IMAGE_NAMES[idx] + ".jpg"
                        )
                        move(
                            ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                            MINIVAL_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml"
                        )
                    minival_file.write(
                        "/" + MINIVAL_DIR + IMAGE_NAMES[idx] + ".jpg /" + MINIVAL_ANN_DIR + IMAGE_NAMES[idx] + ".xml\n")
        else:   #if we don't want to have val set(i.e. only create training set)
            with open('train.txt', 'w') as train_file:
                IMG_RELATIVE = '/ssd/train/'
                TRAIN_ANN_DIR = 'ssd/annotations/train/'
                TRAIN_FULL_ANN_DIR = "%s%s" % (ROOT_CCPD, TRAIN_ANN_DIR)
                if not path.isdir(TRAIN_FULL_ANN_DIR):
                    mkdir(TRAIN_FULL_ANN_DIR)
                for i in range(len(IMAGE_NAMES)):
                    train_file.write(
                        IMG_RELATIVE + IMAGE_NAMES[i] + ".jpg /" + TRAIN_ANN_DIR + IMAGE_NAMES[i] + ".xml\n")
    else:   #test set
        with open('test.txt', 'w') as val_file:
            IMG_RELATIVE = '/ssd/test/'
            VAL_ANN_DIR = 'ssd/annotations/test/'
            VALL_FULL_ANN_DIR = '%s%s' % (ROOT_CCPD, VAL_ANN_DIR)
            if not path.isdir(VAL_ANN_DIR):
                mkdir(VAL_ANN_DIR)
            for i in range(len(IMAGE_NAMES)):
                val_file.write(
                    IMG_RELATIVE + IMAGE_NAMES[i] + ".jpg /" + VAL_ANN_DIR + IMAGE_NAMES[i] + ".xml\n")



if __name__ == '__main__':
    create_xml(PATH_TO_IMAGES)
    write_pathtoxml_txt()


"""
Example of Pascal VOC 2009 annotation XML
<annotation>
	<filename>2009_005311.jpg</filename>
	<folder>VOC2012</folder>
	<object>
		<name>diningtable</name>
		<bndbox>
			<xmax>364</xmax>
			<xmin>161</xmin>
			<ymax>301</ymax>
			<ymin>200</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>298</xmax>
			<xmin>176</xmin>
			<ymax>375</ymax>
			<ymin>300</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Rear</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>432</xmax>
			<xmin>273</xmin>
			<ymax>339</ymax>
			<ymin>205</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>413</xmax>
			<xmin>297</xmin>
			<ymax>375</ymax>
			<ymin>268</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>465</xmax>
			<xmin>412</xmin>
			<ymax>273</ymax>
			<ymin>177</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Left</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>463</xmax>
			<xmin>427</xmin>
			<ymax>329</ymax>
			<ymin>225</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Left</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>186</xmax>
			<xmin>85</xmin>
			<ymax>374</ymax>
			<ymin>250</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>232</xmax>
			<xmin>74</xmin>
			<ymax>307</ymax>
			<ymin>175</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>273</xmax>
			<xmin>233</xmin>
			<ymax>200</ymax>
			<ymin>148</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Frontal</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>369</xmax>
			<xmin>313</xmin>
			<ymax>235</ymax>
			<ymin>165</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>151</xmax>
			<xmin>94</xmin>
			<ymax>244</ymax>
			<ymin>166</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>204</xmax>
			<xmin>156</xmin>
			<ymax>210</ymax>
			<ymin>157</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Frontal</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>350</xmax>
			<xmin>299</xmin>
			<ymax>216</ymax>
			<ymin>157</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>bottle</name>
		<bndbox>
			<xmax>225</xmax>
			<xmin>215</xmin>
			<ymax>230</ymax>
			<ymin>196</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>bottle</name>
		<bndbox>
			<xmax>180</xmax>
			<xmin>170</xmin>
			<ymax>210</ymax>
			<ymin>184</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>179</xmax>
			<xmin>110</xmin>
			<ymax>244</ymax>
			<ymin>160</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>87</xmax>
			<xmin>66</xmin>
			<ymax>270</ymax>
			<ymin>192</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<segmented>0</segmented>
	<size>
		<depth>3</depth>
		<height>375</height>
		<width>500</width>
	</size>
	<source>
		<annotation>PASCAL VOC2009</annotation>
		<database>The VOC2009 Database</database>
"""