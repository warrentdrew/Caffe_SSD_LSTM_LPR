cd /home/zhuyipin/projects/caffe_ssd_lpr/caffe
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300.log
