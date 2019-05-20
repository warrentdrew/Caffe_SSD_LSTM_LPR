cd /home/zhuyipin/CAFFE_LPR
./build/tools/caffe train \
--solver="models/VGGNet/ccpd/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/ccpd/SSD_300x300/VGG_ccpd_SSD_300x300_iter_1000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/ccpd/SSD_300x300/VGG_ccpd_SSD_300x300.log
