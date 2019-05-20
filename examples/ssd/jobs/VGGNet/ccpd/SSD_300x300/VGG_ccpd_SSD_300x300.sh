cd /home/zhuyipin/projects/caffe_ssd_lpr/caffe/examples/ssd
./build/tools/caffe train \
--solver="models/VGGNet/ccpd/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/ccpd/SSD_300x300/VGG_ccpd_SSD_300x300.log
