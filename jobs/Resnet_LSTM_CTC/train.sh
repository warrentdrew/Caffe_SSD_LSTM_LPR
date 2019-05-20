cd /home/zhuyipin/CAFFE_LPR
./build/tools/caffe train \
--solver="./jobs/Resnet_LSTM_CTC/solver.prototxt" \
--gpu 0 2>&1 | tee ./jobs/Resnet_LSTM_CTC//lpr_resnet_lstm.log
