# 介绍
基于caffe的中文车牌识别项目，包含基于SSD的车牌检测模型和基于Resnet + (LSTM) + CTC损失的车牌字符识别，目前只支持单张标准蓝牌识别。CTC损失函数理论上可以支持不定长车牌序列的识别，但目前没有在包含不定长车牌数据（如新能源车牌）的数据集上训练和测试。


# 环境
* Python == 3.6
* Cuda == 9.0
* Cudnn == 7.0
* OpenCV == 3.4
* 如果需要训练自己的数据集，百度有开源的CTC损失的优化加速实现warp-ctc，能够提升CTC运算速度，安装请参考[kasyoukin/caffe_ocr_for_linux](https://github.com/kasyoukin/caffe_ocr_for_linux)
* 如果不需要进行训练，不安装warp-ctc的情况下，请删除(或修改后缀名)`src/caffe/layers/`中的warp_ctc_loss_layer.cpp以及warp_ctc_loss_layer.cu, 避免在编译时产生错误

# 数据集
训练使用车牌识别公开数据集 CCPD， 完整版数据集下载参考原github项目: [ detectRecog/CCPD
](https://github.com/detectRecog/CCPD)

完整版数据集有明显的样本不平衡的问题，本项目是采用的重采样后的数据集进行训练

CCPD数据集目前只包含传统蓝牌，并且一图一牌， 所以目前训练的模型不支持新能源等其他车牌的检测识别以及一图多牌的情况

# 编译 
本项目为合并后caffe项目， 编译过程与标准caffe的编译过程相同，请根据运行环境修改Makefile.config后进行编译
```script
make all -j8
make pycaffe -j8
```

# demo使用
1. 将测试图片放入`sample_images/`
2. 运行demo.py
```script
python demo.py
```
3. 运行结果图片将保存在`results/`中

# 准确率测试
1. 将测试图片放入`test_images/`中，或软链接测试集路径
```script
ln -sf $your_own_path/* test_images/
```
2. 运行accuracy_test.py
```script
python accuracy_test.py
```

# 模型评价
从重采样后的CCPD数据集中随机采样500张作为测试机，车牌识别准确率为94.6%

# 结果示例
![img_text](https://github.com/warrentdrew/Caffe_SSD_LSTM_LPR/blob/master/results/203_%E4%BA%ACP676E1.jpg)
![img_text](https://github.com/warrentdrew/Caffe_SSD_LSTM_LPR/blob/master/results/38_%E8%B1%ABAWK983.jpg)


# TODO
- [ ] 创建重采样ccpd数据集下载链接
- [ ] 维护训练及前处理代码
- [ ] 训练并评价包含新能源以及其他类型车牌(不定长)的数据集
- [ ] 训练并评价单图多车牌的情况

# Reference
- SSD检测: [weiliu89/caffe](https://github.com/weiliu89/caffe/tree/ssd)
- 车牌OCR识别: [kasyoukin/caffe_ocr_for_linux](https://github.com/kasyoukin/caffe_ocr_for_linux)
- 参考开源项目: [zeusees/HyperLPR](https://github.com/zeusees/HyperLPR)
- 公开数据集: [detectRecog/CCPD](https://github.com/detectRecog/CCPD)
- CTC加速: [ChWick/warp-ctc](https://github.com/ChWick/warp-ctc)

