# image_classification_cnn_keras
## 使用Python深度学习API框架keras 构建卷积神经网络进行图像分类

使用方式：
-  1.往 /data/train 和 /data/val 两个文件夹里从0开始创建并命名文件夹名，有多少个分类就建多少个文件夹,文件夹名即分类名序号.
-  2.往各个文件夹内放对应类的训练图片和测试数据图片.
-  3.确定好序号对应的类名，预测脚本内需要配置.
-  4.根目录下运行 pip3 install -r requirements.txt 安装依赖项
-  5.运行train.py 训练模型.
-  6.命令行运行val.py 并指定要预测的图片路径，以验证模型精度。
-  7.网络调参
