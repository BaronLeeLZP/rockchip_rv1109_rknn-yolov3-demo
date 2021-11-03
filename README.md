# rockchip_rv1109_rknn-yolov3-demo
在瑞芯微rockchip的AI芯片rv1109上，利用rknn和opencv库，修改了官方yolov3后处理部分代码Bug，
交叉编译yolov3-demo示例后可成功上板部署运行。
同时，对在瑞芯微rockchip的其他AI芯片上利用NPU rknn部署CNN模型提供参考和借鉴。
以下为demo工程说明：
## libopencv_api
交叉编译时所依赖的opencv库，本工程opencv版本为4.3.0，已经打包好了；
亦可根据自己项目需要选择不同版本/不同阉割程度的opencv依赖库，需自行编译。

## librknn_api
交叉编译时所依赖的rknn库，瑞芯微官方提供，rknn_api.h/librknn_api.so；
使用过程详见yolov3检测模型的预处理/加载/推断过程，main.cpp文件。

## src
### yolov3_post_process.h
yolov3 rknn模型推断原始结果的后处理函数接口：
1.int outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height, detection* dets);首要的后处理操作，数据转换
2.int do_nms_sort(detection *dets, int total, int classes, float thresh);非极大值抑制算法实现，欢迎码友提供更高效代码的实现
3.int draw_image(cv::Mat img, detection* dets, int total, float thresh);利用opencv绘制检测结果输出图像并保存为out.jpg
可封装为标准的输入输出接口

### yolov3_post_process.cpp
yolov3 rknn模型推断原始结果的后处理过程详细的代码实现

## main.cpp
demo主函数代码实现，可了解物体检测时rknn接口函数调用过程。
识别模型相比较于检测模型的推断，处理过程则更简单。

## 交叉编译环境
交叉编译工具见瑞芯微rockchip官方提供的SDK，路径位于**SDK/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf**
手里拿有板子的码友应该都有SDK的，对吧。依照文档配置好交叉编译环境。

## demo使用步骤如下：
1.工程编译生成test可执行文件；
  命令**make**
2.工程编译生成libyolo库文件；
  命令**make arm_lib**
3.在rv11xx上运行test文件可得yolov3-demo运行结果。
  将test文件/dog.jpg图像及其依赖库移植到RV1109上，登录嵌入式系统，命令**./test**运行demo即可得out.jpg文件输出结果。
![result]https://github.com/BaronLeeLZP/rockchip_rv1109_rknn-yolov3-demo/blob/master/out.jpg


如有技术问题欢迎交流！
如有版权问题请及时告知！


