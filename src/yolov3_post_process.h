/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/
#ifndef YOLOV3_POST_PROCESS_H
#define YOLOV3_POST_PROCESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#define IMG_WID 416
#define IMG_HGT 416
#define GRID0 13
#define GRID1 26
#define GRID2 52
#define nclasses 80
#define nyolo 3
#define nanchor 3
#define OBJ_THRESH 0.6
#define DRAW_CLASS_THRESH 0.6
#define NMS_THRESH 0.4

typedef struct Box{
	float x,y,w,h;
}box;

typedef struct Detection{
    box bbox;
    int classes;
    float *prob;
    float objectness;
    int sort_class;
} detection;


#ifdef __cplusplus
extern "C"{
#endif
int outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height, detection* dets);
int do_nms_sort(detection *dets, int total, int classes, float thresh);
int draw_image(cv::Mat img, detection* dets, int total, float thresh);
#ifdef __cplusplus
}
#endif


#endif //YOLOV3_POST_PROCESS_H

