/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned y
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <atomic>
#include <mutex>
#include <chrono>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <signal.h>

#include "rknn_api.h"
#include "yolov3_post_process.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <sys/syscall.h>

using namespace std;
using namespace cv;

typedef pair<int, Mat> imagePair;
class paircomp {
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) return n1.first > n2.first;
        return n1.first > n2.first;
    }
};

mutex mtxQuit; 
mutex mtxQueueInput;               // mutex of input queue
mutex mtxQueueShow;                // mutex of display queue
queue<pair<int, Mat>> queueInput;  // input queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;  // display queue

int multi_npu_process_initialized[2] = {0, 0};
int demo_done=0;

bool bReading = true;
bool quit;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main()
{
	const char *model_path = "./yolov3_416.rknn";
	const int net_width=416;
	const int net_height=416;
	const int img_channels=3;
	const char *img_path = "./dog.jpg";

    // Load image
    cv::Mat orig_img = cv::imread(img_path, 1);
    cv::Mat img = orig_img.clone();
    if(!orig_img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
	
	cpu_set_t mask;
	int cpuid = 0;
	int ret = 0;
    int thread_id=0;
	if (thread_id == 0)
		cpuid = 4;
	else if (thread_id == 1)
		cpuid = 5;
	else
		cpuid = 0;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind NPU process(%d) to CPU %d\n", thread_id, cpuid);

	rknn_input inputs[1];
  	rknn_output outputs[3];
  	rknn_tensor_attr outputs_attr[3];
	detection* dets = 0;

	int nboxes_total = (GRID0*GRID0+GRID1*GRID1+GRID2*GRID2)*nanchor;
	dets =(detection*) calloc(nboxes_total,sizeof(detection));
	for(int i = 0; i < nboxes_total; ++i)
		dets[i].prob = (float*) calloc(nclasses,sizeof(float));

	// Load model
	FILE *fp = fopen(model_path, "rb");
    	if(fp == NULL) {
        	printf("fopen %s fail!\n", model_path);
        	return;
    	}
    	fseek(fp, 0, SEEK_END);   //fp指向end,fseek(FILE *stream, long offset, int fromwhere);
    	int model_len = ftell(fp);   //相对文件首偏移
    	void *model = malloc(model_len);
    	fseek(fp, 0, SEEK_SET);   //SEEK_SET为文件头
    	if(model_len != fread(model, 1, model_len, fp)) {
        	printf("fread %s fail!\n", model_path);
        	free(model);
        	return;
    	}
	
	//init
	rknn_context ctx = 0;
	ret = rknn_init(&ctx,model,model_len,RKNN_FLAG_PRIOR_MEDIUM);
	if(ret < 0) {
        	printf("rknn_init fail! ret=%d\n", ret);
        	return;
    	}
	
	//rknn inputs
	inputs[0].index = 0;
	inputs[0].size = net_width * net_height * img_channels;
	inputs[0].pass_through = false;         //需要type和fmt
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;

	//rknn outputs
	outputs[0].want_float = true;
	outputs[0].is_prealloc = false;
	outputs[1].want_float = true;
	outputs[1].is_prealloc = false;
	outputs[2].want_float = true;
	outputs[2].is_prealloc = false;

	//rknn outputs_attr
	outputs_attr[0].index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
	if(ret < 0) {
	    printf("rknn_query fail! ret=%d\n", ret);
	    return;
	}

	multi_npu_process_initialized[thread_id] = 1;
  	printf("The initialization of NPU Process %d has been completed.\n", thread_id);
	
	//==================================================================================//
	// YOLO Process
	int nboxes_left = 0;
	cv::Mat resimg;
	cv::resize(img, resimg, cv::Size(net_width, net_height), (0, 0), (0, 0), cv::INTER_LINEAR);
	//cv::cvtColor(resimg, resimg, cv::COLOR_BGR2RGB);
	
	double start_time,end_time;
	
	start_time=what_time_is_it_now();
	inputs[0].buf = resimg.data;
	ret = rknn_inputs_set(ctx, 1, inputs);
	if (ret < 0) {
		printf("rknn_input_set fail! ret=%d\n", ret);
		return ret;
	}
		
	ret = rknn_run(ctx, nullptr);
	if (ret < 0) {
	    printf("rknn_run fail! ret=%d\n", ret);
	    return ret;
	}

	ret = rknn_outputs_get(ctx, 3, outputs, NULL);
	if (ret < 0) {
		printf("rknn_outputs_get fail! ret=%d\n", ret);
		return ret;
	}
	end_time=what_time_is_it_now();
	//cout<<"rknn use time: "<<(end_time - start_time)<<"\n";
	
	start_time = what_time_is_it_now();
	for(int i = 0; i < nboxes_total; ++i)
		dets[i].objectness = 0;
		
	outputs_transform(outputs, IMG_WID, IMG_HGT, dets);
	// outputs_transform(outputs, net_width, net_height, dets);
	end_time=what_time_is_it_now();
	//cout<<"outputs_transform use time: "<<(end_time - start_time)<<"\n";

	start_time = what_time_is_it_now();
	nboxes_left=do_nms_sort(dets, nboxes_total, nclasses, NMS_THRESH);
	end_time=what_time_is_it_now();
	//cout<<"do_nms_sort use time: "<<(end_time - start_time)<<"\n";
	
	start_time = what_time_is_it_now();
	draw_image(img, dets, nboxes_left, DRAW_CLASS_THRESH);
	// resimg = Mat::zeros(cv::Size(net_width, net_height),CV_8UC3);
	// draw_image(resimg, dets, nboxes_left, DRAW_CLASS_THRESH);
	// cv::resize(resimg, resimg, cv::Size(IMG_WID, IMG_HGT), (0, 0), (0, 0), cv::INTER_LINEAR);
	// resimg.copyTo(pairIndexImage.second);
	end_time=what_time_is_it_now();
	//cout<<"draw_image use time: "<<(end_time - start_time)<<"\n";

	rknn_outputs_release(ctx, 3, outputs);
}
