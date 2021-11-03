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
#include <sys/syscall.h>
#include <dirent.h>
#include <unistd.h>
#include <signal.h>

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "yolov3_post_process.h"

using namespace std;
using namespace cv;

Scalar colorArray[10]={
	Scalar(139,0,0,255),
	Scalar(139,0,139,255),
	Scalar(0,0,139,255),
	Scalar(0,100,0,255),
	Scalar(139,139,0,255),
	Scalar(209,206,0,255),
	Scalar(0,127,255,255),
	Scalar(139,61,72,255),
	Scalar(0,255,0,255),
	Scalar(255,0,0,255),
};

static string labels[80]={"person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat",
"traffic light","fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow",
"elephant","bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
"snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
"bottle","wine glass","cup","fork","knife ","spoon","bowl","banana","apple","sandwich","orange","broccoli",
"carrot","hot dog","pizza ","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet ","tvmonitor",
"laptop","mouse","remote ","keyboard ","cell phone","microwave ","oven ","toaster","sink","refrigerator ",
"book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "};

static int nboxes_0=GRID0*GRID0*nanchor;
static int nboxes_1=GRID1*GRID1*nanchor;
static int nboxes_2=GRID2*GRID2*nanchor;
static int nboxes_total=nboxes_0+nboxes_1+nboxes_2;

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
    }
    free(dets);
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int netw, int neth, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / netw;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / neth;
    return b;
}

void get_network_boxes(float *predictions, int netw,int neth,int GRID,int* masks, float* anchors, int box_off, detection* dets)
{
	int lw=GRID;
	int lh=GRID;
	int nboxes=GRID*GRID*nanchor;
	int LISTSIZE=1+4+nclasses;
	//darkent output排列格式: box顺序为先grid再anchor
	//1个anchor: 7*7*x+7*7*y+7*7*w+7*7*w+7*7*obj+7*7*classes1+7*7*classes2..+7*7*classes80,共3个anchor
	//x和y(先做,或者放在后面转换到实际坐标这一步中也行),以及obj和classes做logisic
	//xy做logistic
	for(int n=0;n<nanchor;n++){
		int index=n*lw*lh*LISTSIZE;
		int index_end=index+2*lw*lh;
		for(int i=index;i<index_end;i++)
			predictions[i]=1./(1.+exp(-predictions[i]));			
	}
	//类别和obj做logistic
	for(int n=0;n<nanchor;n++){
		int index=n*lw*lh*LISTSIZE+4*lw*lh;
		int index_end=index+(1+nclasses)*lw*lh;
		for(int i=index;i<index_end;i++){
			predictions[i]=1./(1.+exp(-predictions[i]));		
		}
	}
	//dets将outpus重排列,dets[i]为第i个框,box顺序为先anchor再grid

	int count=box_off;
	for(int i=0;i<lw*lh;i++){
		int row=i/lw;
		int col=i%lw;
		for(int n=0;n<nanchor;n++){
			int box_loc=n*lw*lh+i;  
			int box_index=n*lw*lh*LISTSIZE+i;            //box的x索引,ywh索引只要依次加上lw*lh
			int obj_index=box_index+4*lw*lh;
			float objectness=predictions[obj_index];
			if(objectness<OBJ_THRESH) continue;
			dets[count].objectness=objectness;
			dets[count].classes=nclasses;
			dets[count].bbox=get_yolo_box(predictions,anchors,masks[n],box_index,col,row,lw,lh,netw,neth,lw*lh);
			 for(int j=0;j<nclasses;j++){
			//for(int j=0;j<1;j++){
				int class_index=box_index+(5+j)*lw*lh;
				float prob=objectness*predictions[class_index];
				dets[count].prob[j]=prob;
				//cout<<j<<"==>"<<dets[count].prob[j]<<"\n";
			}
			++count;
		}
	}
	//cout<<"count: "<<count-box_off<<"\n";
    //return dets;
}

int outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height, detection* dets){
	float* output_0=(float*)rknn_outputs[0].buf;
	float* output_1=(float*)rknn_outputs[1].buf;
	float* output_2=(float*)rknn_outputs[2].buf;
	int masks_0[3] = {6, 7, 8};
	int masks_1[3] = {3, 4, 5};
	int masks_2[3] = {0, 1, 2};
	//float anchors[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
	float anchors[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
	//输出xywh均在0-1范围内
	get_network_boxes(output_0,net_width,net_height,GRID0,masks_0,anchors,0,dets);
	get_network_boxes(output_1,net_width,net_height,GRID1,masks_1,anchors,nboxes_0,dets);
	get_network_boxes(output_2,net_width,net_height,GRID2,masks_2,anchors,nboxes_0+nboxes_1,dets);
	return 0;
}

float overlap(float x1,float w1,float x2,float w2){
	float l1=x1-w1/2;
	float l2=x2-w2/2;
	float left=l1>l2? l1:l2;
	float r1=x1+w1/2;
	float r2=x2+w2/2;
	float right=r1<r2? r1:r2;
	return right-left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
int do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;
	//cout<<"total after OBJ_THRESH: "<<total<<"\n";

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
	return total;
}


int draw_image(cv::Mat img,detection* dets,int total,float thresh)
{
	//::cvtColor(img, img, cv::COLOR_RGB2BGR);
	for(int i=0;i<total;i++){
		char labelstr[4096]={0};
		int class_=-1;
		int topclass=-1;
		float topclass_score=0;
		if(dets[i].objectness==0) continue;
		for(int j=0;j<nclasses;j++){
			if(dets[i].prob[j]>thresh){
				if(topclass_score<dets[i].prob[j]){
					topclass_score=dets[i].prob[j];
					topclass=j;
				}
				if(class_<0){
					sprintf(labelstr,"%d",int(dets[i].prob[j]*100+0.5));
					// strcat(labelstr,labels[j].data());
					class_=j;
				}
				else{
					strcat(labelstr,",");
					sprintf(labelstr,"%d",int(dets[i].prob[j]*100+0.5));
					// strcat(labelstr,labels[j].data());
				}
				//printf("%s: %.02f%%\n",labels[j].data(),dets[i].prob[j]*100);
			}
		}
		//如果class>0说明框中有物体,需画框
		if(class_>=0){
			box b=dets[i].bbox;
			int x1 =(b.x-b.w/2.)*img.cols;
			int x2=(b.x+b.w/2.)*img.cols;
			int y1=(b.y-b.h/2.)*img.rows;
			int y2=(b.y+b.h/2.)*img.rows;

            if(x1  < 0) x1  = 0;
            if(x2> img.cols-1) x2 = img.cols-1;
            if(y1 < 0) y1 = 0;
			if(y2 > img.rows-1) y2 = img.rows-1;
			//std::cout << labels[topclass] << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

            rectangle(img, Point(x1, y1), Point(x2, y2), colorArray[class_%10], 2);
			//rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255, 255), 2);
            putText(img, labels[topclass]+":"+labelstr, Point(x1, y1 - 10), 2, 1, Scalar(0, 0, 255, 255),2,8);
			imwrite("./out.jpg", img);
            }
		}
	return 0;
}


