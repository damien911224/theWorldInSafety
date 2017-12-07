//
// Created by yjxiong on 11/18/15.
//
#include "dense_flow.h"
#include "stdio.h"
#include "opencv2/gpu/gpu.hpp"
using namespace cv::gpu;


void calcDenseFlowGPUPerFrame(const char* frame_prefix, int start_index, int end_index, int bound, int type, int step, int dev_id,
                      vector<vector<uchar> >& output_x,
                      vector<vector<uchar> >& output_y,
                      vector<vector<uchar> >& output_img,
                      int new_width, int new_height){
    setDevice(dev_id);
    Mat first_capture_frame, second_capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow_x, flow_y;
    Size new_size(new_width, new_height);

    GpuMat d_frame_0, d_frame_1;
    GpuMat d_flow_x, d_flow_y;

    OpticalFlowDual_TVL1_GPU alg_tvl1;

	for(int index=start_index;index<=end_index;index++) {
		if(index == start_index) {
	        char first_frame_path[1000];
	        sprintf(first_frame_path, "%s_%07d.jpg", frame_prefix, index-1);
            first_capture_frame = imread(first_frame_path);
            first_capture_frame.copyTo(prev_image);
	        cvtColor(prev_image, prev_gray, CV_BGR2GRAY);
		}

	    char second_frame_path[1000];
  	    sprintf(second_frame_path, "%s_%07d.jpg", frame_prefix, index);
        second_capture_frame = imread(second_frame_path);
	    second_capture_frame.copyTo(capture_image);
            
        cvtColor(capture_image, capture_gray, CV_BGR2GRAY);
        
	    d_frame_0.upload(prev_gray);
        d_frame_1.upload(capture_gray);

        alg_tvl1(d_frame_0, d_frame_1, d_flow_x, d_flow_y);
            
	    //get back flow map
        d_flow_x.download(flow_x);
        d_flow_y.download(flow_y);
   
        vector<uchar> str_x, str_y, str_img;
        encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);

        output_x.push_back(str_x);
        output_y.push_back(str_y);

		std::swap(prev_gray, capture_gray);
		std::swap(prev_image, capture_image);
	}
}
