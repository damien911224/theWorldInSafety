//
// Created by yjxiong on 11/18/15.
//
#include "dense_flow.h"
#include "stdio.h"
#include "opencv2/gpu/gpu.hpp"
using namespace cv::gpu;


void calcDenseFlowGPUFromFrames(const char* frame_prefix, int frame_count,int bound, int type, int step, int dev_id,
                      vector<vector<uchar> >& output_x,
                      vector<vector<uchar> >& output_y,
                      vector<vector<uchar> >& output_img,
                      int new_width, int new_height){
    setDevice(dev_id);
    Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow_x, flow_y;
    Size new_size(new_width, new_height);

    GpuMat d_frame_0, d_frame_1;
    GpuMat d_flow_x, d_flow_y;

    OpticalFlowDual_TVL1_GPU alg_tvl1;
    BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

    bool do_resize = (new_height > 0) && (new_width > 0);

    bool initialized = false;
    int cnt = 1;

    while(true){
        //build mats for the first frame
        if (!initialized){
		   char first_frame_path[1000];
           sprintf(first_frame_path, "%s_%05d.jpg", frame_prefix, cnt++);

           capture_frame = imread(first_frame_path);
           if (capture_frame.empty()) return; // read frames until end

            if (!do_resize){
                initializeMats(capture_frame, capture_image, capture_gray,
                           prev_image, prev_gray);
                capture_frame.copyTo(prev_image);
            }else{
                capture_image.create(new_size, CV_8UC3);
                capture_gray.create(new_size, CV_8UC1);
                prev_image.create(new_size, CV_8UC3);
                prev_gray.create(new_size, CV_8UC1);
                cv::resize(capture_frame, prev_image, new_size);
            }
            cvtColor(prev_image, prev_gray, CV_BGR2GRAY);
            initialized = true;
        }else {
		    char frame_path[1000];
			sprintf(frame_path, "%s_%05d.jpg", frame_prefix, cnt++);
            capture_frame = imread(frame_path);

            if (!do_resize) {
                capture_frame.copyTo(capture_image);
			}
            else {
                cv::resize(capture_frame, capture_image, new_size);
			}
            
            cvtColor(capture_image, capture_gray, CV_BGR2GRAY);
            d_frame_0.upload(prev_gray);
            d_frame_1.upload(capture_gray);

            switch(type){
                case 0: {
                    break;
                }
                case 1: {
                    alg_tvl1(d_frame_0, d_frame_1, d_flow_x, d_flow_y);
                    break;
                }
                case 2: {
                    GpuMat d_buf_0, d_buf_1;
                    d_frame_0.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
                    d_frame_1.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
                    alg_brox(d_buf_0, d_buf_1, d_flow_x, d_flow_y);
                    break;
                }
                default:
                    LOG(ERROR)<<"Unknown optical method: "<<type;
            }
            
			//get back flow map
            d_flow_x.download(flow_x);
            d_flow_y.download(flow_y);

            vector<uchar> str_x, str_y, str_img;
            encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
            imencode(".jpg", capture_image, str_img);

            output_x.push_back(str_x);
            output_y.push_back(str_y);
            output_img.push_back(str_img);

            std::swap(prev_gray, capture_gray);
            std::swap(prev_image, capture_image);

            if (cnt > frame_count){
                return;
            }
        }


    }

}
