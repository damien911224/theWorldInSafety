#include "common.h"
#include "dense_flow.h"

using namespace std;

void calcDenseFlowCPUFromFrames(const char* frame_prefix, int start_index, int end_index, int bound, int type, int step,
                   vector<vector<uchar> >& output_x,
                   vector<vector<uchar> >& output_y,
                   vector<vector<uchar> >& output_img){

    Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow, flow_split[2];


    bool initialized = false;
    for(int index=start_index;index<=end_index;index++){
        //build mats for the first frame
        if (!initialized){
            char first_frame_path[1000];
            sprintf(first_frame_path, "%s_%07d.jpg", frame_prefix, index-1);
            capture_frame = imread(first_frame_path);
            initializeMats(capture_frame, capture_image, capture_gray, prev_image, prev_gray);
            capture_frame.copyTo(prev_image);
            cvtColor(prev_image, prev_gray, CV_BGR2GRAY);
            initialized = true;
        }
            
	char frame_path[1000];
        sprintf(frame_path, "%s_%07d.jpg", frame_prefix, index);
        capture_frame = imread(frame_path);
        capture_frame.copyTo(capture_image);
        cvtColor(capture_image, capture_gray, CV_BGR2GRAY);
        calcOpticalFlowFarneback(prev_gray, capture_gray, flow,
                                 0.702, 5, 10, 2, 7, 1.5,
                                 cv::OPTFLOW_FARNEBACK_GAUSSIAN );

        vector<uchar> str_x, str_y, str_img;
        split(flow, flow_split);
        encodeFlowMap(flow_split[0], flow_split[1], str_x, str_y, bound);
        imencode(".jpg", capture_image, str_img);

        output_x.push_back(str_x);
        output_y.push_back(str_y);
        output_img.push_back(str_img);

        std::swap(prev_gray, capture_gray);
        std::swap(prev_image, capture_image);    
    }
}
