#include "dense_flow.h"
#include "utils.h"

INITIALIZE_EASYLOGGINGPP

using namespace cv::gpu;

int main(int argc, char** argv){
	// IO operation
	const char* keys =
		{
			"{ v  | vidFile      | ex2.avi | filename of video }"
			"{ f  | frame_prefix | ex2    | frame_prefix from frames }"
			"{ p  | start_index  | 2      | frame_index for per frame operation }"
			"{ e  | end_index    | 30     | frame_index for per frame operation }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
			"{ t  | type | 0 | specify the optical flow algorithm }"
			"{ d  | device_id    | 0  | set gpu id}"
			"{ s  | step  | 1 | specify the step for frame sampling}"
			"{ o  | out | zip | output style}"
			"{ w  | newWidth | 0 | output style}"
			"{ h  | newHeight | 0 | output style}"
			"{ a  | dump         | -1     | whether to dump frames }"
			"{ c  | frame_count | 0 | frame count }"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string frame_prefix_string = cmd.get<string>("frame_prefix");
	const char* frame_prefix = frame_prefix_string.c_str();
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	string output_style = cmd.get<string>("out");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");
    int new_height = cmd.get<int>("newHeight");
    int new_width = cmd.get<int>("newWidth");
	int dump = cmd.get<int>("dump");
	int frame_count = cmd.get<int>("frame_count");
	int start_index = cmd.get<int>("start_index");
	int end_index = cmd.get<int>("end_index");

	vector<vector<uchar> > out_vec_x, out_vec_y, out_vec_img;

	if (vidFile == "None") {
	    calcDenseFlowGPUFromFrames(frame_prefix, frame_count, bound, type, step, device_id,
		  out_vec_x, out_vec_y, out_vec_img, new_width, new_height);
	}
	else if (start_index == -1) {
        calcDenseFlowGPU(vidFile, bound, type, step, device_id,
					 out_vec_x, out_vec_y, out_vec_img, new_width, new_height);
	}
	else {
	    calcDenseFlowGPUPerFrame(frame_prefix, start_index, end_index, bound, type, step, device_id,
		  out_vec_x, out_vec_y, out_vec_img, new_width, new_height);
	}

	if (output_style == "dir") {
	    if ( start_index == -1 ) {
		    writeImages(out_vec_x, xFlowFile);
		    writeImages(out_vec_y, yFlowFile);
		}
		else {
		    writeImageIndex(out_vec_x, xFlowFile, start_index, end_index);
			writeImageIndex(out_vec_y, yFlowFile, start_index, end_index);
		}
		if (dump == 1) {
		    writeImages(out_vec_img, imgFile);
		}
	}else{
//		LOG(INFO)<<"Writing results to Zip archives";
		writeZipFile(out_vec_x, "x_%05d.jpg", xFlowFile+".zip");
		writeZipFile(out_vec_y, "y_%05d.jpg", yFlowFile+".zip");
		if (dump == 1) {
		    writeZipFile(out_vec_img, "img_%05d.jpg", imgFile+".zip");
		}
	}

	return 0;
}
