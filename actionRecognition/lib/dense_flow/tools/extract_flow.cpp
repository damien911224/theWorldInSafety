#include "dense_flow.h"
#include "utils.h"
INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
	const char* keys =
		{
			"{ v  | vidFile      | ex2.avi | filename of video }"
			"{ f  | frame_prefix | ex2    | frame_prefix from frames }"
			"{ p  | start_index  | 2      | frame_index for per frame operation }"
			"{ e  | end_index    | 30     | frame_index for per frame operation }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ c  | frame_count | 0 | frame count }"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
			"{ o  | out | zip | output style}"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	string frame_prefix_string = cmd.get<string>("frame_prefix");
	const char* frame_prefix = frame_prefix_string.c_str();
	string output_style = cmd.get<string>("out");
	int bound = cmd.get<int>("bound");
	int frame_count = cmd.get<int>("frame_count");
	int start_index = cmd.get<int>("start_index");
	int end_index = cmd.get<int>("end_index");

	vector<vector<uchar> > out_vec_x, out_vec_y, out_vec_img;

	calcDenseFlowCPUFromFrames(frame_prefix, start_index, end_index, bound, 0, 1, out_vec_x, out_vec_y, out_vec_img);

	writeImages(out_vec_x, xFlowFile);
	writeImages(out_vec_y, yFlowFile);
	
	return 0;
}
