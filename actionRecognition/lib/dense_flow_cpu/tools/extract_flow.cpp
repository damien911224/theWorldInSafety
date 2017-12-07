#include "dense_flow.h"
#include "utils.h"
INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
	const char* keys =
		{
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

        const char* frame_prefix = argv[1];
	string xFlowFile = argv[2];
	string yFlowFile = argv[3];
	string output_style = argv[4];
        int bound = atoi(argv[5]);
	int frame_count = atoi(argv[6]);
	int start_index = atoi(argv[7]);
	int end_index = atoi(argv[8]);

	vector<vector<uchar> > out_vec_x, out_vec_y, out_vec_img;

	calcDenseFlowCPUFromFrames(frame_prefix, start_index, end_index, bound, 0, 1, out_vec_x, out_vec_y, out_vec_img);

	writeImageIndex(out_vec_x, xFlowFile, start_index, end_index);
	writeImageIndex(out_vec_y, yFlowFile, start_index, end_index);
	
	return 0;
}
