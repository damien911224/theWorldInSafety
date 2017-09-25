import cv2
import sys
sys.path.append("/home/damien/temporal-segment-networks/lib/caffe-action/python")
import caffe
import os
import random
import numpy as np
import argparse
from pipes import quote
from sklearn.metrics import confusion_matrix
from caffe.io import oversample
from utils.io import flow_stack_oversample, fast_list2arr
from utils.video_funcs import default_aggregation_func

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_path', type=str, default='/home/damien/temporal-segment-networks/mine/n11.avi')
parser.add_argument('--num_of_snippets', type=int, default=5)
args = parser.parse_args()

# extracted_frame_path = "/media/damien/DATA/cvData/TSN_data/UCF-101"
spatial_net_proto = "/home/damien/temporal-segment-networks/models/twis/tsn_bn_inception_rgb_deploy.prototxt"
spatial_net_weights = "/home/damien/temporal-segment-networks/models/twis_caffemodels/twis_split1_tsn_rgb_bn_inception_iter_3500.caffemodel"
temporal_net_proto = "/home/damien/temporal-segment-networks/models/twis/tsn_bn_inception_flow_deploy.prototxt"
temporal_net_weights = "/home/damien/temporal-segment-networks/models/twis_caffemodels/twis_split1_tsn_optical_flow_bn_inception_iter_18000.caffemodel"

class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        self._net = caffe.Net(net_proto, net_weights, caffe.TEST)

        input_shape = self._net.blobs['data'].data.shape

        if input_size is not None:
            input_shape = input_shape[:2] + input_size

        transformer = caffe.io.Transformer({'data': input_shape})

        if self._net.blobs['data'].data.shape[1] == 3:
            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        else:
            pass # non RGB data need not use transformer

        self._transformer = transformer

        self._sample_shape = self._net.blobs['data'].data.shape

    def predict_single_frame(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size, interpolation=cv2.INTER_AREA) for x in frame]

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size,  interpolation=cv2.INTER_AREA) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

def makeInputFromVideo():
    input_video_path = args.input_video_path
    num_of_snippets = args.num_of_snippets

    video_cap = cv2.VideoCapture(input_video_path)
    fcount = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frames = []
    rgb_selected_frames = []
    flow_selected_frames = []

    for i in xrange(num_of_snippets):
        start_frames.append(int(fcount/num_of_snippets*i))

    for i in xrange(num_of_snippets):
        if i != num_of_snippets -1:
            rgb_selected_frames.append(random.sample(range(start_frames[i], start_frames[i+1]-1), 1)[0])
            flow_selected_frames.append(random.sample(range(start_frames[i], start_frames[i+1]-11),1)[0])
        else:
            rgb_selected_frames.append(random.sample(range(start_frames[i], fcount), 1)[0])
            flow_selected_frames.append(random.sample(range(start_frames[i], fcount -10), 1)[0])

    print fcount
    print start_frames
    print rgb_selected_frames
    print flow_selected_frames

    output_path = "./data/" + input_video_path.split('/')[-1].split('.')[-2]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in xrange(num_of_snippets):
        video_cap.set(1, rgb_selected_frames[i])
        ret, frame = video_cap.read()
        if ret:
            cv2.imwrite(output_path + "/rgb_0%d.jpg" %(i+1), frame)

def build_net():
    global spatial_net
    global temporal_net
    device_id = 0

    spatial_net = CaffeNet(spatial_net_proto, spatial_net_weights, device_id)
    temporal_net = CaffeNet(temporal_net_proto, temporal_net_weights, device_id)

def eval_video():
    global spatial_net
    global temporal_net

    build_net()

    video_file_path = args.input_video_path
    video_name = video_file_path.split('/')[-1]

    video_cap = cv2.VideoCapture(video_file_path)
    frame_cnt = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    num_frame_per_video = int(video_cap.get(cv2.CAP_PROP_FPS))

    rgb_prefix = "img_"
    flow_x_prefix = "flow_x_"
    flow_y_prefix = "flow_y_"
    out_format = "dir"
    ext = "avi"
    new_size = (340, 256)
    df_path = "/home/damien/temporal-segment-networks/lib/dense_flow/"
    out_path = "/home/damien/temporal-segment-networks/mine/processed_input_data"
    class_info_path = "/home/damien/temporal-segment-networks/data/twis_splits/classInd.txt"

    out_full_path = os.path.join(out_path, video_name.split('.')[-2])
    image_full_path = out_full_path + "/images"
    optical_flow_full_path = out_full_path + "/optical_flow"
    warped_flow_full_path = out_full_path + "/warped_flow"
    score_layer_name = 'fc-twis'

    class_info_fp = open(class_info_path, "r")
    class_label = { }
    while True:
        line = class_info_fp.readline()
        if not line:
            break
        class_idx = line.split(' ')[-2]
        class_name = line.split(' ')[-1][:-2]
        class_label[class_idx] = class_name
    class_info_fp.close()

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass

    dev_id = 0
    image_path = '{}/images/img'.format(out_full_path)
    optical_flow_x_path = '{}/optical_flow/flow_x'.format(out_full_path)
    optical_flow_y_path = '{}/optical_flow/flow_y'.format(out_full_path)
    warped_flow_x_path = '{}/warped_flow/flow_x'.format(out_full_path)
    warped_flow_y_path = '{}/warped_flow/flow_y'.format(out_full_path)

    if not os.path.exists(image_full_path):
        try:
            os.mkdir(image_full_path)
        except OSError:
            pass

    if not os.path.exists(optical_flow_full_path):
        try:
            os.mkdir(optical_flow_full_path)
        except OSError:
            pass

    if not os.path.exists(warped_flow_full_path):
        try:
            os.mkdir(warped_flow_full_path)
        except OSError:
            pass

    cmd = os.path.join(
        df_path + 'build/extract_gpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path), dev_id, out_format,
        new_size[0],
        new_size[1])

    os.system(cmd)

    sys.stdout.flush()

    cmd = os.path.join(df_path + 'build/extract_warp_gpu') + ' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        quote(video_file_path), quote(warped_flow_x_path), quote(warped_flow_y_path), dev_id, out_format)

    os.system(cmd)

    print '{} done'.format(video_name)
    sys.stdout.flush()

    rgb_stack_depth = 1
    flow_stack_depth = 5

    rgb_step = (frame_cnt - rgb_stack_depth) / (num_frame_per_video-1)
    if rgb_step > 0:
        rgb_frame_ticks = range(1, min((2 + rgb_step * (num_frame_per_video-1)), frame_cnt+1), rgb_step)
    else:
        rgb_frame_ticks = [1] * num_frame_per_video

    assert(len(rgb_frame_ticks) == num_frame_per_video)

    flow_step = (frame_cnt - flow_stack_depth) / (num_frame_per_video - 1)
    if flow_step > 0:
        flow_frame_ticks = range(1, min((2 + flow_step * (num_frame_per_video - 1)), frame_cnt + 1), flow_step)
    else:
        flow_frame_ticks = [1] * num_frame_per_video

    assert (len(flow_frame_ticks) == num_frame_per_video)

    rgb_frame_scores = []
    for tick in flow_frame_ticks:
        name = '{}{:05d}.jpg'.format(rgb_prefix, tick)
        frame = cv2.imread(os.path.join(image_full_path, name), cv2.IMREAD_COLOR)
        scores = spatial_net.predict_single_frame([frame,], score_layer_name, frame_size=(340, 240))
        rgb_frame_scores.append(scores)

    flow_frame_scores = []
    for tick in flow_frame_ticks:
        frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(flow_stack_depth)]
        flow_stack = []
        for idx in frame_idx:
            x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
            y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
            x_flow_field = cv2.imread(os.path.join(warped_flow_full_path, x_name), cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(os.path.join(warped_flow_full_path, y_name), cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)

        scores = temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, frame_size=(340, 240))
        flow_frame_scores.append(scores)

    print 'video {} test done'.format(video_name)
    rgb_prediction = np.argmax(default_aggregation_func(rgb_frame_scores)) + 1
    flow_prediction = np.argmax(default_aggregation_func(flow_frame_scores)) + 1

    whole_scores = [x*1.0 + y*2.0 for x in rgb_frame_scores for y in flow_frame_scores]

    whole_prediction = np.argmax(default_aggregation_func(whole_scores)) + 1

    print 'prediction from spatial net: ' + class_label['%d' %rgb_prediction]
    print 'prediction from temporal net: '+ class_label['%d' %flow_prediction]
    print 'prediction from whole net: ' + class_label['%d' %whole_prediction]

    sys.stdin.flush()

def test_net(flow_type= "opticalFlow", split=1):
    test_video_name_list = []
    test_video_label_list = []
    label_and_index = []
    class_ind_fp = open("/home/damien/temporal-segment-networks/data/twis_splits/classInd.txt", "r")
    while True:
        line = class_ind_fp.readline()
        if not line:
            break

        label = line.split(' ')[-1][:-2]
        index = line.split(' ')[-2]

        label_and_index.append((label, index))

    class_ind_fp.close()

    split_fp = open("/home/damien/temporal-segment-networks/data/twis_splits/testlist0%d.txt" % split, "r")
    while True:
        line = split_fp.readline()
        if not line:
            break

        video_name = line.split('/')[-1][:-1]
        video_label_name = line.split('/')[-2]
        video_label_index = 0
        for ii in xrange(len(label_and_index)):
            if label_and_index[ii][0] == video_label_name:
                video_label_index = label_and_index[ii][1]
                break

        test_video_name_list.append(video_name)
        test_video_label_list.append(video_label_index)

        print video_name
        print video_label_index

    split_fp.close()

eval_video()