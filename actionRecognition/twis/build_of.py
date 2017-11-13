import sys
sys.path.insert(0, "../lib/caffe-action/python")
import caffe
sys.path.append('..')
print sys.path
from pyActionRecog import parse_directory, build_split_list
from pyActionRecog import parse_split_file
import os
import glob
import sys
import random
import numpy as np
import time
import cv2
import math
import shutil
import string
from pipes import quote
from multiprocessing import Pool, Value, Lock, current_process, Manager
from ctypes import c_int, c_float
from caffe.io import oversample
from utils.io import flow_stack_oversample, fast_list2arr
from utils.video_funcs import default_aggregation_func

class CaffeNet(object):
    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        if device_id >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        else:
            caffe.set_mode_cpu()
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


    def predict_simple_single_flow_stack(self, frame, score_name, over_sample=False, frame_size=None):
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


def build_net():
    global spatial_net_gpu
    global temporal_net_gpu
    global spatial_net_cpu
    global temporal_net_cpu
    spatial_net_proto = "../temporal-segment-networks/models/twis/tsn_bn_inception_rgb_deploy.prototxt"
    spatial_net_weights = "../models/twis_caffemodels/v1/twis_spatial_net_v1.caffemodel"
    temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
    temporal_net_weights = "../models/twis_caffemodels/v1/twis_temporal_net_v1.caffemodel"

    device_id = 0

    spatial_net_gpu = CaffeNet(spatial_net_proto, spatial_net_weights, device_id)
    temporal_net_gpu = CaffeNet(temporal_net_proto, temporal_net_weights, device_id)

    spatial_net_cpu = CaffeNet(spatial_net_proto, spatial_net_weights, -1)
    temporal_net_cpu = CaffeNet(temporal_net_proto, temporal_net_weights, -1)


def showProgress(progress_type='', file_name='', message=' DONE', process_start_time=0.0, process_id=0):
    global len_video_list
    global num_counter
    global time_counter
    global counter_lock
    global global_start_time
    global global_num_using_gpu
    global global_num_worker
    global scan_counter
    global scan_counter_lock
    global len_scan_list
    global global_clipping_remaining_hours
    global global_clipping_avg_duration
    global global_clipping_one_duration

    if progress_type == 'Extracting':
        with counter_lock:
            num_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            average_duration = time_counter.value / float(num_counter.value) / 3600.0
            remaining_hours = float(len_video_list - num_counter.value) * average_duration
            if process_id % global_num_worker < global_num_using_gpu:
                process_type = 'GPU'
            else:
                process_type = 'CPU'

            print \
            "{0}|{5:05d}th|{1:06.3f}%|Remaining: {2:.2f}Hours|Current: {3} {7} Worker|AvgDuration: {6:.2f}Secs|OneDuration: {4:.2f}Secs\n".format(progress_type,
            float(num_counter.value)/float(len_video_list)*100.0, remaining_hours,
            process_id%global_num_worker+1, elapsed_time_per_process, num_counter.value, average_duration * 3600.0, process_type) + \
            " " * (len(progress_type)+16) + "|FileName: " + file_name + message
    elif progress_type == 'Clipping':
        with counter_lock:
            num_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            average_duration = time_counter.value / float(num_counter.value) / 3600.0
            remaining_hours = float(len_video_list - num_counter.value) * average_duration
            if process_id % global_num_worker < global_num_using_gpu:
                process_type = 'GPU'
            else:
                process_type = 'CPU'

            print \
                "{0}|{1:05d}th|{2:06.3f}%|Remaining: {3:.2f}Hours|AvgDuration: {4:.2f}Secs|OneDuration: {5:.2f}Secs\n".format(
                    progress_type, num_counter.value,
                    float(num_counter.value) / float(len_video_list) * 100.0, remaining_hours, average_duration * 3600.0,
                    elapsed_time_per_process) + \
                " " * (len(progress_type) + 16) + "|FileName: " + file_name + message
    elif progress_type == 'Scanning':
        with scan_counter_lock:
            scan_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            if process_id % global_num_worker < global_num_using_gpu:
                process_type = 'GPU'
            else:
                process_type = 'CPU'

            if scan_counter.value % 50 == 0:
                print \
                    "{0}|{1:05d}th|{2:06.3f}%|Remaining: {3:.2f}Hours|AvgDuration: {4:.2f}Secs|OneDuration: {5:.2f}Sec\n" \
                    "{6}|{7:05d}th|{8:06.3f}%|Current: {9} {10} Worker|OneDuration: {11:.2f}Secs\n".format(
                        'Clipping', num_counter.value, float(num_counter.value) / float(len_video_list) * 100.0,
                        global_clipping_remaining_hours, global_clipping_avg_duration*3600, global_clipping_one_duration,
                        progress_type, scan_counter.value,
                        scan_counter.value / float(len_scan_list) * 100.0,
                        process_id % global_num_worker + 1, process_type, elapsed_time_per_process) + \
                    " " * (len(progress_type) + 16) + "|FileName: " + file_name + message


def extractFrames(vid_path, out_full_path):
    import cv2
    video = cv2.VideoCapture(vid_path)

    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}_{:05d}.jpg'.format(out_full_path, i+1), frame)


def extractOpticalFlowClips(vid_item):
    start_time = time.time()

    vid_path = vid_item[0]

    clips = clippingVideos(vid_path)

    for clip in clips:
        vid_name = clip.split('/')[-1].split('.')[0]

        new_size = (340, 256)
        out_format = 'dir'
        df_path = "~/temporal-segment-networks/lib/dense_flow/"
        out_full_path = os.path.join(save_path, vid_name)
        current = current_process()
        current_id = int(current._identity[0]) - 1
        dev_id = 0

        if current_id % global_num_worker < global_num_using_gpu:
            if os.path.exists(out_full_path):
                files = glob.glob(out_full_path + '/*')
                for f in files:
                    os.remove(f)
                os.removedirs(out_full_path)
                # message = ' !! PASS !!'
                # showProgress(progress_type='Extracting Optical Flow', file_name=out_full_path, message=message,
                #              process_start_time=start_time, process_id=current_id)
                # return

            try:
                os.mkdir(out_full_path)
            except OSError:
                pass

            image_path = '{}/img'.format(out_full_path)
            flow_x_path = '{}/flow_x'.format(out_full_path)
            flow_y_path = '{}/flow_y'.format(out_full_path)

            cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
                quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])

        else:
            if os.path.exists(out_full_path):
                files = glob.glob(out_full_path + '/*')
                for f in files:
                    os.remove(f)
                os.removedirs(out_full_path)
                # message = ' !! PASS !!'
                # showProgress(progress_type='Extracting Optical Flow', file_name=out_full_path, message=message,
                #              process_start_time=start_time, process_id=current_id)
                # return

            try:
                os.mkdir(out_full_path)
            except OSError:
                pass

            image_path = '{}/img'.format(out_full_path)
            flow_x_path = '{}/flow_x'.format(out_full_path)
            flow_y_path = '{}/flow_y'.format(out_full_path)

            cmd = os.path.join(
                df_path + 'build/extract_cpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -s 1 -o {}'.format(
                quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path),out_format)

        os.system(cmd)
        sys.stdout.flush()

    message = ' DONE'
    showProgress(progress_type='Extracting Optical Flow', file_name=out_full_path, message=message,
                 process_start_time=start_time, process_id=current_id)

    return


def extractOpticalFlowSimple(extractItems):
    video_file_path = extractItems[0]
    extract_dst_path = extractItems[1]
    processor_type = extractItems[2]

    global num_extract
    global num_counter_lock
    global len_extract

    video_name = video_file_path.split('/')[-1]

    new_size = (340, 256)
    out_format = 'dir'
    df_path = "../lib/dense_flow/"
    dev_id = 0

    out_full_path = os.path.join(extract_dst_path, video_name.split('.')[-2])
    image_full_path = out_full_path + "/images"
    optical_flow_full_path = out_full_path + "/optical_flow"

    video_cap = cv2.VideoCapture(video_file_path)
    if video_cap.isOpened():
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        print '!! VIDEO CAP ERROR !!'
        return
    video_cap.release()

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass
    else:
        file_count = len(glob.glob(out_full_path + '/images/*'))
        if file_count < frame_count - 1:
            images = glob.glob(out_full_path + '/images/*')
            flows = glob.glob(out_full_path + '/optical_flow/*')
            if len(images) != 0:
                for image_file in images:
                    try:
                        os.remove(image_file)
                    except OSError:
                        pass
            if len(flows) != 0:
                for flow_file in flows:
                    try:
                        os.remove(flow_file)
                    except OSError:
                        pass
            try:
                os.removedirs(out_full_path + '/images')
            except OSError:
                pass
            try:
                os.removedirs(out_full_path + '/optical_flow')
            except OSError:
                pass
            try:
                os.removedirs(out_full_path)
            except OSError:
                pass
        else:
            with num_counter_lock:
                num_extract.value += 1
                print 'EXTRACT PASS|{:05.02f}%|{:05d}th|{}'.format(
                    float(num_extract.value) / float(len_extract) * 100.0, num_extract.value, video_file_path)
            return

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass

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

    image_path = '{}/images/img'.format(out_full_path)
    optical_flow_x_path = '{}/optical_flow/flow_x'.format(out_full_path)
    optical_flow_y_path = '{}/optical_flow/flow_y'.format(out_full_path)

    if processor_type == 'gpu':
        cmd = os.path.join(
            df_path + 'build/extract_gpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
            quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path),
            dev_id,
            out_format, new_size[0], new_size[1])
    else:
        cmd = os.path.join(
            df_path + 'build/extract_cpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -s 1 -o {} -w {} -h {}'.format(
            quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path),
            out_format, new_size[0], new_size[1])

    os.system(cmd)
    sys.stdout.flush()

    with num_counter_lock:
        num_extract.value += 1
        print 'EXTRACT DONE|{:05.02f}%|{:05d}th|{}'.format(float(num_extract.value)/float(len_extract)*100.0, num_extract.value, video_file_path)


def extractOpticalFlow(video_file_path):
    global global_extract_dst_path

    start_time = time.time()
    video_name = video_file_path.split('/')[-1]

    current = current_process()
    current_id = current._identity[0] - 1

    new_size = (340, 256)
    out_format = 'dir'
    df_path = "~/temporal-segment-networks/lib/dense_flow/"
    dev_id = 0

    out_full_path = os.path.join(global_extract_dst_path, video_name.split('.')[-2])
    image_full_path = out_full_path + "/images"
    optical_flow_full_path = out_full_path + "/optical_flow"

    video_cap = cv2.VideoCapture(video_file_path)
    if video_cap.isOpened():
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        vid_list.remove(video_file_path)
        message = ' !! VIDEO CAP ERROR !!'
        showProgress(progress_type='Extracting', file_name=video_file_path, message=message,
                     process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return
    video_cap.release()

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass
    else:
        image_count = len(glob.glob(out_full_path +'/images/*'))
        flow_count = len(glob.glob(out_full_path +'/flows/*'))
        if image_count < frame_count -2 or flow_count / 2 < frame_count -2:
            images = glob.glob(out_full_path + '/images/*')
            flows = glob.glob(out_full_path + '/optical_flow/*')
            if len(images) != 0:
                for image_file in images:
                    try:
                        os.remove(image_file)
                    except OSError:
                        pass
            if len(flows) != 0:
                for flow_file in flows:
                    try:
                        os.remove(flow_file)
                    except OSError:
                        pass
            try:
                os.removedirs(out_full_path + '/images')
            except OSError:
                pass
            try:
                os.removedirs(out_full_path + '/optical_flow')
            except OSError:
                pass
            try:
                os.removedirs(out_full_path)
            except OSError:
                pass
        else:
            message = ' !! PASS !!'
            showProgress(progress_type='Extracting', file_name=video_file_path, message=message,
                         process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            return

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass

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

    image_path = '{}/images/img'.format(out_full_path)
    optical_flow_x_path = '{}/optical_flow/flow_x'.format(out_full_path)
    optical_flow_y_path = '{}/optical_flow/flow_y'.format(out_full_path)

    if current_id % global_num_worker < global_num_using_gpu:
        cmd = os.path.join(
            df_path + 'build/extract_gpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
            quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path), dev_id,
            out_format, new_size[0], new_size[1])
    else:
        cmd = os.path.join(
            df_path + 'build/extract_cpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -s 1 -o {} -w {} -h {}'.format(
            quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path),
            out_format, new_size[0], new_size[1])

    os.system(cmd)
    sys.stdout.flush()

    message = ' DONE'
    showProgress(progress_type='Extracting', file_name=video_file_path, message=message,
                 process_start_time=start_time,
                 process_id=int(current._identity[0] - 1))


def extractWarpedFlowClips(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]

    new_size = (340, 256)
    out_format = 'zip'
    df_path = "~/temporal-segment-networks/lib/dense_flow/"
    out_path = "/media/damien/DATA/cvData/TSN_data/TWIS/warpedFlow"
    out_full_path = os.path.join(out_path, vid_name)
    current = current_process()
    current_id = int(current._identity[0])-1

    if os.path.exists(out_full_path):
        message = ' !! PASS !!'
        showProgress(progress_type='Extracting Warped Flow', file_name=video_full_path, message=message,
                     process_start_time=start_time, process_id=current_id)
        return

    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), dev_id, out_format)

    os.system(cmd)
    print 'warp on {} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return True


def extractWarpedFlow(video_file_path, this_save_path):
    video_name = video_file_path.split('/')[-1]

    new_size = (340, 256)
    out_format = 'dir'
    df_path = "~/temporal-segment-networks/lib/dense_flow/"
    dev_id = 0

    out_full_path = os.path.join(this_save_path, video_name.split('.')[-2])
    image_full_path = out_full_path + "/images"
    warped_flow_full_path = out_full_path + "/warped_flow"

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass
    else:
        return

    if not os.path.exists(image_full_path):
        try:
            os.mkdir(image_full_path)
        except OSError:
            pass

    if not os.path.exists(warped_flow_full_path):
        try:
            os.mkdir(warped_flow_full_path)
        except OSError:
            pass

    image_path = '{}/images/img'.format(out_full_path)
    warped_flow_x_path = '{}/warped_flow/flow_x'.format(out_full_path)
    warped_flow_y_path = '{}/warped_flow/flow_y'.format(out_full_path)

    extractFrames(video_file_path, image_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu') + ' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        quote(video_file_path), quote(warped_flow_x_path), quote(warped_flow_y_path), dev_id, out_format)

    os.system(cmd)
    sys.stdout.flush()


def clipping(video_path, save_path, start_frame, end_frame):
    video_cap = cv2.VideoCapture(video_path)

    if video_cap.isOpened():
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_writer = cv2.VideoWriter(save_path, video_fourcc, video_fps, (video_width, video_height))
        f_counter = start_frame
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            video_writer.write(frame)
            if f_counter == end_frame:
                break
            f_counter += 1

        video_writer.release()

    video_cap.release()


def clippingVideos(video_path, savedAsFiles=True):
    start_time = time.time()

    global global_clip_dst_path
    global global_frame_src_path
    global global_frame_dst_path
    global global_whole_scores
    global global_num_worker
    global len_scan_list
    global global_num_using_gpu
    global scan_counter
    global whole_scores
    global global_clipping_remaining_hours
    global global_clipping_avg_duration
    global global_clipping_one_duration

    manager = Manager()
    whole_scores = manager.list()

    check_clip_path = os.path.join(global_clip_dst_path, video_path.split('/')[-1].split('.')[-2])
    if os.path.exists(check_clip_path):
        num_counter.value += 1
        current_time = time.time()
        global_clipping_one_duration = current_time - start_time
        whole_elapsed_time = current_time - global_start_time
        time_counter.value += whole_elapsed_time / float(num_counter.value)
        global_clipping_avg_duration = time_counter.value / float(num_counter.value) / 3600.0
        global_clipping_remaining_hours = float(len_video_list - num_counter.value) * global_clipping_avg_duration
        return

    scan_input_path = os.path.join(global_frame_src_path, video_path.split('/')[-1].split('.')[-2])
    scan_counter.value = 0

    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened():
        video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
        video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        check_counter = glob.glob(scan_input_path + '/images/*')
        if len(check_counter) < video_frame_count:
            num_counter.value += 1
            current_time = time.time()
            global_clipping_one_duration = current_time - start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            global_clipping_avg_duration = time_counter.value / float(num_counter.value) / 3600.0
            global_clipping_remaining_hours = float(len_video_list - num_counter.value) * global_clipping_avg_duration
            return

        for i in xrange(video_frame_count):
            whole_scores.append([0.0, 0.0])

        scan_input_paths = []
        for i in xrange(video_frame_count+1):
            input_path = scan_input_path
            scan_input_paths.append(input_path)

        len_scan_list = len(scan_input_paths)
        scan_pool = Pool(global_num_worker)
        scan_pool.map(scanVideo, zip(scan_input_paths, [video_frame_count]*video_frame_count, range(1, video_frame_count+1, 1)))
        scan_pool.close()

        whole_score_sum = [ 0.0 ] * len(whole_scores[0])
        for score in whole_scores:
            for i in xrange(len(whole_scores[0])):
                whole_score_sum[i] += score[i]
        whole_average_score = [whole_score_sum[i] / float(len(whole_scores)) for i in xrange(len(whole_scores[0]))]
        whole_dominant_score = max(whole_average_score)
        whole_dominant_index = np.argmax(whole_average_score)
        good_frame_indices = []

        for ii in xrange(len(whole_scores)):
            if whole_scores[ii][whole_dominant_index] >= whole_dominant_score:
                good_frame_indices.append(ii+1)
    
        selected_scores = []
        for index in good_frame_indices:
            selected_scores.append(whole_scores[index-1][whole_dominant_index])
    
        selected_scores.sort(reverse=True)
        selected_top_k_num = max(1, int(len(selected_scores) * 0.5))
        selected_sum = 0
        for i in xrange(selected_top_k_num):
            selected_sum += selected_scores[i]
    
        selected_avg = selected_sum / float(selected_top_k_num)
        indices_to_remove = []
        for index in good_frame_indices:
            if whole_scores[index-1][whole_dominant_index] < selected_avg:
                indices_to_remove.append(index)

        for index_to_remove in indices_to_remove:
            good_frame_indices.remove(index_to_remove)
    
        selected_slices = []
        temp_group = []
        for ii in xrange(len(good_frame_indices)):
            if ii == 0 or len(temp_group) == 0:
                temp_group.append(good_frame_indices[ii])
                isContinuous = True
            elif temp_group[-1] +9 >= good_frame_indices[ii]:
                temp_group.append(good_frame_indices[ii])
                isContinuous = True
            else:
                isContinuous = False
    
            if (not isContinuous or ii==len(good_frame_indices)-1) and not len(temp_group) == 0:
                temp_start = temp_group[0]
                temp_end = temp_group[-1]
                temp_duration = temp_end - temp_start + 1
                if temp_duration >= 10:
                    temp_sum = 0.0
                    for tt in temp_group:
                        temp_sum += whole_scores[tt-1][whole_dominant_index]
                    temp_avg = temp_sum / float(len(temp_group))
    
                    selected_slices.append([temp_avg, temp_start, temp_end])
                temp_group = []


        biggest_scale_factor = 3.0
        smallest_scale_factor = 0.5
        temporal_scale_factor = math.sqrt(2)
        temporal_scales = []
        scale_index = 0
        while True:
            current_scale = video_fps * biggest_scale_factor
            for ii in range(0, scale_index, 1):
                current_scale /= temporal_scale_factor
            scale_index += 1
            if current_scale < video_fps * smallest_scale_factor:
                break
            else:
                temporal_scales.append(current_scale)

        selected_sets = []

        for scale in temporal_scales:
            scaled_slices = []
            for slice in selected_slices:
                centers = range(slice[1], slice[2]+1, 1)
                max_slice_score = -1000000
                for center in centers:
                    start = center - int(scale/2.0)
                    end = center + int(scale / 2.0)
                    if start <= 0:
                        start = center
                        end = center + int(scale)
                    center_sum = 0.0
                    valid_count = 0
                    for index in range(start, end+1, 1):
                        if index >= video_frame_count:
                            end = video_frame_count
                            break
                        center_sum += whole_scores[index-1][whole_dominant_index]
                        valid_count += 1
                    if valid_count >= 1:
                        center_avg = center_sum / float(valid_count)
                        if center_avg >= max_slice_score:
                            max_slice_score = center_avg
                            max_slice_start = start
                            max_slice_end = end

                scaled_slices.append((max_slice_score, max_slice_start, max_slice_end))

            selected_sets.append(scaled_slices)

        scale_index = 1
        for set in selected_sets:
            for slice in set:
                if savedAsFiles == True:
                    if global_clip_dst_path == '':
                        clip_default_folder = ''
                        for path in video_path.split('/')[:-1]:
                            clip_default_folder += path + '/'
                        clip_default_folder = clip_default_folder[:-1]
                    else:
                        clip_default_folder = os.path.join(global_clip_dst_path, video_path.split('/')[-1].split('.')[-2])
                    try:
                        os.mkdir(clip_default_folder)
                    except OSError:
                        pass
                    clip_save_path = os.path.join(clip_default_folder,
                                                  's{:02d}_c{:03d}.avi'.format(scale_index,  set.index(slice) + 1))
                    clipping(video_path, clip_save_path, slice[1], slice[2])

                frame_src_folder = os.path.join(global_frame_src_path, video_path.split('/')[-1].split('.')[-2])
                frame_dst_folder = os.path.join(global_frame_dst_path, video_path.split('/')[-1].split('.')[-2])
                copy_folder_path = os.path.join(frame_dst_folder, video_path.split('/')[-1].split('.')[-2] + '_s{:02d}_c{:03d}'.format(selected_sets.index(set)+1, set.index(slice)+1))

                flow_src_path = os.path.join(frame_src_folder, 'optical_flow')
                image_src_path = os.path.join(frame_src_folder, 'images')
                flow_dst_path = os.path.join(copy_folder_path, 'optical_flow')
                image_dst_path = os.path.join(copy_folder_path, 'images')

                if not os.path.exists(copy_folder_path):
                    try:
                        os.makedirs(copy_folder_path)
                    except OSError:
                        pass
                if not os.path.exists(flow_dst_path):
                    try:
                        os.mkdir(flow_dst_path)
                    except OSError:
                        pass
                if not os.path.exists(image_dst_path):
                    try:
                        os.mkdir(image_dst_path)
                    except OSError:
                        pass

                flow_x_prefix = "flow_x_"
                flow_y_prefix = "flow_y_"
                image_prefix = "img_"
                src_frame_counter = slice[1]
                dst_frame_counter = 1
                while True:
                    copy_flow_x_src = os.path.join(flow_src_path ,'{}{:05d}.jpg'.format(flow_x_prefix, src_frame_counter))
                    copy_flow_x_dst = flow_dst_path + '/{}{:05d}.jpg'.format(flow_x_prefix, dst_frame_counter)
                    shutil.copy(copy_flow_x_src, copy_flow_x_dst)

                    copy_flow_y_src = os.path.join(flow_src_path ,'{}{:05d}.jpg'.format(flow_y_prefix, src_frame_counter))
                    copy_flow_y_dst = flow_dst_path + '/{}{:05d}.jpg'.format(flow_y_prefix, dst_frame_counter)
                    shutil.copy(copy_flow_y_src, copy_flow_y_dst)

                    copy_image_src = os.path.join(image_src_path ,'{}{:05d}.jpg'.format(image_prefix, src_frame_counter))
                    copy_image_dst = image_dst_path + '/{}{:05d}.jpg'.format(image_prefix, dst_frame_counter)
                    shutil.copy(copy_image_src, copy_image_dst)

                    src_frame_counter += 1
                    dst_frame_counter += 1
                    if src_frame_counter > slice[2]:
                        break

            scale_index += 1

    video_cap.release()

    num_counter.value += 1
    current_time = time.time()
    global_clipping_one_duration = current_time - start_time
    whole_elapsed_time = current_time - global_start_time
    time_counter.value += whole_elapsed_time / float(num_counter.value)
    global_clipping_avg_duration = time_counter.value / float(num_counter.value) / 3600.0
    global_clipping_remaining_hours = float(len_video_list - num_counter.value) * global_clipping_avg_duration


def scanVideo(scan_items):
    start_time = time.time()

    global spatial_net_gpu
    global temporal_net_gpu
    global spatial_net_cpu
    global temporal_net_cpu

    global whole_scores

    input_path = scan_items[0]
    frame_count = scan_items[1]
    index = scan_items[2]

    current = current_process()
    current_id = current._identity[0] -1

    if current_id % global_num_worker < global_num_using_gpu:
        spatial_net = spatial_net_gpu
        temporal_net = temporal_net_gpu
    else:
        spatial_net = spatial_net_cpu
        temporal_net = temporal_net_cpu

    score_layer_name = 'fc-twis'

    image_frame = cv2.imread(os.path.join(input_path, 'images/img_{:05d}.jpg'.format(index)))
    rgb_score = \
    spatial_net.predict_single_frame([image_frame, ], score_layer_name, over_sample=False, multiscale=False,
                                         frame_size=(340, 256))[0].tolist()

    flow_stack = []
    for i in xrange(5):
        if index + i <= frame_count:
            x_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(index + i),
                cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(index + i),
                cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)
        else:
            x_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(frame_count),
                cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(frame_count),
                cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)
    flow_score = \
    temporal_net.predict_simple_single_flow_stack(flow_stack, score_layer_name, over_sample=False,
                                                      frame_size=(224, 224))[0].tolist()
    whole_score = [rgb_score[i] + flow_score[i] for i in xrange(len(rgb_score))]
    whole_scores[index-1] = whole_score

    showProgress(progress_type='Scanning', file_name=input_path + ' index: {}'.format(index), message=' !! Done !!',
                 process_start_time=start_time, process_id=current_id)


def eval_video(video_file_path):
    global spatial_net
    global temporal_net

    video_name = video_file_path.split('/')[-1]

    video_cap = cv2.VideoCapture(video_file_path)
    frame_cnt = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    num_frame_per_video = int(video_cap.get(cv2.CAP_PROP_FPS))

    rgb_prefix = "img_"
    flow_x_prefix = "flow_x_"
    flow_y_prefix = "flow_y_"
    out_format = "dir"
    new_size = (340, 256)
    df_path = "/home/damien/temporal-segment-networks/lib/dense_flow/"
    out_path = "/home/damien/temp"

    out_full_path = os.path.join(out_path, video_name.split('.')[-2])
    image_full_path = out_full_path + "/images"
    optical_flow_full_path = out_full_path + "/optical_flow"
    score_layer_name = 'fc-twis'

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass

    dev_id = 0
    image_path = '{}/images/img'.format(out_full_path)
    optical_flow_x_path = '{}/optical_flow/flow_x'.format(out_full_path)
    optical_flow_y_path = '{}/optical_flow/flow_y'.format(out_full_path)

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

    cmd = os.path.join(
        df_path + 'build/extract_gpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path), dev_id,
        out_format,
        new_size[0],
        new_size[1])

    os.system(cmd)
    sys.stdout.flush()

    rgb_stack_depth = 1
    flow_stack_depth = 5

    rgb_step = (frame_cnt - rgb_stack_depth) / (num_frame_per_video - 1)
    if rgb_step > 0:
        rgb_frame_ticks = range(1, min((2 + rgb_step * (num_frame_per_video - 1)), frame_cnt + 1), rgb_step)
    else:
        rgb_frame_ticks = [1] * num_frame_per_video

    assert (len(rgb_frame_ticks) == num_frame_per_video)

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
        scores = spatial_net.predict_single_frame([frame, ], score_layer_name, frame_size=(340, 256))
        rgb_frame_scores.append(scores)

    flow_frame_scores = []
    for tick in flow_frame_ticks:
        frame_idx = [min(frame_cnt, tick + offset) for offset in xrange(flow_stack_depth)]
        flow_stack = []
        for idx in frame_idx:
            x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
            y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
            x_flow_field = cv2.imread(os.path.join(optical_flow_full_path, x_name), cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(os.path.join(optical_flow_full_path, y_name), cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)

        scores = temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, frame_size=(340, 256))
        flow_frame_scores.append(scores)

    whole_scores = [x * 1.0 + y * 1.5 for x in rgb_frame_scores for y in flow_frame_scores]
    out_scores = default_aggregation_func(whole_scores)
    sys.stdin.flush()

    image_files = glob.glob(out_full_path + '/images/*')
    for f in image_files:
        os.remove(f)
    os.removedirs(out_full_path + '/images')

    optical_flow_files = glob.glob(out_full_path + '/optical_flow/*')
    for f in optical_flow_files:
        os.remove(f)
    os.removedirs(out_full_path + '/optical_flow')

    try:
        os.removedirs(out_full_path)
    except:
        pass

    return out_scores


def eval_clip(input_file_path, start_frame_num, end_frame_num, fps, net_type):
    global spatial_net_gpu
    global temporal_net_gpu
    global spatial_net_cpu
    global temporal_net_cpu

    if net_type == 'gpu':
        spatial_net = spatial_net_cpu
        temporal_net = temporal_net_gpu
    else:
        spatial_net = spatial_net_cpu
        temporal_net = temporal_net_cpu

    rgb_prefix = "img_"
    flow_x_prefix = "flow_x_"
    flow_y_prefix = "flow_y_"

    image_full_path = os.path.join(input_file_path, 'images')
    optical_flow_full_path = os.path.join(input_file_path, 'optical_flow')

    score_layer_name = 'fc-twis'

    rgb_stack_depth = 1
    flow_stack_depth = 5

    fcount = end_frame_num - start_frame_num + 1
    # num_per_clip = 1
    num_per_clip = max(1, int(fcount / float(fps) * 2.0))
    width_per_tick = max(1, int(fcount / float(num_per_clip)))

    rgb_frame_ticks = []
    for ii in xrange(num_per_clip):
        this_range = range(start_frame_num + width_per_tick * (ii),
                           min(start_frame_num + width_per_tick * (ii + 1) + 1, end_frame_num + 2 - rgb_stack_depth), 1)
        samples = random.sample(this_range, 1)
        rgb_frame_ticks.append(samples[0])

    flow_frame_ticks = []
    for ii in xrange(num_per_clip):
        this_range = range(start_frame_num + width_per_tick * (ii),
                           min(start_frame_num + width_per_tick * (ii + 1) + 1, end_frame_num + 2 - flow_stack_depth),
                           1)
        samples = random.sample(this_range, 1)
        flow_frame_ticks.append(samples[0])

    rgb_frame_scores = []
    for tick in rgb_frame_ticks:
        name = '{}{:05d}.jpg'.format(rgb_prefix, tick)
        frame = cv2.imread(os.path.join(image_full_path, name), cv2.IMREAD_COLOR)
        scores = spatial_net.predict_single_frame([frame, ], score_layer_name, frame_size=(340, 240))
        rgb_frame_scores.append(scores)

    flow_frame_scores = []
    for tick in flow_frame_ticks:
        frame_idx = [min(end_frame_num, tick + offset) for offset in xrange(flow_stack_depth)]
        flow_stack = []
        for idx in frame_idx:
            x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
            y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
            x_flow_field = cv2.imread(os.path.join(optical_flow_full_path, x_name), cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(os.path.join(optical_flow_full_path, y_name), cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)

        scores = temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, frame_size=(340, 240))
        flow_frame_scores.append(scores)

    rgb_aggregated_scores = default_aggregation_func(rgb_frame_scores)
    flow_aggregated_scores = default_aggregation_func(flow_frame_scores)
    whole_scores = [rgb * 1.0 + flow * 1.5 for rgb, flow in zip(rgb_aggregated_scores, flow_aggregated_scores)]

    return whole_scores


def preProcessInputDataForTSN(vid_list, extract_dst_path, frame_src_path, frame_dst_path, flow_type = 'optical_flow', num_worker=8, num_worker_using_gpu=4):
    global len_video_list
    global len_scan_list
    global global_start_time

    global num_counter
    global scan_counter
    global time_counter
    global counter_lock
    global scan_counter_lock
    global scan_list_lock

    global global_num_using_gpu
    global global_num_worker

    global global_extract_dst_path
    global global_clip_dst_path
    global global_frame_src_path
    global global_frame_dst_path

    global global_clipping_remaining_hours
    global global_clipping_avg_duration
    global global_clipping_one_duration

    global_clipping_remaining_hours = 0
    global_clipping_one_duration = 0
    global_clipping_avg_duration = 0

    global_num_using_gpu = num_worker_using_gpu
    global_num_worker = num_worker

    global_extract_dst_path = extract_dst_path
    global_clip_dst_path = clip_dst_path
    global_frame_src_path = frame_src_path
    global_frame_dst_path = frame_dst_path

    global_start_time = time.time()
    len_video_list = len(vid_list)
    num_counter = Value(c_int)
    scan_counter = Value(c_int)
    time_counter = Value(c_float)
    counter_lock = Lock()
    scan_counter_lock = Lock()
    scan_list_lock = Lock()


    if flow_type == 'optical_flow':
        extract_pool = Pool(num_worker)
        extract_pool.map(extractOpticalFlow, vid_list)
        extract_pool.close()

        build_net()

        num_counter.value = 0
        time_counter.value = 0
        global_start_time = time.time()

        # for video in vid_list:
        #     clippingVideos(video)

    if flow_type == 'warped_flow':
        extract_pool = Pool(num_worker)
        extract_pool.map(clippingVideos, vid_list)
        extract_pool.close()

        build_net()
        num_counter.value = 0
        time_counter.value = 0
        global_start_time = time.time()

        clipping_pool = Pool(num_worker)
        clipping_pool.map(clippingVideos, vid_list)
        clipping_pool.close()


def makeInputData(version='3'):
    violence_src_folders = ['/media/damien/DATA2/cvData/Youtube/clips*',
                           '/media/damien/DATA2/cvData/TWIS/v2/Violence',
                           '/media/damien/DATA2/cvData/TWIS/v3/Violence']
    normal_src_folder = '/media/damien/DATA2/cvData/TWIS/v2/Normal'
    video_dst_folder = '/media/damien/DATA2/cvData/TWIS/v{}'.format(version)
    extract_dst_folder = '/media/damien/DATA2/cvData/TSN_data/TWIS/v{}'.format(version)


    identity_length = 13
    identity_list = []

    violence_src_paths = []
    for violence_src_folder in violence_src_folders:
        violence_src_paths += glob.glob(os.path.join(violence_src_folder, '*'))

    print len(violence_src_paths)

    # for v_src_path in violence_src_paths:
    #     video_cap = cv2.VideoCapture(v_src_path)
    #     frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    #     video_cap.release()
    #     while True:
    #         identity = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(identity_length))
    #         if not identity in identity_list:
    #             break
    #
    #     new_file_name = '{}_{:05d}.avi'.format(identity, frame_count)
    #     dst_path = os.path.join(video_dst_folder, 'Violence', new_file_name)
    #
    #     identity_list.append(identity)
    #     shutil.copy(v_src_path, dst_path)


    violence_counter = len(glob.glob(os.path.join(video_dst_folder, 'Violence/*')))
    normal_paths = glob.glob(os.path.join(normal_src_folder, '*'))

    normal_indices = random.sample(range(0, len(normal_paths), 1), violence_counter * 3)
    normal_sampled_paths = []
    for index in normal_indices:
        normal_sampled_paths.append(normal_paths[index])


    # for src_path in normal_sampled_paths:
    #     video_cap = cv2.VideoCapture(src_path)
    #     frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    #     video_cap.release()
    #
    #     while True:
    #         identity = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(identity_length))
    #         if not identity in identity_list:
    #             break
    #
    #     new_file_name = '{}_{:05d}.avi'.format(identity, frame_count)
    #     dst_path = os.path.join(video_dst_folder, 'Normal', new_file_name)
    #
    #     identity_list.append(identity)
    #     shutil.copy(src_path, dst_path)


    num_worker = 12

    global num_extract
    global num_counter_lock
    global len_extract
    num_counter_lock = Lock()
    num_extract = Value(c_int)

    whole_video_paths = glob.glob(os.path.join(video_dst_folder, '*/*'))

    len_extract = len(whole_video_paths)

    # extract_pool = Pool(processes=num_worker)
    # extract_pool.map(extractOpticalFlowSimple, zip(whole_video_paths, [extract_dst_folder]*len(whole_video_paths), ['gpu']*len(whole_video_paths)))
    # extract_pool.close()


    v_file_list = glob.glob("/media/damien/DATA2/cvData/TWIS/v{}/Violence/*.avi".format(version))
    n_file_list = glob.glob("/media/damien/DATA2/cvData/TWIS/v{}/Normal/*.avi".format(version))

    num_of_splits = 3

    for i in range(0, num_of_splits):
        train_list_f = open("../data/twis_splits/trainlist0%d.txt" % (i + 1), "w")
        test_list_f = open("../data/twis_splits/testlist0%d.txt" % (i + 1), "w")

        v_train_indices = random.sample(range(0, len(v_file_list), 1), int(len(v_file_list) * 1.00))
        v_test_indices = random.sample(range(0, len(v_file_list), 1), int(len(v_file_list) * 0.30))

        # for i in xrange(len(v_file_list)):
        #     if i not in v_train_indices:
        #         v_test_indices.append(i)

        print len(v_train_indices)
        print len(v_test_indices)

        for i in xrange(len(v_train_indices)):
            train_list_f.write(
                v_file_list[v_train_indices[i]].split('/')[-2] + "/" + v_file_list[v_train_indices[i]].split('/')[
                    -1] + " 1\n")

        for i in xrange(len(v_test_indices)):
            test_list_f.write(
                v_file_list[v_test_indices[i]].split('/')[-2] + "/" + v_file_list[v_test_indices[i]].split('/')[
                    -1] + "\n")

        ratio_of_normal_to_violence = float(len(v_file_list)) / float(len(n_file_list))

        n_train_indices = random.sample(range(0, len(n_file_list), 1),
                                        int(len(n_file_list) * 1.00 * ratio_of_normal_to_violence))
        n_test_indices = random.sample(range(0, len(n_file_list), 1),
                                        int(len(n_file_list) * 0.30 * ratio_of_normal_to_violence))

        # n_test_count = 0
        # while True:
        #     i = random.sample(range(0, (len(n_file_list) - 1)), 1)
        #     if i[0] not in n_train_indices:
        #         n_test_indices.append(i[0])
        #         n_test_count += 1
        #         if n_test_count >= len(n_file_list) * 0.10 * ratio_of_normal_to_violence:
        #             break

        print len(n_train_indices)
        print len(n_test_indices)

        for i in xrange(len(n_train_indices)):
            train_list_f.write(
                n_file_list[n_train_indices[i]].split('/')[-2] + "/" + n_file_list[n_train_indices[i]].split('/')[
                    -1] + " 2\n")

        for i in xrange(len(n_test_indices)):
            test_list_f.write(
                n_file_list[n_test_indices[i]].split('/')[-2] + "/" + n_file_list[n_test_indices[i]].split('/')[
                    -1] + "\n")

        train_list_f.close()
        test_list_f.close()

    dataset = "twis"
    frame_path = "/media/damien/DATA2/cvData/TSN_data/TWIS/v{}".format(version)
    rgb_p = 'img_'
    flow_x_p = 'flow_x'
    flow_y_p = 'flow_y'
    num_split = 3
    out_path = '../data/twis_parsed'
    shuffle = False

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # operation
    print 'processing dataset {}'.format(dataset)
    split_tp = parse_split_file(dataset)
    f_info = parse_directory(frame_path, rgb_p, flow_x_p, flow_y_p)

    print 'writing list files for training/testing'
    for i in xrange(max(num_split, len(split_tp))):
        lists = build_split_list(split_tp, f_info, i, shuffle)
        open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][0])
        open(os.path.join(out_path, '{}_rgb_val_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][1])
        open(os.path.join(out_path, '{}_flow_train_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[1][0])
        open(os.path.join(out_path, '{}_flow_val_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[1][1])


def convertInputSize(video_path):
    new_size = (340, 256)

    frame_default_path = '/media/damien/DATA2/cvData/TSN_data/TWIS/v2'
    frame_path = os.path.join(frame_default_path, video_path.split('/')[-1].split('.')[-2])

    video_cap = cv2.VideoCapture(video_path)
    video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    video_cap.release()

    image_paths = glob.glob(frame_path + '/images/*')
    flow_paths = glob.glob(frame_path + '/optical_flow/*')

    image_count = len(image_paths)
    flow_count = len(flow_paths)/2

    if image_count < video_frame_count or flow_count < video_frame_count:
        video_extract_dst = '/media/damien/DATA2/cvData/TSN_data/TWIS/v2'
        extractOpticalFlowSimple(video_path,video_extract_dst)
        image_paths = glob.glob(frame_path + '/images/*')
        flow_paths = glob.glob(frame_path + '/optical_flow/*')
        print 'FILE {} ReExtracted!'.format(video_path)

    all_frame_paths = image_paths + flow_paths
    for path in all_frame_paths:
        frame = cv2.imread(path)
        height = int(frame.shape[0])
        width = int(frame.shape[1])

        if not (width == new_size[0] and height == new_size[1]):
            new_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            try:
                os.remove(path)
            except OSError:
                pass
            cv2.imwrite(path, new_frame)
            print 'Image ({}, {})|{} ReSized!'.format(width, height, path)

    print'FILE {} DONE DONE!!'.format(video_path)


if __name__ == '__main__':
    makeInputData(4)