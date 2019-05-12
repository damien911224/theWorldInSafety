import cv2
import sys
sys.path.insert(0, '../lib/caffe-action/python')
import caffe
from caffe.io import oversample
import os
import glob
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import shutil
import math
from sklearn import mixture
from pipes import quote
from utils.io import flow_stack_oversample, fast_list2arr
from multiprocessing import Pool, Value, Lock, current_process, Manager
from ctypes import c_int
from pytube import YouTube
from apiclient.discovery import build
from utils.video_funcs import default_aggregation_func
from random import shuffle

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = ""
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


class CaffeNet(object):
    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        if True:
            caffe.set_logging_disabled()

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

    spatial_net_proto = "../models/twis/tsn_bn_inception_rgb_deploy.prototxt"
    spatial_net_weights = "../models/twis_caffemodels/v4/twis_spatial_net_v4.caffemodel"
    temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
    temporal_net_weights = "../models/twis_caffemodels/v4/twis_temporal_net_v4.caffemodel"

    global score_layer_name

    score_layer_name = 'fc-twis'
    device_id = 0

    spatial_net_gpu = CaffeNet(spatial_net_proto, spatial_net_weights, device_id)
    temporal_net_gpu = CaffeNet(temporal_net_proto, temporal_net_weights, device_id)

    spatial_net_cpu = CaffeNet(spatial_net_proto, spatial_net_weights, -1)
    temporal_net_cpu = CaffeNet(temporal_net_proto, temporal_net_weights, -1)


def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


def checkClips(check_items):
    global spatial_net_gpu
    global temporal_net_gpu
    global score_layer_name

    clip_path = check_items[0]
    temp_extract_folder = check_items[1]

    print 'Checking|File {}'.format(clip_path)

    violence_index = 0
    num_per_clip_factor = 0.3
    video_cap = cv2.VideoCapture(clip_path)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_cap.release()

    temp_extract_path = os.path.join(temp_extract_folder, clip_path.split('/')[-1].split('.')[-2])
    try:
        os.makedirs(temp_extract_path)
    except OSError:
        pass
    extractOpticalFlow(clip_path, temp_extract_folder, 'gpu')

    spatial_stack_depth = 1
    temporal_stack_depth = 5
    spatial_step_count = int((frame_count - spatial_stack_depth) * num_per_clip_factor)
    temporal_step_count = int((frame_count - temporal_stack_depth) * num_per_clip_factor)

    spatial_step = int((frame_count - spatial_stack_depth) / spatial_step_count)
    temporal_step = int((frame_count - temporal_stack_depth) / temporal_step_count)

    spatial_base_points = range(1, frame_count - spatial_stack_depth + 1, spatial_step)
    temporal_base_points = range(1, frame_count - temporal_stack_depth + 1, temporal_step)

    spatial_sampled_frames = []
    for ii in xrange(len(spatial_base_points)-1):
        sampled_frame = random.sample(range(spatial_base_points[ii], spatial_base_points[ii+1], 1), 1)
        spatial_sampled_frames.append(sampled_frame[0])

    temporal_sampled_frames = []
    for ii in xrange(len(temporal_base_points) - 1):
        sampled_frame = random.sample(range(temporal_base_points[ii], temporal_base_points[ii + 1], 1), 1)
        temporal_sampled_frames.append(sampled_frame[0])

    image_path = os.path.join(temp_extract_path, 'images')
    flow_path = os.path.join(temp_extract_path, 'optical_flow')

    rgb_scores = []
    for spatial_frame in spatial_sampled_frames:
        frame_stack = []
        for offset in xrange(spatial_stack_depth):
            frame_path = os.path.join(image_path, 'img_{:05d}.jpg'.format(spatial_frame + offset))
            frame = cv2.imread(frame_path)
            frame_stack.append(frame)
        score = spatial_net_gpu.predict_single_frame(frame_stack, score_layer_name, frame_size=(340, 256))
        rgb_scores.append(score)

    flow_scores = []
    for temporal_frame in temporal_sampled_frames:
        flow_stack = []
        for offset in xrange(temporal_stack_depth):
            x_flow_path = os.path.join(flow_path, 'flow_x_{:05d}.jpg'.format(temporal_frame + offset))
            y_flow_path = os.path.join(flow_path, 'flow_y_{:05d}.jpg'.format(temporal_frame + offset))
            x_flow = cv2.imread(x_flow_path, cv2.IMREAD_GRAYSCALE)
            y_flow = cv2.imread(y_flow_path, cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow)
            flow_stack.append(y_flow)
        score = temporal_net_gpu.predict_single_flow_stack(flow_stack, score_layer_name, frame_size=(340, 256))
        flow_scores.append(score)

    rgb_aggregation_score = default_aggregation_func(rgb_scores, normalization=False)
    flow_aggregation_score = default_aggregation_func(flow_scores, normalization=False)

    whole_aggregation_score = [rgb_aggregation_score[i]*1.0 + flow_aggregation_score[i]*5.0 for i in xrange(len(rgb_aggregation_score))]

    prediction = np.argmax(whole_aggregation_score)

    print 'File {}|Score ({}, {})'.format(clip_path, whole_aggregation_score[0], whole_aggregation_score[1])

    if (not prediction == violence_index) or whole_aggregation_score[violence_index] < 10.0:
        try:
            os.remove(clip_path)
        except OSError:
            pass

    frame_paths = glob.glob(temp_extract_path + '/*/*')
    for path in frame_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    try:
        os.removedirs(image_path)
    except OSError:
        pass

    try:
        os.removedirs(flow_path)
    except OSError:
        pass

    try:
        os.removedirs(temp_extract_path)
    except OSError:
        pass


def youtube_search(keyword, max_results=25, order="relevance", token=None, location=None, location_radius=None):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY)

    # Call the search.list method to retrieve results matching the specified
    # query term.
    search_response = youtube.search().list(
        q=keyword,
        type="video",
        pageToken=token,
        order=order,
        part="id,snippet",
        maxResults=max_results,
        location=location,
        locationRadius=location_radius
    ).execute()

    video_urls = []
    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            video_urls.append('https://www.youtube.com/watch?v={}'.format(search_result["id"]["videoId"]))
    try:
        nexttok = search_response["nextPageToken"]
        return nexttok, video_urls
    except Exception as e:
        nexttok = "last_page"
        return nexttok, video_urls


def download_youtube_video(download_items):
    video_url = download_items[0]
    dst_folder = download_items[1]
    keyword = download_items[2]


    try:
        youtube = YouTube(video_url)
    except:
        print '{} !! YOUTUBE ERROR !!'.format(video_url)
        return

    filename = keyword.replace(' ', '') + '_' + youtube.video_id
    check_save_path = glob.glob(os.path.join(dst_folder, filename) + '.*')
    if len(check_save_path) == 0:
        youtube.set_filename(filename)

        video = youtube.get_videos()[-2]

        try:
            video.download(dst_folder)
        except:
            print '{} !! YOUTUBE DOWNLOAD ERROR !!'.format(filename)
            return

        print '{} !! DOWNLOAD DONE !!'.format(filename)


def makeActionProposals(video_path, frame_src_path, clip_dst_path='', saved_as_files=True, plt_showed=False):
    global global_num_worker
    global global_num_using_gpu
    global len_scan_list
    global scan_counter
    global scan_counter_lock
    global whole_scores

    global_num_worker = 5
    global_num_using_gpu = 1
    violence_index = 0
    normal_index = 1

    scan_counter = Value(c_int)
    scan_counter_lock = Lock()

    manager = Manager()
    whole_scores = manager.list()

    scan_input_path = os.path.join(frame_src_path, video_path.split('/')[-1].split('.')[-2])
    scan_counter.value = 0

    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened():
        video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        for i in xrange(video_frame_count):
            whole_scores.append([0.0, 0.0])

        scan_input_paths = []
        for i in xrange(video_frame_count + 1):
            input_path = scan_input_path
            scan_input_paths.append(input_path)

        len_scan_list = len(scan_input_paths)
        scan_pool = Pool(global_num_worker)
        scan_pool.map(scanVideo, zip(scan_input_paths, [video_frame_count] * video_frame_count,
                                     range(1, video_frame_count + 1, 1)))
        scan_pool.close()

        whole_violence_sum = 0
        whole_normal_sum = 0
        violence_count = 0
        normal_count = 0
        for score in whole_scores:
            for i in xrange(len(whole_scores[0])):
                if np.argmax(score) == violence_index:
                    whole_violence_sum += max(score)
                    violence_count += 1
                elif np.argmax(score) == normal_index:
                    whole_normal_sum += max(score)
                    normal_count += 1
        whole_violence_avg_score = whole_violence_sum / violence_count
        whole_normal_avg_score = whole_normal_sum / normal_count

        violence_maxima = []
        normal_maxima = []
        for si in xrange(len(whole_scores)):
            if si >= 1 and si < len(whole_scores) - 1:
                previous_score = whole_scores[si - 1]
                next_score = whole_scores[si + 1]
                current_score = whole_scores[si]

                current_dominant_index = np.argmax(current_score)
                if previous_score[current_dominant_index] < current_score[current_dominant_index] \
                        and next_score[current_dominant_index] < current_score[current_dominant_index]:
                    if current_dominant_index == violence_index and current_score[current_dominant_index] >= whole_violence_avg_score:
                        violence_maxima.append([si + 1, current_score[current_dominant_index]])
                    elif current_dominant_index == normal_index and current_score[current_dominant_index] >= whole_normal_avg_score:
                        normal_maxima.append([si + 1, current_score[current_dominant_index]])

        violence_max_scores = []
        for maxi in violence_maxima:
            violence_max_scores.append(whole_scores[maxi[0] - 1][violence_index])
        violence_max_score_sum = np.sum(violence_max_scores)
        violence_max_score_avg = violence_max_score_sum / float(len(violence_maxima))

        for maxi in violence_maxima:
            if whole_scores[maxi[0] - 1][violence_index] < violence_max_score_avg:
                violence_maxima.remove(maxi)

        normal_max_scores = []
        for maxi in normal_maxima:
            normal_max_scores.append(whole_scores[maxi[0] - 1][normal_index])
        normal_max_score_sum = np.sum(normal_max_scores)
        normal_max_score_avg = normal_max_score_sum / float(len(normal_maxima))

        for maxi in normal_maxima:
            if whole_scores[maxi[0] - 1][normal_index] < normal_max_score_avg:
                normal_maxima.remove(maxi)

        violence_local_max = [-1, -100000.0]
        violence_comp_index = 1
        violence_last_index = -1
        while True:
            if violence_comp_index >= len(violence_maxima):
                break

            if violence_local_max[0] == -1:
                violence_local_max = violence_maxima[violence_comp_index - 1]

            if violence_last_index == -1:
                violence_last_index = violence_maxima[violence_comp_index - 1][0]

            if violence_last_index + video_fps > violence_maxima[violence_comp_index][0]:
                if violence_maxima[violence_comp_index][1] > violence_local_max[1]:
                    violence_maxima.remove(violence_local_max)
                    violence_last_index = violence_maxima[violence_comp_index - 1][0]
                    violence_local_max = violence_maxima[violence_comp_index - 1]
                else:
                    violence_maxima.remove(violence_maxima[violence_comp_index])
            else:
                violence_local_max = [-1, -100000.0]
                violence_last_index = -1
                violence_comp_index += 1

        normal_local_max = [-1, -100000.0]
        normal_comp_index = 1
        normal_last_index = -1
        while True:
            if normal_comp_index >= len(normal_maxima):
                break

            if normal_local_max[0] == -1:
                normal_local_max = normal_maxima[normal_comp_index - 1]

            if normal_last_index == -1:
                normal_last_index = normal_maxima[normal_comp_index - 1][0]

            if normal_last_index + video_fps > normal_maxima[normal_comp_index][0]:
                if normal_maxima[normal_comp_index][1] > normal_local_max[1]:
                    normal_maxima.remove(normal_local_max)
                    normal_last_index = normal_maxima[normal_comp_index - 1][0]
                    normal_local_max = normal_maxima[normal_comp_index - 1]
                else:
                    normal_maxima.remove(normal_maxima[normal_comp_index])
            else:
                normal_local_max = [-1, -100000.0]
                normal_last_index = -1
                normal_comp_index += 1

        violence_big_slices = []
        for maxi in violence_maxima:
            start_bound_count = 0
            start_bound_index = violence_maxi[0]
            while True:
                start_bound_index -= 1
                if start_bound_index >= 1:
                    if whole_scores[start_bound_index - 1][violence_index] < whole_violence_avg_score * 0.75:
                        start_bound_count += 1
                else:
                    start_bound_index = 1
                    break

                if start_bound_count >= 3:
                    break

            end_bound_count = 0
            end_bound_index = violence_maxi[0]
            while True:
                end_bound_index += 1
                if not end_bound_index > video_frame_count:
                    if whole_scores[end_bound_index - 1][violence_index] < whole_violence_avg_score * 0.75:
                        end_bound_count += 1
                else:
                    end_bound_index = video_frame_count
                    break

                if end_bound_count >= 3:
                    break

            violence_big_slices.append([start_bound_index, end_bound_index])

        normal_big_slices = []
        for maxi in normal_maxima:
            start_bound_count = 0
            start_bound_index = normal_maxi[0]
            while True:
                start_bound_index -= 1
                if start_bound_index >= 1:
                    if whole_scores[start_bound_index - 1][normal_index] < whole_normal_avg_score * 0.75:
                        start_bound_count += 1
                else:
                    start_bound_index = 1
                    break

                if start_bound_count >= 3:
                    break

            end_bound_count = 0
            end_bound_index = normal_maxi[0]
            while True:
                end_bound_index += 1
                if not end_bound_index > video_frame_count:
                    if whole_scores[end_bound_index - 1][normal_index] < whole_normal_avg_score * 0.75:
                        end_bound_count += 1
                else:
                    end_bound_index = video_frame_count
                    break

                if end_bound_count >= 3:
                    break

            normal_big_slices.append([start_bound_index, end_bound_index])

        violence_selected_slices = []
        isLeft = False
        if len(violence_big_slices) >= 1:
            current_ss = violence_big_slices[0]
            bs_index = 1
            while True:
                if bs_index >= len(violence_big_slices):
                    if isLeft == True:
                        violence_selected_slices.append(current_ss)
                    break

                isLeft = False

                current_slice_length = current_ss[1] - current_ss[0] + 1
                compare_slice_length = violence_big_slices[bs_index][1] - violence_big_slices[bs_index][0] + 1
                end_index = current_ss[1]
                start_index = violence_big_slices[bs_index][0]
                union_length = end_index - start_index + 1

                if union_length >= current_slice_length / 2 or union_length >= compare_slice_length / 2:
                    current_ss = [current_ss[0], violence_big_slices[bs_index][1]]
                    bs_index += 1
                else:
                    violence_selected_slices.append(current_ss)
                    current_ss = violence_big_slices[bs_index]
                    isLeft = True
                    bs_index += 1

        normal_selected_slices = []
        isLeft = False
        if len(normal_big_slices) >= 1:
            current_ss = normal_big_slices[0]
            bs_index = 1
            while True:
                if bs_index >= len(normal_big_slices):
                    if isLeft == True:
                        normal_selected_slices.append(current_ss)
                    break

                isLeft = False

                current_slice_length = current_ss[1] - current_ss[0] + 1
                compare_slice_length = normal_big_slices[bs_index][1] - normal_big_slices[bs_index][0] + 1
                end_index = current_ss[1]
                start_index = normal_big_slices[bs_index][0]
                union_length = end_index - start_index + 1

                if union_length >= current_slice_length / 2 or union_length >= compare_slice_length / 2:
                    current_ss = [current_ss[0], normal_big_slices[bs_index][1]]
                    bs_index += 1
                else:
                    normal_selected_slices.append(current_ss)
                    current_ss = normal_big_slices[bs_index]
                    isLeft = True
                    bs_index += 1

        for slice in violence_selected_slices:
            if saved_as_files == True:
                video_name = video_path.split('/')[-1].split('.')[-2]

                if clip_dst_path == '':
                    clip_default_folder = ''
                    for path in video_path.split('/')[:-1]:
                        clip_default_folder += path + '/'
                    clip_default_folder = os.path.join(clip_default_folder[:-1],
                                                       video_path.split('/')[-1].split('.')[-2])
                else:
                    clip_default_folder = os.path.join(clip_dst_path,
                                                       video_path.split('/')[-1].split('.')[-2])
                try:
                    os.makedirs(clip_default_folder)
                except OSError:
                    pass
                clip_save_path = os.path.join(clip_default_folder,
                                              '{}_{}_c{:03d}.avi'.format(video_name, 'violence',violence_selected_slices.index(slice) + 1))
                clipping(video_path, clip_save_path, slice[0], slice[1])

        for slice in normal_selected_slices:
            if saved_as_files == True:
                video_name = video_path.split('/')[-1].split('.')[-2]

                if clip_dst_path == '':
                    clip_default_folder = ''
                    for path in video_path.split('/')[:-1]:
                        clip_default_folder += path + '/'
                    clip_default_folder = os.path.join(clip_default_folder[:-1],
                                                       video_path.split('/')[-1].split('.')[-2])
                else:
                    clip_default_folder = os.path.join(clip_dst_path,
                                                       video_path.split('/')[-1].split('.')[-2])
                try:
                    os.makedirs(clip_default_folder)
                except OSError:
                    pass
                clip_save_path = os.path.join(clip_default_folder,
                                              '{}_{}_c{:03d}.avi'.format(video_name, 'normal',
                                                                         normal_selected_slices.index(slice) + 1))
                clipping(video_path, clip_save_path, slice[0], slice[1])

        video_cap.release()

        if plt_showed:
            plt.rcdefaults()
            xPos = range(0, video_frame_count, 1)
            xLabels = []
            for pos in xPos:
                xLabels.append('{:05d}'.format(pos + 1))
            yScores = []
            for score in whole_scores:
                yScores.append(score[violence_index])

            bar_colors = []
            colors = ['k', 'maroon']
            element_index = 1

            for slice in violence_selected_slices:
                while element_index < slice[0]:
                    bar_colors.append(colors[0])
                    element_index += 1

                while element_index <= slice[1]:
                    bar_colors.append(colors[1])
                    element_index += 1

                if slice[1] == selected_slices[-1][1]:
                    while element_index <= video_frame_count:
                        bar_colors.append(colors[0])
                        element_index += 1

            xLine = xPos
            yLine = [whole_violence_avg_score] * video_frame_count

            plt.bar(xPos, yScores, align='center', alpha=0.5, color=bar_colors)
            plt.xticks(xPos, xLabels)
            plt.ylabel('Score')
            plt.title('Scan Results')
            plt.plot(xLine, yLine, color='red')
            plt.show()

    return [whole_scores, selected_slices]


def makeActionProposalsForViolence(video_path, frame_src_path, clip_dst_path='', saved_as_files=True, plt_showed=False):
    global global_num_worker
    global global_num_using_gpu
    global len_scan_list
    global scan_counter
    global scan_counter_lock
    global whole_scores

    global_num_worker = 12
    global_num_using_gpu = 6
    violence_index = 0
    top_selected_maxima_factor = 0.5

    scan_counter = Value(c_int)
    scan_counter_lock = Lock()

    manager = Manager()
    whole_scores = manager.list()

    scan_input_path = os.path.join(frame_src_path, video_path.split('/')[-1].split('.')[-2])
    scan_counter.value = 0

    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened():
        video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
        video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        scan_indices = range(1, video_frame_count+1, 1)

        for i in xrange(len(scan_indices)):
            whole_scores.append([0.0, 0.0])

        len_scan_list = len(scan_indices)

        scan_pool = Pool(global_num_worker)
        try:
            scan_pool.map(temporalScanVideo, zip([scan_input_path]*len(scan_indices), [video_frame_count] * len(scan_indices),
                                        scan_indices))
        except:
            scan_pool.close()
            return [ 0.0, 0.0 ]

        scan_pool.close()

        whole_violence_sum = 0
        violence_count = 0
        for score in whole_scores:
            if np.argmax(score) == violence_index:
                whole_violence_sum += max(score)
                violence_count += 1

        if violence_count == 0:
            return[ 0.0, 0.0 ]

        whole_violence_avg_score = whole_violence_sum / violence_count

        min_violence_score = 0.0
        max_threshold_score = 5.0
        end_index = -1

        big_slices = []
        while True:
            start_index = end_index + 1
            first_index = start_index
            for index in range(first_index, len(whole_scores)+1, 1):
                start_index += 1
                if whole_scores[index-1][violence_index] >= min_violence_score:
                    break

            if start_index >= len(whole_scores):
                break

            maxima = []
            falling_counter = 0
            end_index = start_index + 1
            for index in range(start_index+1, len(whole_scores)+1, 1):
                end_index = index
                if whole_scores[index-1][violence_index] < min_violence_score:
                    falling_counter += 1

                if falling_counter >= 5:
                    end_index = index -falling_counter/2- 1
                    break

                if index >= 2 and index <= len(whole_scores)-1:
                    previous_score = whole_scores[index -1 - 1][violence_index]
                    next_score = whole_scores[index -1 + 1][violence_index]
                    current_score = whole_scores[index -1][violence_index]
                    if previous_score < current_score and next_score < current_score:
                        if current_score >= max_threshold_score:
                            maxima.append([index, current_score])

            number_of_components = max(1, len(maxima))
            if len(maxima) == 0:
                maxima = None

            gmm_elements = []
            for index in range(start_index, end_index+1, 1):
                current_score = whole_scores[index-1][violence_index]

                if index >= 1 and index < len(whole_scores):
                    previous_score = whole_scores[index-2][violence_index]
                    next_score =  whole_scores[index][violence_index]
                    if current_score < min_violence_score and (previous_score >= min_violence_score and next_score >= min_violence_score):
                        current_score = (previous_score + next_score)/2.0

                gmm_elements.append([index, current_score])

            if len(gmm_elements) == 0:
                continue

            gmm = mixture.GaussianMixture(n_components=number_of_components, covariance_type='spherical',
                                          max_iter=500, means_init=maxima)
            gmm.fit(gmm_elements)

            means = gmm.means_
            covariances = gmm.covariances_

            components = []
            for ii in xrange(len(means)):
                if means[ii][1] >= max_threshold_score:
                    components.append([means[ii][0], means[ii][1], covariances[ii] * 1.5])

            components.sort()

            for component in components:
                lower_bound = max(1, int(component[0] - component[2]))
                upper_bound = min(video_frame_count, int(math.ceil(component[0] + component[2])))
                big_slices.append([lower_bound, upper_bound, component[1]])

            if end_index >= video_frame_count:
                break


        selected_slices = []
        isLeft = False
        if len(big_slices) >= 1:
            current_ss = big_slices[0]
            bs_index = 1
            while True:
                if bs_index >= len(big_slices):
                    if isLeft:
                        selected_slices.append(current_ss)
                    break

                current_start = current_ss[0]
                current_end = current_ss[1]

                compare_start = big_slices[bs_index][0]
                compare_end = big_slices[bs_index][1]

                if current_end < compare_start:
                    if compare_start - current_end < video_fps:
                        start_index = current_start
                        end_index = compare_end
                        current_ss = [start_index, end_index]
                        bs_index += 1
                        isLeft = True
                    else:
                        selected_slices.append(current_ss)
                        bs_index += 1
                        if bs_index >= len(big_slices):
                            break
                        current_ss = big_slices[bs_index]
                else:
                    start_index = current_start
                    end_index = compare_end
                    current_ss = [start_index, end_index]
                    bs_index += 1
                    isLeft = True

        removed_slices = []
        for slice in selected_slices:
            duration = slice[1] - slice[0] + 1
            if duration < 10:
                removed_slices.append(slice)

        for slice in removed_slices:
            selected_slices.remove(slice)

        scores_slices = []
        for slice in selected_slices:
            slice_sum = 0.0
            for ii in range(slice[0] - 1, slice[1] - 1, 1):
                slice_sum += whole_scores[ii][violence_index]
            duration = slice[1] - slice[0] * 1
            slice_avg = slice_sum / duration
            scores_slices.append([slice_avg, slice])

        scores_slices.sort(reverse=True)
        top_selected_maxima_number = int(float(len(selected_slices)) * top_selected_maxima_factor)

        new_ss = []
        for i in xrange(top_selected_maxima_number):
            new_ss.append(scores_slices[i][1])
        selected_slices = new_ss

        for slice in selected_slices:
            if saved_as_files == True:
                if clip_dst_path == '':
                    clip_default_folder = ''
                    for path in video_path.split('/')[:-1]:
                        clip_default_folder += path + '/'
                    clip_default_folder = os.path.join(clip_default_folder[:-1], video_path.split('/')[-1].split('.')[-2])
                else:
                    clip_default_folder = os.path.join(clip_dst_path)
                try:
                    os.makedirs(clip_default_folder)
                except OSError:
                    pass
                clip_save_path = os.path.join(clip_default_folder,
                                              video_path.split('/')[-1].split('.')[-2] + '_c{:03d}.avi'.format(selected_slices.index(slice) + 1))
                clipping(video_path, clip_save_path, slice[0], slice[1])

        video_cap.release()

        if plt_showed:
            plt.rcdefaults()
            xPos = range(0, video_frame_count, 1)
            xLabels = []
            for pos in xPos:
                xLabels.append('{:05d}'.format(pos + 1))
            yScores = []
            for score in whole_scores:
                yScores.append(score[violence_index])

            bar_colors = []
            colors = ['k', 'maroon']
            element_index = 1

            for slice in selected_slices:
                while element_index < slice[0]:
                    bar_colors.append(colors[0])
                    element_index += 1

                while element_index <= slice[1]:
                    bar_colors.append(colors[1])
                    element_index += 1

                if slice[1] == selected_slices[-1][1]:
                    while element_index <= video_frame_count:
                        bar_colors.append(colors[0])
                        element_index += 1

            xLine = xPos
            yLine = [whole_violence_avg_score] * video_frame_count

            plt.bar(xPos, yScores, align='center', alpha=0.5, color=bar_colors)
            plt.xticks(xPos, xLabels)
            plt.ylabel('Score')
            plt.title('Scan Results')
            plt.plot(xLine, yLine, color='red')
            plt.show()

    return [whole_scores, selected_slices]


def showProgress(progress_type='', file_name='', message=' DONE', process_start_time=0.0, process_id=0):
    global global_num_using_gpu
    global global_num_worker
    global scan_counter
    global scan_counter_lock
    global len_scan_list

    if progress_type == 'Scanning':
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
                    "{0}|{1:05d}th|{2:06.3f}%|Current: {3:02d} {4} Worker|OneDuration: {5:.2f}Secs\n".format(
                        progress_type, scan_counter.value,
                        scan_counter.value / float(len_scan_list) * 100.0,
                        process_id % global_num_worker + 1, process_type, elapsed_time_per_process) + \
                    " " * (len(progress_type) + 16) + "|FileName: " + file_name + message


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
    spatial_net.predict_single_frame([image_frame, ], score_layer_name, over_sample=False,
                                         frame_size=(340, 256))[0].tolist()

    flow_stack = []
    for i in range(-2, 3, 1):
        if index + i >= 1 and index + i <= frame_count:
            x_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(index + i),
                cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(index + i),
                cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)
        elif index + i < 1:
            x_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(1),
                cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(1),
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
    temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, over_sample=False,
                                                      frame_size=(224, 224))[0].tolist()
    whole_score = [rgb_score[i]*1.0 + flow_score[i]*5.0 for i in xrange(len(rgb_score))]
    whole_scores[index-1] = whole_score

    showProgress(progress_type='Scanning', file_name=input_path + ' index: {}'.format(index), message=' !! Done !!',
                 process_start_time=start_time, process_id=current_id)


def temporalScanVideo(scan_items):
    start_time = time.time()

    global temporal_net_gpu
    global temporal_net_cpu

    global whole_scores

    input_path = scan_items[0]
    frame_count = scan_items[1]
    index = scan_items[2]

    current = current_process()
    current_id = current._identity[0] -1

    if current_id % global_num_worker < global_num_using_gpu:
        temporal_net = temporal_net_gpu
    else:
        temporal_net = temporal_net_cpu

    score_layer_name = 'fc-twis'

    flow_stack = []
    for i in range(-2, 3, 1):
        if index + i >= 1 and index + i <= frame_count:
            x_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(index + i),
                cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(index + i),
                cv2.IMREAD_GRAYSCALE)
            flow_stack.append(x_flow_field)
            flow_stack.append(y_flow_field)
        elif index + i < 1:
            x_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(1),
                cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(
                os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(1),
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
    temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, over_sample=False,
                                                      frame_size=(224, 224))[0].tolist()
    whole_score = flow_score
    whole_scores[index-1] = whole_score

    showProgress(progress_type='Scanning', file_name=input_path + ' index: {}'.format(index), message=' !! Done !!',
                 process_start_time=start_time, process_id=current_id)


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


def extractOpticalFlow(video_file_path, extract_dst_path, processor_type='gpu'):
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
        print '!!FILE {}!! VIDEO CAP ERROR !!'.format(video_file_path)
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
            # print '!! EXTRACT PASS !!'
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


def extractFrames(video_path, out_full_path):
    extract_worker = 8
    image_prefix = 'img_'

    video_cap = cv2.VideoCapture(video_path)
    fcount = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1

    for i in xrange(fcount):
        ret, frame = video_cap.read()
        assert ret
        cv2.imwrite('{}/{}{:05d}.jpg'.format(out_full_path, image_prefix, i+1), frame)

    video_cap.release()


def extractAFrame(extract_items):
    image_prefix = extract_items[0]
    video_path = extract_items[1]
    out_full_path = extract_items[2]
    index = extract_items[3]

    video_cap = cv2.VideoCapture(video_path)

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = video_cap.read()
    if ok:
        cv2.imwrite('{}/{}{:05d}.jpg'.format(out_full_path, image_prefix, index+1), frame)

    video_cap.release()


def youtube_crawler():
    temp_dst_folder = '/media/damien/DATA2/cvData/Youtube'
    video_dst_folder = temp_dst_folder + '/videos'
    clip_dst_folder = temp_dst_folder + '/clips'
    mask_dst_folder = temp_dst_folder + '/masked'
    mask_frame_folder = temp_dst_folder + '/temp_frames'
    dst_frame_folder = temp_dst_folder + '/frames'
    temp_extract_folder = temp_dst_folder + '/temp'
    video_ids_file = temp_dst_folder + '/video_ids.txt'


    keywords = \
        [
            'UFC highlight',
            'how to kick',
            'how to punch',
            'how to fight',
            'kickboxing',
            'UFC knockouts'
        ]


    needed_clip_number = 2000
    download_worker = 12
    check_worker = 8

    video_ids = []

    if os.path.exists(video_ids_file):
        video_ids_fp = open(video_ids_file, 'r')
        while True:
            line = video_ids_fp.readline()
            if not line:
                break
            video_ids.append(line[:-1])
        video_ids_fp.close()


    remaining_clip_results = needed_clip_number
    max_results_per_keyword = int(remaining_clip_results / len(keywords) * 1.0)
    while True:
        for keyword in keywords:
            token = None
            remaining_result = max_results_per_keyword
            while True:
                video_urls = []
                current_keyword_result_counter = 0
                max_results = max(1, min(50, remaining_result))
                while True:
                    token, current_video_urls = youtube_search(keyword=keyword, max_results=max_results, token=token)
                    removed_current_video_urls = []
                    for url in current_video_urls:
                        video_id = url.split('=')[-1]
                        if video_id in video_ids:
                            removed_current_video_urls.append(url)
                    for url in removed_current_video_urls:
                        current_video_urls.remove(url)

                    current_keyword_result_counter += len(current_video_urls)
                    video_urls += current_video_urls

                    if token == 'last_page' or current_keyword_result_counter >= remaining_result:
                        break

                    max_results = max(1, min(50, remaining_result - current_keyword_result_counter))

                download_pool = Pool(processes=download_worker)
                download_pool.map(download_youtube_video, zip(video_urls, [video_dst_folder]*len(video_urls), [keyword]*len(video_urls)))
                download_pool.close()

                downloaded_counter = len(glob.glob(video_dst_folder + '/{}_*'.format(keyword.replace(' ', ''))))
                if downloaded_counter >= max_results_per_keyword or token == 'last_page':
                    break

                remaining_result = max_results_per_keyword - downloaded_counter

        print ''
        print 'Downloading Youtube Files Done!'
        print ''


        # for path in downloaded_paths:
        #     mask_frame_path = os.path.join(mask_frame_folder, path.split('/')[-1].split('.')[-2])
        #
        #     if not os.path.exists(mask_frame_path):
        #         try:
        #             os.makedirs(mask_frame_path)
        #         except OSError:
        #             pass
        #
        #         extractFrames(path, mask_frame_path)
        #
        #     masking_cmd = 'python ../lib/darknet/darknet.py -v {} -f {} -s {}'.format(quote(path), quote(mask_frame_path), quote(mask_dst_folder))
        #     os.system(masking_cmd)
        #
        #     crop_frames = glob.glob(mask_frame_path + '/*')
        #     for frame in crop_frames:
        #         try:
        #             os.remove(frame)
        #         except OSError:
        #             pass
        #
        #     try:
        #         os.removedirs(mask_frame_path)
        #     except OSError:
        #         pass


        build_net()


        print ''
        print 'Clipping Strats'
        print ''

        downloaded_paths = glob.glob(video_dst_folder + '/*')
        shuffle(downloaded_paths)

        for path in downloaded_paths:
            video_ids.append(path.split('_')[-1].split('.')[-2])

            print 'Extracting|{:05d}|{:.3f}%|File {}'.format(downloaded_paths.index(path)+1,
                                                            (downloaded_paths.index(path)+1)/float(len(downloaded_paths)*100.0)
                                                            ,path)
            extractOpticalFlow(path, dst_frame_folder)

            [whole_scores, selected_slices] = makeActionProposalsForViolence(path, dst_frame_folder, clip_dst_folder)

            extract_paths = glob.glob(os.path.join(dst_frame_folder, path.split('/')[-1].split('.')[-2], '*/*'))
            for path in extract_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

            try:
                os.removedirs(os.path.join(dst_frame_folder, path.split('/')[-1].split('.')[-2], 'images'))
            except OSError:
                pass

            try:
                os.removedirs(os.path.join(dst_frame_folder, path.split('/')[-1].split('.')[-2], 'optical_flow'))
            except OSError:
                pass

            try:
                os.removedirs(os.path.join(dst_frame_folder, path.split('/')[-1].split('.')[-2]))
            except OSError:
                pass


        clip_paths = glob.glob(clip_dst_folder + '/*')

        print ''
        print 'Checking Clips Start'
        print ''

        check_pool = Pool(processes=check_worker)
        check_pool.map(checkClips, zip(clip_paths, [temp_extract_folder]*len(clip_paths)))
        check_pool.close()


        clip_counter = len(glob.glob(clip_dst_folder + '/*'))
        if clip_counter >= needed_clip_number:
            break


        remaining_clip_results = needed_clip_number - clip_counter
        max_results_per_keyword = int(remaining_clip_results / len(keywords) * 1.0)


    if not os.path.exists(video_ids_file):
        video_ids_fp = open(video_ids_file, 'w')
    else:
        video_ids_fp = open(video_ids_file, 'a')

    for id in video_ids:
        video_ids_fp.write('{}\n'.format(id))
    video_ids_fp.close()


if __name__ == '__main__':
    youtube_crawler()
