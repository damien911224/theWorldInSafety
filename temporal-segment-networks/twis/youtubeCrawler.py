import cv2
import sys
sys.path.append("/home/damien/temporal-segment-networks/lib/caffe-action/python")
import caffe
import os
import glob
import random
import string
import numpy as np
import time
from pipes import quote
from sklearn.metrics import confusion_matrix
from caffe.io import oversample
from utils.io import flow_stack_oversample, fast_list2arr
from utils.video_funcs import default_aggregation_func
from multiprocessing import Pool, Value, Lock, current_process, Manager
from ctypes import c_int, c_float

from pytube import YouTube
from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser


# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = "AIzaSyDSodYpZo2uf6NLxrdIc-bcmsSR7BQp9pI"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

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
    spatial_net_proto = "../models/twis/tsn_bn_inception_rgb_deploy.prototxt"
    spatial_net_weights = "../models/twis_caffemodels/v1/twis_spatial_net_v1.caffemodel"
    temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
    temporal_net_weights = "../models/twis_caffemodels/v1/twis_temporal_net_v1.caffemodel"

    device_id = 0

    spatial_net_gpu = CaffeNet(spatial_net_proto, spatial_net_weights, device_id)
    temporal_net_gpu = CaffeNet(temporal_net_proto, temporal_net_weights, device_id)

    spatial_net_cpu = CaffeNet(spatial_net_proto, spatial_net_weights, -1)
    temporal_net_cpu = CaffeNet(temporal_net_proto, temporal_net_weights, -1)


def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


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


def download_youtube_video(video_url, dst_folder):
    try:
        youtube = YouTube(video_url)
    except:
        print ' !! YOUTUBE ERROR !!'
        return

    filename = youtube.video_id
    check_save_path = glob.glob(os.path.join(dst_folder, filename) + '.*')
    if len(check_save_path) == 0:
        youtube.set_filename(filename)

        video = youtube.get_videos()[-2]

        try:
            video.download(dst_folder)
        except:
            print ' !! YOUTUBE DOWNLOAD ERROR !!'
            return


def makeActionProposalsForViolence(video_path, frame_src_path, clip_dst_path='', savedAsFiles=True):
    global global_num_worker
    global global_num_using_gpu
    global len_scan_list
    global scan_counter
    global scan_counter_lock
    global whole_scores

    global_num_worker = 5
    global_num_using_gpu = 1
    violence_index = 0

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

        whole_score_sum = [0.0] * len(whole_scores[0])
        for score in whole_scores:
            for i in xrange(len(whole_scores[0])):
                whole_score_sum[i] += score[i]
        whole_average_score = [whole_score_sum[i] / float(len(whole_scores)) for i in xrange(len(whole_scores[0]))]
        whole_dominant_score = max(softmax(whole_average_score))

        previous_dominant_index = -1
        selected_slices = []
        temp_group = []
        for index in xrange(len(whole_scores)):
            current_dominant_score = max(softmax(whole_scores)[index])
            current_dominant_index = np.argmax(whole_scores[index])

            if current_dominant_score >= whole_dominant_score * 0.7:
                if (not len(temp_group) == 0 and previous_dominant_index == current_dominant_index) or len(temp_group) == 0:
                    temp_group.append(index + 1)
                    previous_dominant_index = current_dominant_index
                elif len(temp_group) >= 5:
                    temp_start = temp_group[0]
                    temp_end = temp_group[-1]
                    temp_sum = [0.0] * len(whole_scores[0])
                    for temp_index in temp_group:
                        for i in xrange(len(whole_scores[0])):
                            temp_sum[i] = whole_scores[temp_index - 1][i]

                    temp_avg = [temp_sum[i] / float(len(temp_group)) for i in xrange(len(temp_sum))]
                    selected_slices.append([temp_avg, temp_start, temp_end])
                    temp_group = []
                    previous_dominant_index == -1
                else:
                    temp_group = []
                    previous_dominant_index == -1

            elif len(temp_group) >= 5:
                temp_start = temp_group[0]
                temp_end = temp_group[-1]
                temp_sum = [0.0] * len(whole_scores[0])
                for temp_index in temp_group:
                    for i in xrange(len(whole_scores[0])):
                        temp_sum[i] += whole_scores[temp_index - 1][i]

                temp_avg = [temp_sum[i] / float(len(temp_group)) for i in xrange(len(temp_sum))]
                selected_slices.append([temp_avg, temp_start, temp_end])
                temp_group = []
            else:
                temp_group = []

        for slice in selected_slices:
            ss_index = np.argmax(slice[0])
            if savedAsFiles == True and ss_index == violence_index:
                if clip_dst_path == '':
                    clip_default_folder = ''
                    for path in video_path.split('/')[:-1]:
                        clip_default_folder += path + '/'
                    clip_default_folder = os.path.join(clip_default_folder[:-1], video_path.split('/')[-1].split('.')[-2])
                else:
                    clip_default_folder = os.path.join(clip_dst_path,
                                                       video_path.split('/')[-1].split('.')[-2])
                try:
                    os.mkdir(clip_default_folder)
                except OSError:
                    pass
                clip_save_path = os.path.join(clip_default_folder,
                                              'c{:03d}.avi'.format(selected_slices.index(slice) + 1))
                clipping(video_path, clip_save_path, slice[1], slice[2])

    video_cap.release()

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
    whole_score = [rgb_score[i] + flow_score[i]*5.0 for i in xrange(len(rgb_score))]
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


def extractOpticalFlow(video_file_path, extract_dst_path, processor_type='cpu'):
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
            print '!! EXTRACT PASS !!'
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

    print '{} EXTRACT DONE'.format(video_file_path)


def youtube_crawler():
    temp_dst_folder = '/home/damien/temp/youtube'
    video_dst_folder = temp_dst_folder + '/videos'
    clip_dst_folder = temp_dst_folder + '/clips'
    dst_frame_folder = temp_dst_folder + '/frames'
    max_results = 5
    search_epochs = 1

    keywords = \
        [
            'UFC highlight',
            'how to kick',
            'how to punch',
            'how to fight'
            'kickboxing'
        ]

    # for keyword in keywords:
    #     token = None
    #     for iter in xrange(search_epochs):
    #         if token == 'last_page':
    #             break
    #         token, video_urls = youtube_search(keyword=keyword, max_results=max_results, token=token)
    #         for url in video_urls:
    #             download_youtube_video(url, video_dst_folder)

    downloaded_paths = glob.glob(video_dst_folder + '/*')

    build_net()

    for path in downloaded_paths:
        extractOpticalFlow(path, dst_frame_folder)
        makeActionProposalsForViolence(path, dst_frame_folder, clip_dst_folder)


if __name__ == '__main__':
    youtube_crawler()