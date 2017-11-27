import cv2
import sys
sys.path.append("../lib/caffe-action/python")
import caffe
import os
import numpy as np
import time
import glob
import math
import matplotlib.pyplot as plt
import threading
from sklearn import mixture
from pipes import quote
from caffe.io import oversample
from utils.io import flow_stack_oversample, fast_list2arr
from multiprocessing import Pool, Value, Lock, current_process, Manager
from ctypes import c_int


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
    spatial_net_weights = "../models/twis_caffemodels/v2/twis_spatial_net_v2.caffemodel"
    temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
    temporal_net_weights = "../models/twis_caffemodels/v2/twis_temporal_net_v2.caffemodel"

    device_id = 0

    spatial_net_gpu = CaffeNet(spatial_net_proto, spatial_net_weights, device_id)
    temporal_net_gpu = CaffeNet(temporal_net_proto, temporal_net_weights, device_id)

    spatial_net_cpu = CaffeNet(spatial_net_proto, spatial_net_weights, -1)
    temporal_net_cpu = CaffeNet(temporal_net_proto, temporal_net_weights, -1)


def build_temporal_net(version=2):
    global temporal_net_gpu
    global temporal_net_cpu

    temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
    temporal_net_weights = "../models/twis_caffemodels/v{0}/twis_temporal_net_v{0}.caffemodel".format(version)

    device_id = 0

    temporal_net_gpu = CaffeNet(temporal_net_proto, temporal_net_weights, device_id)
    temporal_net_cpu = CaffeNet(temporal_net_proto, temporal_net_weights, -1)


def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


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
        scan_pool.map(temporalScanVideo, zip(scan_input_paths, [video_frame_count] * video_frame_count,
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
                    if current_dominant_index == violence_index and current_score[
                        current_dominant_index] >= whole_violence_avg_score:
                        violence_maxima.append([si + 1, current_score[current_dominant_index]])
                    elif current_dominant_index == normal_index and current_score[
                        current_dominant_index] >= whole_normal_avg_score:
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
        for violence_maxi in violence_maxima:
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
        for normal_maxi in normal_maxima:
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
                        violence_selected_slices.append(current_ss + [violence_index])
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
                    violence_selected_slices.append(current_ss + [violence_index])
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
                        normal_selected_slices.append(current_ss + [normal_index])
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
                    normal_selected_slices.append(current_ss + [normal_index])
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
                                              '{}_{}_c{:03d}.avi'.format(video_name, 'violence',
                                                                         violence_selected_slices.index(slice) + 1))
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

    return [whole_scores, violence_selected_slices, normal_selected_slices]


def makeActionProposalsForViolence(video_path, frame_src_path, clip_dst_path='', saved_as_files=False, plt_showed=False):
    global global_num_worker
    global global_num_using_gpu
    global len_scan_list
    global scan_counter
    global scan_counter_lock
    global whole_scores

    global_num_worker = 12
    global_num_using_gpu = 6
    violence_index = 0

    scan_counter = Value(c_int)
    scan_counter_lock = Lock()

    manager = Manager()
    whole_scores = manager.list()

    scan_input_path = os.path.join(frame_src_path)
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
        scan_pool.map(temporalScanVideo, zip([scan_input_path]*len(scan_indices), [video_frame_count] * len(scan_indices),
                                     scan_indices))
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

        whole_violence_scores = []
        for ii in xrange(len(whole_scores)):
            whole_violence_scores.append([ii+1, whole_scores[ii][violence_index]])

        n_components = max(1, int(float(len(whole_scores)) / float(video_fps)))

        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='spherical')
        gmm.fit(whole_violence_scores)

        means = gmm.means_
        covariances = gmm.covariances_


        components = []
        for ii in xrange(len(means)):
            if means[ii][1] >= whole_violence_avg_score:
                components.append([means[ii][0], means[ii][1], covariances[ii]])

        components.sort()

        big_slices = []
        for component in components:
            lower_bound = max(1, int(component[0] - component[2]))
            upper_bound = min(video_frame_count, int(math.ceil(component[0] + component[2])))
            big_slices.append([lower_bound, upper_bound, component[1]])


        selected_slices = []
        isLeft = False
        if len(big_slices) >= 1:
            current_ss = big_slices[0]
            bs_index = 1
            while True:
                if bs_index >= len(big_slices):
                    if isLeft == True:
                        selected_slices.append(current_ss)
                    break

                isLeft = False

                current_slice_length = current_ss[1] - current_ss[0] + 1
                compare_slice_length = big_slices[bs_index][1] - big_slices[bs_index][0] + 1
                end_index = current_ss[1]
                start_index = big_slices[bs_index][0]
                union_length = end_index - start_index + 1

                if union_length >= current_slice_length/2 or union_length >= compare_slice_length/2:
                    current_ss = [current_ss[0], big_slices[bs_index][1]]
                    isLeft=True
                    bs_index += 1
                else:
                    selected_slices.append(current_ss)
                    current_ss = big_slices[bs_index]
                    isLeft = True
                    bs_index += 1

        # removed_slices = []
        # for slice in selected_slices:
        #     duration = slice[1] - slice[0] + 1
        #     if duration < video_fps:
        #         removed_slices.append(slice)
        #
        # for slice in removed_slices:
        #     selected_slices.remove(slice)


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


        plt = None
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


    return [whole_scores, selected_slices, plt]


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
    whole_score = [rgb_score[i]*0.0 + flow_score[i]*5.0 for i in xrange(len(rgb_score))]
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


def extractFlow(index):
    global frames
    global frame_dst_folder

    if index == 1:
        frame_avg = np.average(frames[index-1])
        first_frame = cv2.cvtColor(np.array(np.multiply(np.ones(frames[index-1].shape), frame_avg), dtype='uint8'), cv2.COLOR_BGR2GRAY)
        second_frame = cv2.cvtColor(frames[index-1], cv2.COLOR_BGR2GRAY)
    else:
        first_frame = cv2.cvtColor(frames[index-2], cv2.COLOR_BGR2GRAY)
        second_frame = cv2.cvtColor(frames[index-1], cv2.COLOR_BGR2GRAY)

    x_flow_path = os.path.join(frame_dst_folder, 'flows', 'flow_x_{}.jpg'.format(index))
    y_flow_path = os.path.join(frame_dst_folder, 'flows', 'flow_y_{}.jpg'.format(index))

    extractor = cv2.DualTVL1OpticalFlow_create()
    flows = extractor.calc(first_frame, second_frame, None)

    bound = 20.0

    x_flow = flows[:, :, 0]
    y_flow = flows[:, :, 1]

    x_flow = np.clip(x_flow, -bound, bound)
    x_flow += bound
    x_flow = np.divide(x_flow, bound*2.0)
    x_flow = np.multiply(x_flow, 255.0)
    x_flow = np.array(x_flow, dtype='uint8')

    y_flow = np.clip(y_flow, -bound, bound)
    y_flow += bound
    y_flow = np.divide(y_flow, bound * 2.0)
    y_flow = np.multiply(y_flow, 255.0)
    y_flow = np.array(y_flow, dtype='uint8')

    cv2.imwrite(x_flow_path, x_flow)
    cv2.imwrite(y_flow_path, y_flow)


def evaluator(eval_items):
    global temporal_net_gpu
    global whole_scores

    input_path = eval_items[0]
    frame_count = eval_items[1]
    index = eval_items[2]

    current = current_process()
    current_id = current._identity[0] -1

    temporal_net = temporal_net_gpu
    score_layer_name = 'fc-twis'


    flow_stack = []
    for i in range(-2, 3, 1):
        if index + i >= 1 and index + i <= frame_count:
            x_flow_path = os.path.join(input_path, 'optical_flow/flow_x_{:05d}.jpg').format(index + i)
            y_flow_path = os.path.join(input_path, 'optical_flow/flow_y_{:05d}.jpg').format(index + i)

            if not os.path.exists(x_flow_path) or not os.path.exists(y_flow_path):
                if i==0:
                    extractFlow(index)
                else:
                    while True:
                        if os.path.exists(x_flow_path) and os.path.exists(y_flow_path):
                            time.sleep(0.1)
                            break

            x_flow_field = cv2.imread(x_flow_path, cv2.IMREAD_GRAYSCALE)
            y_flow_field = cv2.imread(y_flow_path, cv2.IMREAD_GRAYSCALE)

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


def extractOpticalFlowSimple(video_file_path, extract_dst_path, processor_type='gpu'):
    video_name = video_file_path.split('/')[-1]

    new_size = (340, 256)
    out_format = 'dir'
    df_path = "../lib/dense_flow/"
    dev_id = 0

    out_full_path = extract_dst_path
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


def classifyVideo(video_path, out_video_path, net_version=2, extract_path='', clip_dst_path='', isTobeClipped=False, isPloted=False):
    perFrameEvaluation = False

    extractOpticalFlowSimple(video_file_path=video_path, extract_dst_path=extract_path, processor_type='gpu')

    build_temporal_net(net_version)

    [whole_scores, selected_slices, plt] = makeActionProposalsForViolence(video_path=video_path, frame_src_path=extract_path)

    sum_scores = 0
    for score in whole_scores:
        sum_scores = max(score)
    whole_avg_score = sum_scores / float(len(whole_scores))

    selected_slices.sort()

    rgb_path_prefix = os.path.join(extract_path, 'images/img')
    flow_path_prefix = os.path.join(extract_path, 'optical_flow/flow')

    flow_bound = 20

    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened():
        video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
        video_fps = 2
        video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_width = 340
        video_height = 256
        video_cap.release()

        flow_step = 5

        frames = []

        if not perFrameEvaluation:
            frame_index = 1
            for slice in selected_slices:
                while frame_index < slice[0]:
                    print frame_index
                    frame = cv2.imread('{}_{:05d}.jpg'.format(rgb_path_prefix, frame_index))
                    flow_x = cv2.imread('{}_x_{:05d}.jpg'.format(flow_path_prefix, frame_index), cv2.IMREAD_GRAYSCALE)
                    flow_y = cv2.imread('{}_y_{:05d}.jpg'.format(flow_path_prefix, frame_index), cv2.IMREAD_GRAYSCALE)

                    flow_x = np.divide(flow_x, 255.0)
                    flow_x = np.multiply(flow_x, float(flow_bound * 2))
                    flow_x -= float(flow_bound)
                    flow_x = np.clip(flow_x, -20.0, 20.0)

                    flow_y = np.divide(flow_y, 255.0)
                    flow_y = np.multiply(flow_y, float(flow_bound * 2))
                    flow_y -= float(flow_bound)
                    flow_y = np.clip(flow_y, -20.0, 20.0)

                    for row in range(0, frame.shape[0], flow_step):
                        for col in range(0, frame.shape[1], flow_step):
                            pt1 = (col, row)
                            pt2 = (int(max(0, min(video_width, col + flow_x[row][col]))),
                                   int(max(0, min(video_height, row + flow_y[row][col]))))
                            cv2.arrowedLine(img = frame, pt1=pt1, pt2 = pt2, color=(255, 0, 255), thickness=1)

                    frame_label = 'Normal'
                    color = (255, 0, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'Frame {:05d} Prediction: {} {:.3f}'.format(
                        frame_index, frame_label, max(whole_scores[frame_index - 1])), (10, 20), font, 0.35, color, 1, cv2.LINE_AA)
                    frames.append(frame)
                    frame_index += 1

                while frame_index <= slice[1]:
                    print frame_index
                    frame = cv2.imread('{}_{:05d}.jpg'.format(rgb_path_prefix, frame_index))
                    flow_x = cv2.imread('{}_x_{:05d}.jpg'.format(flow_path_prefix, frame_index), cv2.IMREAD_GRAYSCALE)
                    flow_y = cv2.imread('{}_y_{:05d}.jpg'.format(flow_path_prefix, frame_index), cv2.IMREAD_GRAYSCALE)

                    flow_x = np.divide(flow_x, 255.0)
                    flow_x = np.multiply(flow_x, float(flow_bound * 2))
                    flow_x -= float(flow_bound)
                    flow_x = np.clip(flow_x, -20.0, 20.0)

                    flow_y = np.divide(flow_y, 255.0)
                    flow_y = np.multiply(flow_y, float(flow_bound * 2))
                    flow_y -= float(flow_bound)
                    flow_y = np.clip(flow_y, -20.0, 20.0)

                    for row in range(0, frame.shape[0], flow_step):
                        for col in range(0, frame.shape[1], flow_step):
                            pt1 = (col, row)
                            pt2 = (int(max(0, min(video_width, col + flow_x[row][col]))),
                                   int(max(0, min(video_height, row + flow_y[row][col]))))
                            cv2.arrowedLine(img = frame, pt1=pt1, pt2 = pt2, color=(255, 0, 255), thickness=1)

                    frame_label = 'Violence'
                    color = (0, 0, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'Frame {:05d} Prediction: {} {:.3f}'.format(
                        frame_index, frame_label, max(whole_scores[frame_index - 1])), (10, 20), font, 0.35, color, 1,
                                cv2.LINE_AA)
                    frames.append(frame)
                    frame_index += 1

                if slice[1] == selected_slices[-1][1]:
                    while frame_index <= video_frame_count:
                        print frame_index
                        frame = cv2.imread('{}_{:05d}.jpg'.format(rgb_path_prefix, frame_index))
                        flow_x = cv2.imread('{}_x_{:05d}.jpg'.format(flow_path_prefix, frame_index),
                                            cv2.IMREAD_GRAYSCALE)
                        flow_y = cv2.imread('{}_y_{:05d}.jpg'.format(flow_path_prefix, frame_index),
                                            cv2.IMREAD_GRAYSCALE)

                        flow_x = np.divide(flow_x, 255.0)
                        flow_x = np.multiply(flow_x, float(flow_bound * 2))
                        flow_x -= float(flow_bound)
                        flow_x = np.clip(flow_x, -20.0, 20.0)

                        flow_y = np.divide(flow_y, 255.0)
                        flow_y = np.multiply(flow_y, float(flow_bound * 2))
                        flow_y -= float(flow_bound)
                        flow_y = np.clip(flow_y, -20.0, 20.0)

                        for row in range(0, frame.shape[0], flow_step):
                            for col in range(0, frame.shape[1], flow_step):
                                pt1 = (col, row)
                                pt2 = (int(max(0, min(video_width, col + flow_x[row][col]))),
                                       int(max(0, min(video_height, row + flow_y[row][col]))))
                                cv2.arrowedLine(img=frame, pt1=pt1, pt2=pt2, color=(255, 0, 255), thickness=1)

                        frame_label = 'Normal'
                        color = (255, 0, 0)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, 'Frame {:05d} Prediction: {} {:.3f}'.format(
                             frame_index, frame_label, max(whole_scores[frame_index-1])), (10, 20), font, 0.35, color, 1,
                                    cv2.LINE_AA)
                        frames.append(frame)
                        frame_index += 1
        else:
            for score in whole_scores:
                ret, frame = video_cap.read()
                if not ret:
                    break

                dominant_score = max(softmax(score))
                dominant_index = np.argmax(score)

                if dominant_score < whole_avg_score:
                    frame_label = 'Background'
                    color = (0, 255, 0)
                elif dominant_index == 0:
                    frame_label = 'Violence'
                    color = (0, 0, 255)
                else:
                    frame_label = 'Normal'
                    color = (255, 0, 0)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                            'Frame {:05d} Prediction: {} {:.3f}'.format(whole_scores.index(score) + 1, frame_label,
                                                                        dominant_score), (10, 20),
                            font, 0.35, color, 1, cv2.LINE_AA)

                frames.append(frame)

        video_writer = cv2.VideoWriter(out_video_path, video_fourcc, video_fps, (video_width, video_height))
        for frame in frames:
            video_writer.write(frame)

        video_writer.release()


class ServerFromVideo():
    def __init__(self):
        global frames
        global frame_dst_folder

        manager = Manager()
        frames = manager.list()

        frame_dst_folder = '../temp'
        if not os.path.exists(frame_dst_folder):
            try:
                os.makedirs(frame_dst_folder)
            except OSError:
                pass

        image_dst_folder = os.path.join(frame_dst_folder, 'images')
        flow_dst_folder = os.path.join(frame_dst_folder, 'flows')

        try:
            os.mkdir(image_dst_folder)
            os.mkdir(flow_dst_folder)
        except OSError:
            pass

        # build_temporal_net(2)

        self.video_cap = cv2.VideoCapture(0)
        self.video_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)

        thread = threading.Thread(target=self.run)
        thread.start()


    def run(self):
        index = 1
        # self.distributor_id = Distributor()
        first_index = 1
        while True:
            count = 0
            while True:
                ok, frame = self.video_cap.read()
                if ok:
                    count += 1
                    frames.append(frame)

                    image_path = os.path.join(frame_dst_folder, 'images', 'img_{}.jpg'.format(first_index + count))
                    cv2.imwrite(image_path, frames[index - 1])

                    if count >= 100:
                        break
                else:
                    break

            for index in range(first_index, first_index+count+1, 1):
                extractFlow(index)

            first_index += count



        print ''
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


class Distributor():
    def __init__(self):
        global frames

        global eval_start_index
        global eval_end_index
        global eval_start_lock
        global eval_end_lock

        global current_process_counter

        eval_start_index = Value(c_int)
        eval_start_index.value = 1
        eval_end_index = Value(c_int)
        eval_start_lock = Lock()
        eval_end_lock = Lock()

        current_process_counter = Value(c_int)
        self.max_process_number = 8

        thread = threading.Thread(target=self.run)
        thread.start()

    def run(self):
        while True:
            while current_process_counter.value >= self.max_process_number:
                time.sleep(0.5)

            eval_end_index.value = len(frames) - 1
            eval_strat_index.value = 1


if __name__ == '__main__':
    server_id = ServerFromVideo()

    while True:
        time.sleep(100)
