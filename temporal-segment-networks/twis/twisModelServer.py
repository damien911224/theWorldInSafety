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
import copy_reg, types
import gc
from shutil import copyfile
from shutil import rmtree
import datetime



class CaffeNet(object):
    def __init__(self, net_proto, net_weights, device_id, input_size=None):
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
            transformer.set_mean('data',
                                 np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        else:
            pass  # non RGB data need not use transformer

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
                    resized_frame = [cv2.resize(x, (0, 0), fx=1.0 / scale, fy=1.0 / scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)

        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()


    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):
        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size, interpolation=cv2.INTER_AREA) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()


    def predict_simple_single_flow_stack(self, frame, score_name, over_sample=False, frame_size=None):
        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size, interpolation=cv2.INTER_AREA) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()


class ServerFromVideo():
    def __init__(self):
        self.root_folder = os.path.abspath('../progress')
        if not os.path.exists(self.root_folder):
            try:
                os.makedirs(self.root_folder)
            except OSError:
                pass

        self.session_folder = os.path.join(self.root_folder, '{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        self.image_folder = os.path.join(self.session_folder, 'images')
        self.flow_folder = os.path.join(self.session_folder, 'flows')
        self.clip_folder = os.path.join(self.session_folder, 'clips')
        self.keep_folder = os.path.join(self.session_folder, 'keep')

        previous_session_folders = glob.glob(os.path.join(self.root_folder, '20*'))
        for folder in previous_session_folders:
            rmtree(folder, ignore_errors=True)

        try:
            os.mkdir(self.session_folder)
        except OSError:
            pass

        try:
            os.mkdir(self.image_folder)
        except OSError:
            pass

        try:
            os.mkdir(self.flow_folder)
        except OSError:
            pass

        try:
            os.mkdir(self.clip_folder)
        except OSError:
            pass

        try:
            os.mkdir(self.keep_folder)
        except OSError:
            pass

        self.web_cam = False
        if self.web_cam:
            self.test_video_name = 'Webcam.mp4'
        else:
            self.test_video_name = 'test_5.avi'
        self.model_version = 3
        self.build_temporal_net(self.model_version)

        self.show_size = ( 600, 450 )
        self.new_size = ( 224, 224 )
        self.temporal_width = 1
        self.print_term = 50
        self.fps = 25.0
        self.start_index = 1
        self.wait_time = 1.0 / self.fps
        self.dumped_index = 0

        self.print_lock = Lock()

        self.extractor = Extractor(self)
        self.extractor_thread = threading.Thread(target=self.extractor.run, name='Extractor')

        self.server_thread = threading.Thread(target=self.run, name='Server')
        self.server_thread.start()


    def run(self):
        if self.web_cam:
            video_cap = cv2.VideoCapture(0)
        else:
            video_cap = cv2.VideoCapture(os.path.join(self.root_folder, 'test_videos', self.test_video_name))
        self.video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = video_cap.get(cv2.CAP_PROP_FPS)

        self.extractor_thread.start()

        while True:
            frames = []
            while True:
                ok, frame = video_cap.read()
                if ok:
                    frames.append(frame)
                    time.sleep(self.wait_time)

                    if len(frames) >= self.temporal_width:
                        break

            self.dumpFrames(frames)
            del frames
            gc.collect()

            self.start_index += self.temporal_width
            self.dumped_index = self.start_index - 1


    def build_temporal_net(self, version=3):
        global temporal_net_gpu
        global temporal_net_cpu


        self.temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
        self.temporal_net_weights = "../models/twis_caffemodels/v{0}/twis_temporal_net_v{0}.caffemodel".format(
            version)

        device_id = 0

        temporal_net_gpu = CaffeNet(self.temporal_net_proto, self.temporal_net_weights, device_id)
        temporal_net_cpu = CaffeNet(self.temporal_net_proto, self.temporal_net_weights, -1)


    def dumpFrames(self, frames):
        end_index = self.start_index + len(frames) - 1
        if end_index % self.print_term == 0:
            with self.print_lock:
                print '{:10s}|{:12s}| Until {:07d}'.format('Server', 'Dumping', end_index)

        index = self.start_index
        for frame in frames:
            file_name = os.path.join(self.image_folder, 'show_{:07d}.jpg'.format(index))
            new_frame = cv2.resize(frame, self.show_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_name, new_frame)

            file_name = os.path.join(self.image_folder, 'img_{:07d}.jpg'.format(index))
            new_frame = cv2.resize(frame, self.new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_name, new_frame)
            index += 1


class Extractor():
    def __init__(self, server):
        self.server = server

        self.evaluator = Evaluator(self.server, self)
        self.evaluator_thread = threading.Thread(target=self.evaluator.run, name='Evaluator')

        self.cmd_lock = Lock()


    def run(self):
        self.start_index = 2
        self.extracted_index = 0
        self.wait_time = 0.3

        self.evaluator_thread.start()

        while True:
            while self.server.dumped_index <= self.extracted_index:
                time.sleep(self.wait_time)

            self.end_index = self.server.dumped_index

            with self.server.print_lock:
                print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Extractor', 'Extracting', self.start_index, self.end_index)

            self.extractOpticalFlows(self.server.image_folder, self.start_index, self.end_index, self.server.flow_folder)

            self.start_index = self.end_index + 1
            self.extracted_index = self.end_index


    def extractOpticalFlows(self, frame_path, start_index, end_index, flow_dst_folder):
        new_size = (0, 0)
        out_format = 'dir'
        root_abs_path = os.path.abspath('/home/damien/jointFlowNetwork')
        df_path = os.path.join(root_abs_path, 'lib', 'dense_flow')
        dev_id = 0

        image_path = 'None'
        video_file_path = 'Per Frame'
        dump = -1
        frame_prefix = '{}/img'.format(frame_path)
        frame_count = end_index - start_index + 1
        optical_flow_x_path = '{}/flow_x'.format(flow_dst_folder)
        optical_flow_y_path = '{}/flow_y'.format(flow_dst_folder)

        cmd = os.path.join(
            df_path + '/build/extract_gpu') + ' -v {} -f {} -p {} -e {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {} -a {} -c {}'.format(
            quote(video_file_path), quote(frame_prefix), start_index, end_index, quote(optical_flow_x_path),
            quote(optical_flow_y_path), quote(image_path),
            dev_id,
            out_format, new_size[0], new_size[1], dump, frame_count)

        with self.cmd_lock:
            os.system(cmd)
            sys.stdout.flush()


class Evaluator():
    def __init__(self, server, extractor):
        self.server = server
        self.extractor = extractor

        self.num_workers = 12
        self.num_using_gpu = 8

        global scanning_pool
        scanning_pool = Pool(processes=self.num_workers)

        copy_reg.pickle(types.MethodType, self._pickle_method)

        self.scanner = Scanner(self.server.flow_folder, self.num_workers, self.num_using_gpu)

        self.analyzer = Analyzer(self.server, self.extractor, self)
        self.analyzer_thread = threading.Thread(target=self.analyzer.run, name='Analyzer')

        self.secretary = Secretary(self.server, self.extractor, self, self.analyzer)
        self.secretary_thread = threading.Thread(target=self.secretary.run, name='Secretary')

        self.closer = Closer(self.server, self.extractor, self, self.analyzer, self.secretary)
        self.closer_thread = threading.Thread(target=self.closer.run, name='Closer')


    def run(self):
        self.start_index = 2
        self.scanned_index= 1
        self.temporal_gap = 2
        self.actual_start_index = 2
        self.wait_time = 0.3

        self.scores = []

        self.analyzer_thread.start()
        self.secretary_thread.start()
        self.closer_thread.start()

        while True:
            while self.extractor.extracted_index - self.temporal_gap <= self.scanned_index:
                time.sleep(self.wait_time)

            self.actual_extracted_index = self.extractor.extracted_index
            self.end_index = self.actual_extracted_index - self.temporal_gap

            with self.server.print_lock:
                print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Evaluator', 'Evaluating', self.start_index, self.end_index)

            scan_start_time = time.time()

            return_scores = []
            return_scores += self.scanner.scan(self.start_index, self.end_index, self.actual_extracted_index)

            self.scan_time = (time.time() - scan_start_time) / len(return_scores)

            self.scores += return_scores
            self.analyzer.keeping_scores += return_scores
            self.secretary.showing_scores += return_scores
            del return_scores

            self.scanned_index = self.end_index
            self.start_index = self.end_index + 1

            gc.collect()


    def _pickle_method(self,m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)


class Scanner():
    def __init__(self, flow_folder, num_workers, num_using_gpu):
        self.flow_folder = flow_folder
        self.num_workers = num_workers
        self.num_using_gpu = num_using_gpu


    def scan(self, start_index, end_index, actual_extracted_index):
        manager = Manager()
        scan_scores = manager.list()

        indices = range(start_index, end_index + 1, 1)

        for i in xrange(len(indices)):
            scan_scores.append([0.0, 0.0])

        scanning_pool.map(self.temporalScanVideo,
                          zip([self.flow_folder] * len(indices),
                              [self.num_workers] * len(indices),
                              [self.num_using_gpu] * len(indices),
                              [start_index] * len(indices),
                              [actual_extracted_index] * len(indices),
                              [scan_scores] * len(indices),
                              indices))
        return_scores = []
        return_scores += scan_scores

        return return_scores


    def temporalScanVideo(self, scan_items):
        input_path = scan_items[0]
        num_workers = scan_items[1]
        num_using_gpu = scan_items[2]
        start_index = scan_items[3]
        frame_count = scan_items[4]
        scan_scores = scan_items[5]
        index = scan_items[6]

        current = current_process()
        current_id = current._identity[0] - 1

        if current_id % num_workers < num_using_gpu:
            temporal_net = temporal_net_gpu
        else:
            temporal_net = temporal_net_cpu

        score_layer_name = 'fc-twis'

        flow_stack = []
        for i in range(-2, 3, 1):
            if index + i >= 2 and index + i <= frame_count:
                x_flow_field = cv2.imread(
                    os.path.join(input_path, 'flow_x_{:07d}.jpg').format(index + i),
                    cv2.IMREAD_GRAYSCALE)
                y_flow_field = cv2.imread(
                    os.path.join(input_path, 'flow_y_{:07d}.jpg').format(index + i),
                    cv2.IMREAD_GRAYSCALE)
                flow_stack.append(x_flow_field)
                flow_stack.append(y_flow_field)
            elif index + i < 2:
                x_flow_field = cv2.imread(
                    os.path.join(input_path, 'flow_x_{:07d}.jpg').format(2),
                    cv2.IMREAD_GRAYSCALE)
                y_flow_field = cv2.imread(
                    os.path.join(input_path, 'flow_y_{:07d}.jpg').format(2),
                    cv2.IMREAD_GRAYSCALE)
                flow_stack.append(x_flow_field)
                flow_stack.append(y_flow_field)
            else:
                x_flow_field = cv2.imread(
                    os.path.join(input_path, 'flow_x_{:07d}.jpg').format(frame_count),
                    cv2.IMREAD_GRAYSCALE)
                y_flow_field = cv2.imread(
                    os.path.join(input_path, 'flow_y_{:07d}.jpg').format(frame_count),
                    cv2.IMREAD_GRAYSCALE)
                flow_stack.append(x_flow_field)
                flow_stack.append(y_flow_field)

        flow_score = \
            temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, over_sample=False,
                                                   frame_size=None)[0].tolist()

        scan_scores[index - start_index] = [flow_score[i] * 5.0 for i in xrange(len(flow_score))]


class Analyzer():
    def __init__(self, server, extractor, evaluator):
        self.server = server
        self.extractor = extractor
        self.evaluator = evaluator


    def run(self):
        self.analyzing_start_index = 0
        self.analyzed_index = 1
        self.wait_time = 0.2
        self.violence_index = 0
        self.normal_index = 1
        self.lower_bound = 0.0
        self.max_lower_bound = 2.0
        self.variance_factor = 2.0
        self.max_falling_count = 5
        self.falling_counter = 0
        self.median_kernal_size = 7

        self.real_base = 2
        self.not_yet = False
        self.max_iter = 500

        self.keeping_scores = []
        self.keeping_base = 2

        self.keep_number = 1


        while True:
            while self.analyzed_index >= self.evaluator.scanned_index:
                time.sleep(self.wait_time)

            self.analyzed_index = self.evaluator.scanned_index
            self.analyzing_end_index = max(len(self.evaluator.scores) - 1, 0)
            self.not_yet = False

            with self.server.print_lock:
                print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Analyzer', 'Analyzing',
                                                             self.analyzing_start_index + self.real_base,
                                                             self.analyzing_end_index + self.real_base)


            while True:
                clip_start_index = self.analyzing_start_index

                if clip_start_index >= self.analyzing_end_index:
                    break

                for index in range(self.analyzing_start_index, self.analyzing_end_index + 1, 1):
                    clip_start_index = index
                    if self.evaluator.scores[index][self.violence_index] >= self.lower_bound:
                        break

                maxima = []
                clip_end_index = min(clip_start_index + 1, self.analyzing_end_index)

                for index in range(clip_start_index + 1, self.analyzing_end_index + 1, 1):
                    clip_end_index = index
                    if self.evaluator.scores[index][self.violence_index] < self.lower_bound:
                        self.falling_counter += 1
                    else:
                        self.falling_counter = 0

                    if self.falling_counter >= self.max_falling_count:
                        clip_end_index -= min(self.falling_counter/2 - 1, self.analyzing_end_index)
                        self.falling_counter = 0
                        break

                    if index >= 1 and index < self.analyzing_end_index:
                        previous_score = self.evaluator.scores[index - 1][self.violence_index]
                        next_score = self.evaluator.scores[index + 1][self.violence_index]
                        current_score = self.evaluator.scores[index][self.violence_index]
                        if previous_score < current_score and next_score < current_score:
                            if current_score >= self.max_lower_bound:
                                maxima.append([index + self.real_base, current_score])

                if clip_end_index >= self.analyzing_end_index:
                    self.not_yet = True

                selected_slices = []
                if not self.not_yet:
                    big_slices = []
                    selected_slices = []
                    number_of_components = max(1, len(maxima))

                    if len(maxima) == 0:
                        maxima = None

                    if maxima is not None and clip_end_index - clip_start_index+ 1 >= len(maxima):
                        gmm_elements = []
                        scores = np.asarray(self.evaluator.scores[clip_start_index:clip_end_index+1])

                        for index in range(self.median_kernal_size/2, len(scores), 1):
                            if index + self.median_kernal_size/2 > len(scores):
                                break
                            scores[index][self.violence_index] \
                                = np.median(scores[index:index + self.median_kernal_size / 2 + 1, self.violence_index])
                            scores[index][self.normal_index] \
                                = np.median(scores[index:index + self.median_kernal_size / 2 + 1, self.normal_index])


                        for index in range(0, len(scores), 1):
                            current_score = scores[index][self.violence_index]
                            gmm_elements.append([index + clip_start_index + self.real_base, current_score])

                        del scores

                        if not len(gmm_elements) == 0:
                            gmm = mixture.GaussianMixture(n_components=number_of_components, covariance_type='spherical',
                                                          max_iter=self.max_iter, means_init=maxima)
                            gmm.fit(gmm_elements)

                            means = gmm.means_
                            covariances = gmm.covariances_

                            components = []
                            for ii in xrange(len(means)):
                                if means[ii][1] >= self.max_lower_bound:
                                    components.append([means[ii][0], means[ii][1],
                                                           covariances[ii] * self.variance_factor])

                            del maxima
                            del means
                            del covariances
                            del gmm_elements

                            components.sort()

                            for component in components:
                                lower_bound = max(self.keeping_base, int(component[0] - component[2]))
                                upper_bound = min(self.analyzing_end_index + self.real_base,
                                                  int(math.ceil(component[0] + component[2])))
                                big_slices.append([lower_bound, upper_bound])

                            del components

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
                                        if compare_start - current_end < self.server.video_fps:
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

                clips = []
                for slice in selected_slices:
                    keep_folder = os.path.join(self.server.keep_folder, 'keep_{:07d}'.format(self.keep_number))
                    self.keep_number += 1
                    try:
                        os.mkdir(keep_folder)
                    except OSError:
                        pass

                    clip = dict()
                    clip['keep_folder'] = keep_folder
                    clip['time_intervals'] = [slice[0], slice[1]]
                    clip['frames'] = []

                    for index in range(slice[0], slice[1]+1, 1):
                        image_src_path = os.path.join(self.server.image_folder, 'show_{:07d}.jpg'.format(index))
                        flow_x_src_path = os.path.join(self.server.flow_folder, 'flow_x_{:07d}.jpg'.format(index))
                        flow_y_src_path = os.path.join(self.server.flow_folder, 'flow_y_{:07d}.jpg'.format(index))

                        image_dst_path = os.path.join(keep_folder, 'show_{:07d}.jpg'.format(index))
                        flow_x_dst_path = os.path.join(keep_folder, 'flow_x_{:07d}.jpg'.format(index))
                        flow_y_dst_path = os.path.join(keep_folder, 'flow_y_{:07d}.jpg'.format(index))

                        copyfile(image_src_path, image_dst_path)
                        copyfile(flow_x_src_path, flow_x_dst_path)
                        copyfile(flow_y_src_path, flow_y_dst_path)

                        score = self.keeping_scores[index -self.keeping_base]

                        frame = dict()
                        frame['index'] = index
                        frame['score'] = score
                        frame['image'] = image_dst_path
                        frame['flows'] = [flow_x_dst_path, flow_y_dst_path]
                        clip['frames'].append(frame)
                        del frame

                    clips.append(clip)
                    del clip

                del selected_slices

                if len(clips) >= 1:
                    self.evaluator.closer.clips += clips
                del clips


                if self.not_yet:
                    self.remove_amount = clip_start_index
                else:
                    self.remove_amount = clip_end_index + 1

                if self.remove_amount >= 1:
                    del self.evaluator.scores[0:self.remove_amount]
                    self.analyzing_end_index -= self.remove_amount
                    self.real_base += self.remove_amount
                    self.remove_amount = 0

                gap_of_keeping_and_current =  (self.analyzing_start_index + self.real_base) - (self.keeping_base)
                if gap_of_keeping_and_current >= int(100.0 * server.video_fps):
                    del self.keeping_scores[0:gap_of_keeping_and_current]
                    self.keeping_base += gap_of_keeping_and_current

                if self.not_yet:
                    break

                gc.collect()


class Secretary():
    def __init__(self, server, extractor, evaluator, analyzer):
        self.server = server
        self.extractor = extractor
        self.evaluator = evaluator
        self.analyzer = analyzer

        self.progress_viewer = self.Viewer(self)
        self.progress_viewer.window_name = 'Progress Viewer'
        self.progress_viewer.window_position = (0, 0)
        self.progress_viewer.every_time_close = False
        self.progress_viewer.view_type = 'frames'
        self.progress_viewer.step = 1.0
        self.progress_viewer_thread = threading.Thread(target=self.progress_viewer.run, name='Progress Viewer')

        self.clip_viewer = self.Viewer(self)
        self.clip_viewer.window_name = 'Clip Viewer'
        self.clip_viewer.view_time = self.server.wait_time
        self.clip_viewer.every_time_close = True
        self.clip_viewer.view_type = 'clips'
        self.clip_viewer_thread = threading.Thread(target=self.clip_viewer.run, name='Clip Viewer')


    def run(self):
        self.start_index = 2
        self.end_index = 2
        self.wait_time = 0.5
        self.make_views_index = 1
        self.removing_start_index = 1
        self.removing_end_index = 1
        self.temporal_gap = 2
        self.violence_index = 0
        self.showing_scores = []
        self.view_has_next = False
        self.removing_late_term = int(100 * self.server.video_fps)

        self.progress_viewer_thread.start()
        self.clip_viewer_thread.start()

        while True:
            while self.make_views_index >= self.evaluator.scanned_index:
                time.sleep(self.wait_time)

            number_of_showing_scores = len(self.showing_scores)
            self.end_index = self.start_index + number_of_showing_scores - 1

            with self.server.print_lock:
                print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Secretary', 'Viewing', self.start_index, self.end_index)


            view_frames = []
            for index in range(self.start_index, self.end_index + 1, 1):
                frame = dict()

                score = self.showing_scores[index - self.start_index]
                image = os.path.join(self.server.image_folder, 'show_{:07d}.jpg'.format(index))
                flow_x = os.path.join(self.server.flow_folder, 'flow_x_{:07d}.jpg'.format(index))
                flow_y = os.path.join(self.server.flow_folder, 'flow_y_{:07d}.jpg'.format(index))

                frame['index'] = index
                frame['score'] = score
                frame['image'] = image
                frame['flows'] = [flow_x, flow_y]

                view_frames.append(frame)
                del frame

            self.progress_viewer.view_time = self.evaluator.scan_time
            if len(self.progress_viewer.view_frames) > 0:
                self.progress_viewer.view_has_next = True

            self.progress_viewer.view_frames += view_frames
            del view_frames

            del self.showing_scores[0:number_of_showing_scores]
            gc.collect()
            self.make_views_index = self.end_index
            self.start_index = self.end_index + 1

            self.removing_end_index = self.analyzer.real_base + self.temporal_gap - self.removing_late_term - 1
            removing_amount = self.removing_end_index - self.removing_start_index + 1
            if removing_amount >= 1 and self.removing_end_index <= self.progress_viewer.viewed_index:
                self.remove(self.removing_start_index, self.removing_end_index)
                self.removing_start_index = self.removing_end_index + 1


    def remove(self, start_index, end_index):
        with self.server.print_lock:
            print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Secretary', 'Removing', start_index, end_index)

        remove_path_prefixes = ['img', 'show', 'flow_x', 'flow_y']

        for index in range(start_index, end_index + 1, 1):
            for path_prefix in remove_path_prefixes:
                if path_prefix in ['img', 'show']:
                    remove_path = os.path.join(self.server.image_folder, '{}_{:07d}.jpg'.format(path_prefix, index))
                else:
                    remove_path = os.path.join(self.server.flow_folder, '{}_{:07d}.jpg'.format(path_prefix, index))
                try:
                    os.remove(remove_path)
                except:
                    pass


    class Viewer():
        def __init__(self, secretary):
            self.secretary = secretary

            self.viewed_index = -1
            self.wait_time = 0.5
            self.step = 1.0
            self.time_factor = 1.0
            self.view_time = 0.0
            self.view_frames = []
            self.view_clips = []
            self.violence_index = 0

            self.window_name = ''
            self.window_position = (0, 0)
            self.every_time_close = False
            self.view_has_next = False

            self.view_type = 'frames'


        def run(self):
            if self.view_type == 'frames':
                flow_bound = 20.0
                angle_bound = 180.0
                magnitude_bound = flow_bound * flow_bound * 2.0

                while True:
                    while len(self.view_frames) <= 0:
                        time.sleep(self.wait_time)

                    cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow(self.window_name, self.window_position[0], self.window_position[1])

                    view_frames = []
                    view_frames += self.view_frames
                    view_time = max(min(int(self.view_time * 1000.0 * self.step / self.time_factor),
                                        int(1000.0 / self.secretary.server.video_fps * 3.0)),
                                    1)

                    for frame_index in range(0, len(view_frames), int(self.step)):
                        index = view_frames[frame_index]['index']
                        score = view_frames[frame_index]['score']
                        image = cv2.imread(view_frames[frame_index]['image'])
                        flow_x = cv2.resize(cv2.imread(view_frames[frame_index]['flows'][0],cv2.IMREAD_GRAYSCALE),
                                            self.secretary.server.show_size, interpolation=cv2.INTER_AREA)
                        flow_y = cv2.resize(cv2.imread(view_frames[frame_index]['flows'][1],cv2.IMREAD_GRAYSCALE),
                                            self.secretary.server.show_size, interpolation=cv2.INTER_AREA)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        top_left = (0, 0)
                        text_margin = (15, 10)

                        scale = 0.60
                        thickness = 2
                        line_type = cv2.LINE_8

                        text_color = (255, 255, 255)
                        head_background_color = (0, 0, 0)

                        if np.argmax(score) == self.violence_index:
                            frame_label = 'Violence'
                            label_background_color = (0, 0, 255)
                        else:
                            frame_label = 'Normal'
                            label_background_color = (255, 0, 0)

                        label_text = 'Frame {:07d} | Prediction {:^8s} | Score {:05.02f}'.format(index, frame_label,
                                                                                               max(score))
                        flow_head_text = '{}'.format('Optical Flow | Between {:07d} And {:07d}'.format(index-1, index))

                        box_size, dummy = cv2.getTextSize(label_text, font, scale, thickness)

                        box_top_left = top_left
                        box_bottom_right = (image.shape[1],
                                            top_left[1] + box_size[1] + text_margin[1] * 2)
                        box_width = box_bottom_right[0]
                        box_height = box_bottom_right[1]

                        image_label_text_bottom_left = (top_left[0] + text_margin[0],
                                                        top_left[1] + text_margin[1] + box_size[1])
                        flow_head_text_bottom_left = (top_left[0] + text_margin[0],
                                                      top_left[1] + text_margin[1] + box_size[1])

                        image_box_top_left = box_top_left
                        image_box_bottom_right = box_bottom_right

                        image_headline = np.zeros((box_height, box_width, 3), dtype='uint8')
                        cv2.rectangle(image_headline, image_box_top_left, image_box_bottom_right,
                                      label_background_color, cv2.FILLED)

                        cv2.putText(image_headline, label_text, image_label_text_bottom_left,
                                    font, scale, text_color, thickness, line_type)

                        flow_box_top_left = box_top_left
                        flow_box_bottom_right = box_bottom_right

                        flow_headline = np.zeros((box_height, box_width, 3), dtype='uint8')
                        cv2.rectangle(flow_headline, flow_box_top_left, flow_box_bottom_right,
                                      head_background_color, cv2.FILLED)

                        cv2.putText(flow_headline, flow_head_text, flow_head_text_bottom_left,
                                    font, scale, text_color, thickness, line_type)

                        flow = np.zeros_like(image)

                        flow_x = np.divide(flow_x, 255.0)
                        flow_x = np.multiply(flow_x, float(flow_bound * 2))
                        flow_x -= float(flow_bound)
                        flow_x = np.clip(flow_x, -flow_bound, flow_bound)

                        flow_y = np.divide(flow_y, 255.0)
                        flow_y = np.multiply(flow_y, float(flow_bound * 2))
                        flow_y -= float(flow_bound)
                        flow_y = np.clip(flow_y, -flow_bound, flow_bound)

                        magnitude = flow_x * flow_x + flow_y * flow_y
                        magnitude += 40.0
                        magnitude = np.clip(magnitude, 0.0, magnitude_bound)
                        magnitude = np.divide(magnitude, magnitude_bound)
                        magnitude = np.multiply(magnitude, 255.0 * 1.5)

                        angle = np.divide(flow_y, flow_x)
                        angle = np.arctan(angle)
                        angle = np.multiply(angle, 180.0)
                        angle = np.divide(angle, np.pi)

                        angle = np.clip(angle, -angle_bound, angle_bound)
                        angle += float(angle_bound)
                        angle = np.divide(angle, angle_bound)
                        angle = np.multiply(angle, 255.0)

                        del flow_x
                        del flow_y

                        hue = np.array(angle, dtype='uint8')
                        saturation = np.array(magnitude, dtype='uint8')
                        value = np.array(np.multiply(np.ones_like(hue), 255.0), dtype='uint8')

                        del angle
                        del magnitude

                        flow[..., 0] = hue
                        flow[..., 1] = saturation
                        flow[..., 2] = value

                        flow = cv2.cvtColor(flow, cv2.COLOR_HSV2BGR)
                        del hue
                        del saturation
                        del value

                        image_frame = np.concatenate((image_headline, image), axis=0)
                        flow_frame = np.concatenate((flow_headline, flow), axis=0)

                        del image
                        del flow
                        del image_headline
                        del flow_headline

                        boundary_top_left = (0, 0)
                        image_boundary_bottom_right = (image_frame.shape[1], image_frame.shape[0])
                        flow_boundary_bottom_right = (flow_frame.shape[1], flow_frame.shape[0])
                        boundary_color = (255, 255, 255)

                        cv2.rectangle(image_frame, boundary_top_left, image_boundary_bottom_right,
                                      boundary_color, thickness=3)
                        cv2.rectangle(flow_frame, boundary_top_left, flow_boundary_bottom_right,
                                      boundary_color, thickness=3)

                        frame = np.concatenate((image_frame, flow_frame), axis=0)
                        del image_frame
                        del flow_frame

                        cv2.imshow(self.window_name, frame)
                        cv2.moveWindow(self.window_name, self.window_position[0], self.window_position[1])
                        cv2.waitKey(view_time)
                        del frame

                        if self.view_has_next:
                            view_time = max(int(view_time / 5.0), 1)
                            self.view_has_next = False

                    if len(view_frames) >= 1:
                        self.viewed_index = view_frames[-1]['index']
                    del self.view_frames[0:len(view_frames)]
                    del view_frames
                    gc.collect()
            elif self.view_type == 'clips':
                while True:
                    while len(self.view_clips) <= 0:
                        time.sleep(self.wait_time)

                    view_clips = []
                    view_clips += self.view_clips

                    for clip in view_clips:
                        video_cap = cv2.VideoCapture(clip)
                        if video_cap.isOpened():
                            frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                            video_fps = int(video_cap.get(cv2.CAP_PROP_FPS)) + 1
                            view_time = 1.0 / video_fps
                            video_cap.release()
                        else:
                            continue

                        cmd = 'xdg-open {}'.format(quote(clip))

                        with self.secretary.extractor.cmd_lock:
                            os.system(cmd)
                            sys.stdout.flush()

                        for ii in xrange(frame_count):
                            time.sleep(view_time)
                            if self.view_has_next:
                                view_time = int(view_time / 2.0)
                                self.view_has_next = False

                    del self.view_clips[0:len(view_clips)]
                    del view_clips
                    gc.collect()


class Closer():
    def __init__(self, server, extractor, evaluator, analyzer, secretary):
        self.server = server
        self.extractor = extractor
        self.evaluator = evaluator
        self.analyzer = analyzer
        self.secretary = secretary


    def run(self):
        self.clips = []
        self.wait_time = 0.5
        self.threshold = 0.0
        self.clip_number = 1
        self.clip_round = 5
        self.violence_index = 0
        self.normal_index = 1


        while True:
            while len(self.clips) <= 0:
                time.sleep(self.wait_time)

            clips = []
            clips += self.clips

            for clip in clips:
                isPassed, filtered_scores = self.check(clip)

                if isPassed:
                    for frame in clip['frames']:
                        frame['score'] = filtered_scores[clip['frames'].index(frame)]

                    self.visualize(clip)
                else:
                    rmtree(clip['keep_folder'], ignore_errors=True)

            del self.clips[0:len(clips)]
            gc.collect()


    def check(self, clip):
        with self.server.print_lock:
            print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Closer', 'Checking',
                                                                clip['time_intervals'][0], clip['time_intervals'][1])

        scores = []
        for frame in clip['frames']:
            scores.append(frame['score'])

        scores = np.asarray(scores)
        for index in range(self.analyzer.median_kernal_size/2, len(scores), 1):
            if index + self.analyzer.median_kernal_size/2 > len(scores):
                break
            scores[index][self.violence_index] \
                = np.median(scores[index:index + self.analyzer.median_kernal_size / 2 + 1, self.violence_index])
            scores[index][self.normal_index] \
                = np.median(scores[index:index + self.analyzer.median_kernal_size / 2 + 1, self.normal_index])

        avg_score = np.divide(np.sum(scores, axis=0), max(1, len(scores))).tolist()

        if len(avg_score) < 2:
            return False, None

        return avg_score[self.violence_index] >= self.threshold , scores


    def visualize(self, clip):
        with self.server.print_lock:
            print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Closer', 'Clipping',
                                                                clip['time_intervals'][0], clip['time_intervals'][1])

        admin_clip_path = os.path.join(self.server.clip_folder,
                                       'Admin_{}.avi'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        user_clip_path = os.path.join(self.server.clip_folder,
                                      'User_{}.avi'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

        admin_writer_initialized = False
        user_writer_initialized = False
        admin_video_writer = None
        user_video_writer = None
        for round in range(self.clip_round, 0, -1):
            for frame in clip['frames']:
                index = frame['index']
                score = frame['score']
                image = cv2.imread(frame['image'])
                flow_x = cv2.resize(cv2.imread(frame['flows'][0], cv2.IMREAD_GRAYSCALE),
                                    self.server.show_size, cv2.INTER_AREA)
                flow_y = cv2.resize(cv2.imread(frame['flows'][1], cv2.IMREAD_GRAYSCALE),
                                    self.server.show_size, cv2.INTER_AREA)

                if not user_writer_initialized:
                    video_fps = self.server.video_fps
                    video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    video_size = (int(image.shape[1]), int(image.shape[0]))
                    user_video_writer = cv2.VideoWriter(user_clip_path, video_fourcc, video_fps, video_size)
                    user_writer_initialized = True

                for iter in xrange(round):
                    user_video_writer.write(image)

                flow_bound = 20.0
                angle_bound = 180.0
                magnitude_bound = flow_bound * flow_bound * 2.0

                font = cv2.FONT_HERSHEY_SIMPLEX
                top_left = (0, 0)
                text_margin = (15, 10)

                scale = 0.60
                thickness = 2
                line_type = cv2.LINE_8

                text_color = (255, 255, 255)
                head_background_color = (0, 0, 0)

                if np.argmax(score) == self.violence_index:
                    frame_label = 'Violence'
                    label_background_color = (0, 0, 255)
                else:
                    frame_label = 'Normal'
                    label_background_color = (255, 0, 0)

                label_text = 'Frame {:07d} | Prediction {:^8s} | Score {:05.02f}'.format(index, frame_label, max(score))

                flow_head_text = '{}'.format('Optical Flow | Between {:07d} And {:07d}'.format(index - 1, index))

                box_size, dummy = cv2.getTextSize(label_text, font, scale, thickness)

                box_top_left = top_left
                box_bottom_right = (image.shape[1],
                                    top_left[1] + box_size[1] + text_margin[1] * 2)
                box_width = box_bottom_right[0]
                box_height = box_bottom_right[1]

                image_label_text_bottom_left = (top_left[0] + text_margin[0],
                                               top_left[1] + text_margin[1] + box_size[1])
                flow_head_text_bottom_left = (top_left[0] + text_margin[0],
                                              top_left[1] + text_margin[1] + box_size[1])

                image_box_top_left = box_top_left
                image_box_bottom_right = box_bottom_right

                image_headline = np.zeros((box_height, box_width, 3), dtype='uint8')
                cv2.rectangle(image_headline, image_box_top_left, image_box_bottom_right,
                              label_background_color, cv2.FILLED)

                cv2.putText(image_headline, label_text, image_label_text_bottom_left,
                            font, scale, text_color, thickness, line_type)

                flow_box_top_left = box_top_left
                flow_box_bottom_right = box_bottom_right

                flow_headline = np.zeros((box_height, box_width, 3), dtype='uint8')
                cv2.rectangle(flow_headline, flow_box_top_left, flow_box_bottom_right,
                              head_background_color, cv2.FILLED)

                cv2.putText(flow_headline, flow_head_text, flow_head_text_bottom_left,
                            font, scale, text_color, thickness, line_type)

                flow = np.zeros_like(image)

                flow_x = np.divide(flow_x, 255.0)
                flow_x = np.multiply(flow_x, float(flow_bound * 2))
                flow_x -= float(flow_bound)
                flow_x = np.clip(flow_x, -flow_bound, flow_bound)

                flow_y = np.divide(flow_y, 255.0)
                flow_y = np.multiply(flow_y, float(flow_bound * 2))
                flow_y -= float(flow_bound)
                flow_y = np.clip(flow_y, -flow_bound, flow_bound)

                magnitude = flow_x * flow_x + flow_y * flow_y
                magnitude += 40.0
                magnitude = np.clip(magnitude, 0.0, magnitude_bound)
                magnitude = np.divide(magnitude, magnitude_bound)
                magnitude = np.multiply(magnitude, 255.0 * 1.5)

                angle = np.divide(flow_y, flow_x)
                angle = np.arctan(angle)
                angle = np.multiply(angle, 180.0)
                angle = np.divide(angle, np.pi)
                angle = np.clip(angle, -angle_bound, angle_bound)

                del flow_x
                del flow_y

                hue = np.array(angle, dtype='uint8')
                saturation = np.array(magnitude, dtype='uint8')
                value = np.array(np.multiply(np.ones_like(hue), 255.0), dtype='uint8')

                del angle
                del magnitude

                flow[..., 0] = hue
                flow[..., 1] = saturation
                flow[..., 2] = value

                flow = cv2.cvtColor(flow, cv2.COLOR_HSV2BGR)
                del hue
                del saturation
                del value

                image_frame = np.concatenate((image_headline, image), axis=0)
                flow_frame = np.concatenate((flow_headline, flow), axis=0)

                del image
                del flow
                del image_headline
                del flow_headline

                boundary_top_left = (0, 0)
                image_boundary_bottom_right = (image_frame.shape[1], image_frame.shape[0])
                flow_boundary_bottom_right = (flow_frame.shape[1], flow_frame.shape[0])
                boundary_color = (255, 255, 255)

                cv2.rectangle(image_frame, boundary_top_left, image_boundary_bottom_right,
                              boundary_color, thickness=7)
                cv2.rectangle(flow_frame, boundary_top_left, flow_boundary_bottom_right,
                              boundary_color, thickness=7)

                frame = np.concatenate((image_frame, flow_frame), axis=1)
                del image_frame
                del flow_frame

                if not admin_writer_initialized:
                    video_fps = self.server.video_fps
                    video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    video_size = (int(frame.shape[1]), int(frame.shape[0]))
                    admin_video_writer = cv2.VideoWriter(admin_clip_path, video_fourcc, video_fps, video_size)
                    admin_writer_initialized = True

                for iter in xrange(round):
                    admin_video_writer.write(frame)

                del frame

        if admin_video_writer is not None:
            admin_video_writer.release()

        if user_video_writer is not None:
            user_video_writer.release()


        gc.collect()

        if len(self.secretary.clip_viewer.view_clips) > 0:
            self.secretary.clip_viewer.view_has_next = True
        self.secretary.clip_viewer.view_clips.append(admin_clip_path)

        sending_clips = [ admin_clip_path, user_clip_path ]

        rmtree(clip['keep_folder'])



if __name__ == '__main__':
    server = ServerFromVideo()

    while True:
        time.sleep(3.1451)

        with server.print_lock:
            print '------------------------------ Memory Checking ---------------------------------'
            cmd = 'free -h'
            with server.extractor.cmd_lock:
                os.system(cmd)
                sys.stdout.flush()
            print '--------------------------------------------------------------------------------'


        