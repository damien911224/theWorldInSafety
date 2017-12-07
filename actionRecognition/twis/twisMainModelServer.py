import sys
sys.path.insert(0, "../lib/caffe-action/python")
sys.path.append('..')
import caffe
import cv2
import os
import numpy as np
import time
import glob
import threading
from pipes import quote
from utils.io import flow_stack_oversample, fast_list2arr
from multiprocessing import Pool, Value, Lock, current_process, Manager, Process
import copy_reg, types
import gc
import socket
from shutil import copyfile
from shutil import rmtree
import random
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

        os_frame = fast_list2arr(frame)

        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()


    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):
        if frame is None:
            return [0.0, 0.0]

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size, interpolation=cv2.INTER_AREA) for x in frame])
        else:
            frame = fast_list2arr(frame)

        os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()


class Evaluator():

    def __init__(self):
        self.in_progress =True

        self.num_workers = 16
        self.num_using_gpu = 12

        self.temporal_gap = 2

        self.scores = []

        self.use_spatial_net = False
        self.model_version = 4
        self.build_net(self.model_version)

        self.main_server_ip_address = self.getMyIpAddress()
        self.main_server_port_number = 7777

        self.element_boundary = b'!element_boundary!'
        self.one_boundary = b'!one_boundary!'
        self.entire_boundary = b'!entire_boundary!'

        self.save_folder = '../progress/temp'

        global scanning_pool_odd
        global scanning_pool_even
        scanning_pool_odd = Pool(self.num_workers)
        scanning_pool_even =Pool(self.num_workers)

        copy_reg.pickle(types.MethodType, self._pickle_method)

        self.scanner = Scanner(self.save_folder, self.save_folder,
                               self.num_workers, self.num_using_gpu, self.use_spatial_net)

        self.evaluator_thread = threading.Thread(target=self.run, name='Evaluator')
        self.evaluator_thread.start()


    def run(self):
        while True:
            while not self.in_progress:
                time.sleep(0.5)

            self.main_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.main_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.main_server_socket.bind((self.main_server_ip_address, self.main_server_port_number))

            self.main_server_socket.listen(5)

            while self.in_progress:
                client_socket, address = self.main_server_socket.accept()

                previous_data = b''
                while self.in_progress:
                    accumulated_data = previous_data + b''
                    while True:
                        recv_data = client_socket.recv(90456)
                        if len(recv_data) == 0:
                            break
                        accumulated_data += recv_data
                        found = accumulated_data.find(self.entire_boundary)
                        if found != -1:
                            previous_data = accumulated_data[found+len(self.entire_boundary):]
                            accumulated_data = accumulated_data[:found]
                            break

                    entire_frame_data = accumulated_data

                    frames_data = []
                    while True:
                        found = entire_frame_data.find(self.one_boundary)
                        if found != -1:
                            frame_data = entire_frame_data[:found]
                            frames_data.append(frame_data)
                            entire_frame_data = entire_frame_data[found + len(self.one_boundary):]
                        else:
                            break

                    rmtree(self.save_folder, ignore_errors=True)

                    try:
                        os.makedirs(self.save_folder)
                    except:
                        pass

                    frame_index = 1
                    for frame_data in frames_data:
                        found = frame_data.find(self.element_boundary)
                        image_data = frame_data[:found]
                        frame_data = frame_data[found + len(self.element_boundary):]
                        found = frame_data.find(self.element_boundary)
                        flow_x_data = frame_data[:found]
                        frame_data = frame_data[found + len(self.element_boundary):]
                        flow_y_data = frame_data

                        image_np_arr = np.fromstring(image_data , np.uint8)
                        image = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)
                        flow_x_arr = np.fromstring(flow_x_data, np.uint8)
                        flow_x = cv2.imdecode(flow_x_arr, cv2.IMREAD_GRAYSCALE)
                        flow_y_arr = np.fromstring(flow_y_data, np.uint8)
                        flow_y = cv2.imdecode(flow_y_arr, cv2.IMREAD_GRAYSCALE)

                        cv2.imwrite(os.path.join(self.save_folder, 'img_{:07d}.jpg'.format(frame_index)), image)
                        cv2.imwrite(os.path.join(self.save_folder, 'flow_x_{:07d}.jpg'.format(frame_index)), flow_x)
                        cv2.imwrite(os.path.join(self.save_folder, 'flow_y_{:07d}.jpg'.format(frame_index)), flow_y)

                        frame_index += 1


                    end_index = len(frames_data) - self.temporal_gap
                    start_index = 1 + self.temporal_gap


                    print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Evaluator', 'Evaluating', start_index, end_index)


                    manager = Manager()
                    return_scores = manager.list()
                    len_of_scores = end_index - start_index + 1
                    for _ in range(len_of_scores):
                        return_scores.append([0.0, 0.0])

                    self.odd_scanner_thread = threading.Thread(target=self.scanner.scan,
                                                               args=(start_index, end_index,
                                                                     len(frames_data), 'odd', return_scores))
                    self.even_scanner_thread = threading.Thread(target=self.scanner.scan,
                                                                args=(start_index, end_index,
                                                                      len(frames_data), 'even', return_scores))

                    self.odd_scanner_thread.start()
                    self.even_scanner_thread.start()

                    self.odd_scanner_thread.join()
                    self.even_scanner_thread.join()


                    for score in return_scores:
                        violence_score_string = b'{:07.03f}'.format(score[0])
                        normal_score_string = b'{:07.03f}'.format(score[1])
                        score_string = violence_score_string + normal_score_string

                        client_socket.send(score_string)
                    client_socket.send(self.entire_boundary)

            self.finalize()


    def getMyIpAddress(self):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))

            my_ip_address = sock.getsockname()[0]
            sock.close()

            print '{:10s}|{:12s}|{}'.format('Evaluator', 'Connection', 'With IP {}'.format(my_ip_address))

            return my_ip_address


    def build_net(self, version=4, use_spatial_net=False):
        global spatial_net_gpu_01
        global spatial_net_gpu_02
        global temporal_net_gpu_01
        global temporal_net_gpu_02

        global spatial_net_cpu
        global temporal_net_cpu

        self.spatial_net_proto = "../models/twis/tsn_bn_inception_rgb_deploy.prototxt"
        self.spatial_net_weights = "../models/twis_caffemodels/v{0}/twis_spatial_net_v{0}.caffemodel".format(
            version)
        self.temporal_net_proto = "../models/twis/tsn_bn_inception_flow_deploy.prototxt"
        self.temporal_net_weights = "../models/twis_caffemodels/v{0}/twis_temporal_net_v{0}.caffemodel".format(
            version)

        spatial_net_gpu_01 = CaffeNet(self.spatial_net_proto, self.spatial_net_weights, 0)
        spatial_net_gpu_02 = CaffeNet(self.spatial_net_proto, self.spatial_net_weights, 1)
        temporal_net_gpu_01 = CaffeNet(self.temporal_net_proto, self.temporal_net_weights, 0)
        temporal_net_gpu_02 = CaffeNet(self.temporal_net_proto, self.temporal_net_weights, 1)

        spatial_net_cpu = CaffeNet(self.spatial_net_proto, self.spatial_net_weights, -1)
        temporal_net_cpu = CaffeNet(self.temporal_net_proto, self.temporal_net_weights, -1)


    def _pickle_method(self,m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)


    def finalize(self):
        global scanning_pool_odd
        global scanning_pool_even

        scanning_pool_odd.close()
        scanning_pool_odd.join()
        del scanning_pool_odd

        scanning_pool_even.close()
        scanning_pool_even.join()
        del scanning_pool_even

        del self.scanner

        gc.collect()

        self.main_server_socket.close()

        self.resume()


    def resume(self):
        self.start_index = 2
        self.scanned_index = 1
        self.actual_start_index = 2

        self.scores = []

        global scanning_pool_odd
        global scanning_pool_even
        scanning_pool_odd = Pool()
        scanning_pool_even = Pool()

        self.scanner = Scanner(self.save_folder, self.save_folder, self.num_workers,
                               self.num_using_gpu, self.use_spatial_net)

        self.in_progress = True


class Scanner():

    def __init__(self, image_folder, flow_folder, num_workers, num_using_gpu, use_spatial_net):
        self.image_folder = image_folder
        self.flow_folder = flow_folder
        self.num_workers = num_workers
        self.num_using_gpu = num_using_gpu
        self.use_spatial_net = use_spatial_net

        self.rate_of_whole = 5.0
        self.rate_of_time = 3.5
        self.rate_of_space = self.rate_of_whole - self.rate_of_time

        self.score_bound = 30.0


    def scan(self, start_index, end_index, actual_extracted_index, scanner_type, return_scores):
        global scanning_pool_odd
        global scanning_pool_even

        if scanner_type == 'odd':
            indices = range(start_index, end_index + 1, 2)
            scanning_pool = scanning_pool_odd
            device_id = 0
        else:
            indices = range(start_index + 1, end_index + 1, 2)
            scanning_pool = scanning_pool_even
            device_id = 1

        scanning_pool.map(self.scanVideo,
                          zip([actual_extracted_index] * len(indices),
                              indices, [device_id] * len(indices),
                              [start_index] * len(indices), [return_scores] * len(indices)))


    def scanVideo(self, scan_items):
        global spatial_net_gpu_01
        global spatial_net_gpu_02
        global temporal_net_gpu_01
        global temporal_net_gpu_02
        global spatial_net_cpu
        global temporal_net_cpu

        frame_count = scan_items[0]
        index = scan_items[1]
        device_id = scan_items[2]
        start_index = scan_items[3]
        return_scores = scan_items[4]

        current = current_process()
        current_id = current._identity[0] -1

        if current_id % self.num_workers < self.num_using_gpu:
            if device_id == 0:
                spatial_net = spatial_net_gpu_01
                temporal_net = temporal_net_gpu_01
            else:
                spatial_net = spatial_net_gpu_02
                temporal_net = temporal_net_gpu_02
        else:
            spatial_net = spatial_net_cpu
            temporal_net = temporal_net_cpu

        score_layer_name = 'fc-twis'

        if self.use_spatial_net:
            image_frame = cv2.imread(os.path.join(self.image_folder, 'img_{:07d}.jpg'.format(index)))

            rgb_score = \
                spatial_net.predict_single_frame([image_frame, ], score_layer_name, over_sample=False,
                                                 frame_size=None)[0].tolist()

        flow_stack = []
        for i in range(-2, 3, 1):
            if index + i >= 2 and index + i <= frame_count:
                x_flow_field = cv2.imread(
                    os.path.join(self.flow_folder, 'flow_x_{:07d}.jpg').format(index + i),
                    cv2.IMREAD_GRAYSCALE)
                y_flow_field = cv2.imread(
                    os.path.join(self.flow_folder, 'flow_y_{:07d}.jpg').format(index + i),
                    cv2.IMREAD_GRAYSCALE)
                flow_stack.append(x_flow_field)
                flow_stack.append(y_flow_field)
            elif index + i < 2:
                x_flow_field = cv2.imread(
                    os.path.join(self.flow_folder, 'flow_x_{:07d}.jpg').format(2),
                    cv2.IMREAD_GRAYSCALE)
                y_flow_field = cv2.imread(
                    os.path.join(self.flow_folder, 'flow_y_{:07d}.jpg').format(2),
                    cv2.IMREAD_GRAYSCALE)
                flow_stack.append(x_flow_field)
                flow_stack.append(y_flow_field)
            else:
                x_flow_field = cv2.imread(
                    os.path.join(self.flow_folder, 'flow_x_{:07d}.jpg').format(frame_count),
                    cv2.IMREAD_GRAYSCALE)
                y_flow_field = cv2.imread(
                    os.path.join(self.flow_folder, 'flow_y_{:07d}.jpg').format(frame_count),
                    cv2.IMREAD_GRAYSCALE)
                flow_stack.append(x_flow_field)
                flow_stack.append(y_flow_field)

        flow_score = \
            temporal_net.predict_single_flow_stack(flow_stack, score_layer_name, over_sample=False,
                                                   frame_size=None)[0].tolist()


        if self.use_spatial_net:
            entire_scores = np.divide(np.clip(np.asarray([rgb_score[i] * self.rate_of_space
                                                                     + flow_score[i] * self.rate_of_time for i in xrange(len(flow_score))]),
                                                                 -self.score_bound, self.score_bound),
                                                         self.score_bound).tolist()
        else:
            entire_scores = np.divide(np.clip(np.asarray([flow_score[i] * self.rate_of_whole for i in xrange(len(flow_score))]),
                                                                 -self.score_bound, self.score_bound),
                                                         self.score_bound).tolist()

        return_scores[index - start_index] = entire_scores



if __name__ == '__main__':
    evaluator = Evaluator()

    while True:
        time.sleep(13.1451)