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
from caffe.io import oversample
from utils.io import flow_stack_oversample, fast_list2arr
from multiprocessing import Pool, Value, Lock, current_process, Manager
import copy_reg, types
import gc
import socket
from shutil import copyfile
from shutil import rmtree
import random
import datetime
from eventlet import GreenPool



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


class Session():

    def __init__(self):
        self.session_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.seesion_closed = False

        self.in_progress = True
        self.please_quit = False
        self.extractor_closed = False

        self.root_folder = os.path.abspath('../progress')
        self.session_folder = os.path.join(self.root_folder, '{}'.format(self.session_name))
        self.image_folder = os.path.join(self.session_folder, 'images')
        self.flow_folder = os.path.join(self.session_folder, 'flows')
        self.clip_folder = os.path.join(self.session_folder, 'clips')
        self.clip_view_folder = os.path.join(self.clip_folder, 'view_clips')
        self.clip_send_folder = os.path.join(self.clip_folder, 'send_clips')
        self.keep_folder = os.path.join(self.session_folder, 'keep')

        folders = [ self.root_folder, self.session_folder, self.image_folder,
                    self.flow_folder, self.clip_folder, self.clip_view_folder,
                    self.clip_send_folder, self.keep_folder ]

        previous_session_folders = glob.glob(os.path.join(self.root_folder, '20*'))
        for folder in previous_session_folders:
            rmtree(folder, ignore_errors=True)

        for folder in folders:
            try:
                os.mkdir(folder)
            except OSError:
                pass


        self.show_size = (600, 450)
        self.new_size = (224, 224)
        self.temporal_width = 1
        self.print_term = 50
        self.fps = 30.0
        self.wait_time = 1.0 / self.fps
        self.wait_please = False
        self.is_rotated = False
        self.rotating_angle = -90

        self.start_index = 1
        self.dumped_index = 0


        self.src_from_out = True
        self.web_cam = False
        if self.web_cam:
            self.test_video_name = 'Webcam.mp4'
        else:
            self.test_video_name = 'test_1.mp4'
        self.model_version = 4
        self.use_spatial_net = True
        self.build_net(self.model_version, self.use_spatial_net)

        self.print_lock = Lock()
        self.average_delay = 0.0

        self.server_ip_address = '13.124.183.55'
        self.server_port_number = 8888

        self.client_host_name = self.getMyIpAddress()
        self.client_port_number = random.sample(range(10000, 20000, 1), 1)[0]

        self.extractor = Extractor(self)
        self.extractor_thread = threading.Thread(target=self.extractor.run, name='Extractor')

        self.session_thread = threading.Thread(target=self.run, name='Session')
        self.session_thread.start()


    def run(self):
        self.child_thread_started = False

        while not self.please_quit:
            while not self.in_progress:
                time.sleep(0.5)

            if self.src_from_out:
                self.video_width = self.show_size[0]
                self.video_height = self.show_size[1]
                self.video_fps = self.fps

                if not self.child_thread_started:
                    self.extractor_thread.start()
                    self.child_thread_started = True


                while self.in_progress:
                    try:
                        self.client_port_number = random.sample(range(10000, 20000, 1), 1)[0]
                        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        self.client_socket.bind((self.client_host_name, self.client_port_number))
                        self.client_socket.connect((self.server_ip_address, self.server_port_number))

                        self.session_is_opened = False

                        previous_data = b''
                        previous_index = 0
                        socket_closed = False
                        while self.in_progress:
                            frame_data = previous_data + b''
                            try:
                                while self.in_progress:
                                    r = self.client_socket.recv(90456)
                                    if len(r) == 0:
                                        socket_closed = True
                                        break

                                    a = r.find(b'!TWIS_END!')
                                    if a != -1:
                                        frame_data += r[:a]
                                        previous_data = r[a+10:]
                                        break
                                    else:
                                        frame_data += r
                            except:
                                continue

                            if socket_closed:
                                break

                            if self.in_progress:
                                header = frame_data[:36]
                                session_name = str(header[:15])
                                frame_index = int(header[15:22])
                                frame_moment = int(header[22:36])
                                frame_data = frame_data[36:]

                                if frame_index - previous_index >= 2:
                                    for i in range(previous_index+1, frame_index, 1):
                                        print 'ERROR {:07d}'.format(i)

                                previous_index = frame_index

                                if not self.session_is_opened:
                                    self.session_name = session_name
                                    self.session_delay = 0.0
                                    self.delay_count = 0
                                    folders = [self.session_folder, self.image_folder,
                                               self.flow_folder, self.clip_folder, self.clip_view_folder,
                                               self.clip_send_folder, self.keep_folder]

                                    previous_session_folders = glob.glob(os.path.join(self.root_folder, '20*'))
                                    for folder in previous_session_folders:
                                        rmtree(folder, ignore_errors=True)

                                    for folder in folders:
                                        try:
                                            os.mkdir(folder)
                                        except OSError:
                                            pass

                                    with self.print_lock:
                                        print '==============================================================================='
                                        print '                         Session {} Start                                      '.format(
                                            self.session_name)
                                        print '==============================================================================='

                                    self.session_is_opened = True

                                np_arr = np.fromstring(frame_data, np.uint8)
                                if np_arr is not None:
                                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                                    if frame is not None:
                                        self.dumpFrames([frame])
                                        self.start_index += 1
                                        self.dumped_index = max(self.start_index - 1, 1)

                                self.session_delay += float(int(datetime.datetime.now().strftime('%M%S%s')) - frame_moment)
                                self.delay_count += 1
                                self.average_delay = self.session_delay / self.delay_count / 10000000000.0
                            else:
                                break

                    except socket.timeout:
                        self.client_socket.close()
                        continue

                    except KeyboardInterrupt:
                        if self.client_socket is not None:
                            self.client_socket.close()

                        self.finalize()

                    self.client_socket.close()

                    self.finalize()

                    while not self.session_closed:
                        time.sleep(0.3)

                    self.resume()


                self.finalize()
            else:
                if self.web_cam:
                    video_cap = cv2.VideoCapture(0)
                else:
                    video_cap = cv2.VideoCapture(os.path.join(self.root_folder, 'test_videos', self.test_video_name))
                self.video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_fps = video_cap.get(cv2.CAP_PROP_FPS)

                if not self.child_thread_started:
                    self.extractor_thread.start()
                    self.child_thread_started = True

                while self.in_progress:
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

                video_cap.release()
                self.finalize()


    def build_net(self, version=4, use_spatial_net=False):
        global spatial_net_gpu_01
        global spatial_net_gpu_02
        global temporal_net_gpu_01
        global temporal_net_gpu_02

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


    def dumpFrames(self, frames):
        end_index = self.start_index + len(frames) - 1
        if end_index % self.print_term == 0:
            with self.print_lock:
                print '{:10s}|{:12s}| Until {:07d}|Delay {:.6f} Seconds'.format('Session', 'Dumping', end_index, self.average_delay)

        index = self.start_index
        for frame in frames:
            file_name = os.path.join(self.image_folder, 'img_{:07d}.jpg'.format(index))
            new_frame = cv2.resize(frame, self.new_size, interpolation=cv2.INTER_AREA)
            if self.is_rotated:
                new_frame = imutils.rotate(new_frame, self.rotating_angle)
            cv2.imwrite(file_name, new_frame)
            index += 1


    def getMyIpAddress(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))

        my_ip_address = sock.getsockname()[0]
        sock.close()

        with self.print_lock:
            print '{:10s}|{:12s}|{}'.format('Session', 'Connection', 'With IP {}'.format(my_ip_address))

        return my_ip_address


    def finalize(self):
        global session_closed

        self.extractor.in_progress = False

        while not self.extractor_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.print_lock:
            print '{:10s}|{} Finalized'.format('Session', 'Extractor')

        gc.collect()

        self.session_closed = True

        with self.print_lock:
            print '==============================================================================='
            print '                         Session {} Closed                                     '.format(
                self.session_name)
            print '==============================================================================='


    def resume(self):
        self.session_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.in_progress = True
        self.please_quit = False
        self.extractor_closed = False

        self.root_folder = os.path.abspath('../progress')
        self.session_folder = os.path.join(self.root_folder, '{}'.format(self.session_name))
        self.image_folder = os.path.join(self.session_folder, 'images')
        self.flow_folder = os.path.join(self.session_folder, 'flows')
        self.clip_folder = os.path.join(self.session_folder, 'clips')
        self.clip_view_folder = os.path.join(self.clip_folder, 'view_clips')
        self.clip_send_folder = os.path.join(self.clip_folder, 'send_clips')
        self.keep_folder = os.path.join(self.session_folder, 'keep')

        folders = [self.root_folder, self.session_folder, self.image_folder,
                   self.flow_folder, self.clip_folder, self.clip_view_folder,
                   self.clip_send_folder, self.keep_folder]

        previous_session_folders = glob.glob(os.path.join(self.root_folder, '20*'))
        for folder in previous_session_folders:
            rmtree(folder, ignore_errors=True)

        for folder in folders:
            try:
                os.mkdir(folder)
            except OSError:
                pass

        self.in_progress = True

        self.start_index = 1
        self.dumped_index = 0

        self.extractor.resume()
        self.extractor_closed = False


class Extractor():

    def __init__(self, session):
        self.in_progress = True
        self.evaluator_closed = False

        self.session = session

        self.evaluator = Evaluator(self.session, self)
        self.evaluator_thread = threading.Thread(target=self.evaluator.run, name='Evaluator')

        self.cmd_lock = Lock()

        self.start_index = 2
        self.extracted_index = 0
        self.wait_time = 0.3


    def run(self):
        self.child_thread_started = False

        while True:
            while not self.in_progress:
                time.sleep(0.5)

            if not self.child_thread_started:
                self.evaluator_thread.start()
                self.child_thread_started = True


            while self.in_progress:
                while self.session.dumped_index <= self.extracted_index and self.in_progress:
                    time.sleep(self.wait_time)

                if self.in_progress:
                    self.end_index = self.session.dumped_index

                    with self.session.print_lock:
                        print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Extractor', 'Extracting', self.start_index, self.end_index)

                    self.extractOpticalFlows(self.session.image_folder, self.start_index, self.end_index, self.session.flow_folder)

                    self.start_index = self.end_index + 1
                    self.extracted_index = self.end_index

            self.finalize()


    def extractOpticalFlows(self, frame_path, start_index, end_index, flow_dst_folder):
        out_format = 'dir'
        root_abs_path = os.path.abspath('..')
        df_path = os.path.join(root_abs_path, 'lib', 'dense_flow')

        image_path = 'None'
        video_file_path = 'Per Frame'
        frame_prefix = '{}/img'.format(frame_path)
        frame_count = end_index - start_index + 1
        optical_flow_x_path = '{}/flow_x'.format(flow_dst_folder)
        optical_flow_y_path = '{}/flow_y'.format(flow_dst_folder)

        cmd = os.path.join(
            df_path + '/build/extract_cpu') + ' {} {} {} {} 20 {} {} {}'.format(
            quote(frame_prefix), quote(optical_flow_x_path), quote(optical_flow_y_path),
            out_format,  frame_count, start_index, end_index)

        with self.cmd_lock:
            os.system(cmd)
            sys.stdout.flush()


    def finalize(self):
        self.evaluator.in_progress = False

        while not self.evaluator_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.session.print_lock:
            print '{:10s}|{} Finalized'.format('Extractor', 'Evaluator')

        gc.collect()

        self.session.extractor_closed = True


    def resume(self):
        self.start_index = 2
        self.extracted_index = 0

        self.in_progress = True

        self.evaluator.resume()
        self.evaluator_closed = False


class Evaluator():

    def __init__(self, session, extractor):
        self.in_progress =True
        self.sender_closed = False

        self.session = session
        self.extractor = extractor

        self.num_workers = 16
        self.num_using_gpu = 8

        self.start_index = 2
        self.scanned_index = 1
        self.temporal_gap = 2
        self.actual_start_index = 2
        self.wait_time = 0.3

        self.scores = []

        global scanning_pool
        scanning_pool = GreenPool(self.num_workers)

        copy_reg.pickle(types.MethodType, self._pickle_method)

        self.scanner = Scanner(self.session.image_folder, self.session.flow_folder,
                               self.num_workers, self.num_using_gpu, self.session.use_spatial_net)

        self.sender = Sender(self.session, self.extractor, self)
        self.sender_thread = threading.Thread(target=self.sender.run, name='Sender')


    def run(self):
        self.child_thread_started = False

        while True:
            while not self.in_progress:
                time.sleep(0.5)

            if not self.child_thread_started:
                self.sender_thread.start()
                self.child_thread_started = True

            while self.in_progress:
                while self.extractor.extracted_index - self.temporal_gap <= self.scanned_index and self.in_progress:
                    time.sleep(self.wait_time)

                if self.in_progress:
                    self.actual_extracted_index = self.extractor.extracted_index
                    self.end_index = self.actual_extracted_index - self.temporal_gap

                    with self.session.print_lock:
                        print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Evaluator', 'Evaluating', self.start_index, self.end_index)

                    scan_start_time = time.time()

                    return_scores = []
                    return_scores += self.scanner.scan(self.start_index, self.end_index, self.actual_extracted_index)

                    self.sender.scores += return_scores

                    self.scan_time = (time.time() - scan_start_time) / len(return_scores)

                    self.scanned_index = self.end_index
                    self.start_index = self.end_index + 1

                    gc.collect()


            self.finalize()


    def _pickle_method(self,m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)


    def finalize(self):
        global scanning_pool

        self.sender.in_progress = False

        scanning_pool.close()
        scanning_pool.join()
        del scanning_pool
        del self.scanner

        while not self.sender_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        gc.collect()

        self.extractor.evaluator_closed = True


    def resume(self):
        self.start_index = 2
        self.scanned_index = 1
        self.actual_start_index = 2

        self.scores = []

        global scanning_pool
        scanning_pool = Pool(processes=self.num_workers)

        self.scanner = Scanner(self.session.image_folder, self.session.flow_folder, self.num_workers,
                               self.num_using_gpu, self.session.use_spatial_net)

        self.in_progress = True

        self.sender.resume()
        self.sender_closed = False


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


    def scan(self, start_index, end_index, actual_extracted_index):
        manager = Manager()
        scan_scores = manager.list()

        indices = range(start_index, end_index + 1, 1)

        for i in xrange(len(indices)):
            scan_scores.append([0.0, 0.0])

        scanning_pool.imap(self.scanVideo,
                          zip([start_index] * len(indices),
                              [actual_extracted_index] * len(indices),
                              [scan_scores] * len(indices),
                              indices))
        return_scores = []
        return_scores += scan_scores

        return return_scores


    def scanVideo(self, scan_items):
        global spatial_net_gpu_01
        global spatial_net_gpu_02
        global temporal_net_gpu_01
        global temporal_net_gpu_02

        start_index = scan_items[0]
        frame_count = scan_items[1]
        scan_scores = scan_items[2]
        index = scan_items[3]

        current = current_process()
        current_id = current._identity[0] - 1

        if current_id % self.num_workers < self.num_using_gpu:
            spatial_net = spatial_net_gpu_01
            temporal_net = temporal_net_gpu_02
        else:
            spatial_net = spatial_net_gpu_01
            temporal_net = temporal_net_gpu_02

        score_layer_name = 'fc-twis'

        if self.use_spatial_net:
            image_frame = cv2.imread(os.path.join(self.image_folder, 'img_{:07d}.jpg'.format(index)))

            rgb_score = \
                spatial_net.predict_single_frame([image_frame, ], score_layer_name, over_sample=False,
                                                 frame_size=None)[0].tolist()

        print rgb_score

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

        print flow_score

        if self.use_spatial_net:
            scan_scores[index - start_index] = np.divide(np.clip(np.asarray([rgb_score[i] * self.rate_of_space
                                                                     + flow_score[i] * self.rate_of_time for i in xrange(len(flow_score))]),
                                                                 -self.score_bound, self.score_bound),
                                                         self.score_bound).tolist()
        else:
            scan_scores[index - start_index] = np.divide(np.clip(np.asarray([flow_score[i] * self.rate_of_whole for i in xrange(len(flow_score))]),
                                                                 -self.score_bound, self.score_bound),
                                                         self.score_bound).tolist()


class Sender():

    def __init__(self, session, extractor, evaluator):
        self.in_progress = True

        self.session = session
        self.extractor = extractor
        self.evaluator = evaluator

        self.scores = []

        self.sub_model_server_ip_address = '115.145.173.237'
        self.sub_model_server_port_number = 11224

        self.main_model_server_ip_address = '115.145.173.160'
        self.main_model_server_port_number = random.sample(range(10000, 20000, 1), 1)[0]

        self.entire_boundary = '!entire_boundary!'
        self.element_boundary = '!element_boundary!'


    def run(self):
        while True:
            while not self.in_progress:
                time.sleep(0.3)

            # self.main_model_server_port_number = random.sample(range(10000, 20000, 1), 1)[0]
            # self.main_model_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.main_model_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # self.main_model_server_socket.bind((self.main_model_server_ip_address, self.main_model_server_port_number))
            # self.main_model_server_socket.connect((self.sub_model_server_ip_address, self.sub_model_server_port_number))

            self.start_index = 2
            self.sent_index = 1
            while self.in_progress:
                while self.sent_index >= self.evaluator.scanned_index and self.in_progress:
                    time.sleep(0.3)

                sending_scores = []
                sending_scores += self.scores
                number_of_sending_scores = len(sending_scores)
                sending_start_index = self.start_index
                sending_end_index = sending_start_index + number_of_sending_scores -1

                with self.session.print_lock:
                    print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Sender', 'Sending', sending_start_index, sending_end_index)

                for frame_index in range(sending_start_index, sending_end_index+1, 1):
                    image = cv2.imread(os.path.join(self.session.image_folder, 'img_{:07d}.jpg'.format(frame_index)))
                    flow_x = cv2.imread(os.path.join(self.session.flow_folder, 'flow_x_{:07d}.jpg'.format(frame_index)), cv2.IMREAD_GRAYSCALE)
                    flow_y = cv2.imread(os.path.join(self.session.flow_folder, 'flow_y_{:07d}.jpg'.format(frame_index)), cv2.IMREAD_GRAYSCALE)

                    image_data = cv2.imencode('.jpg', image)[1].tostring()
                    flow_x_data = cv2.imencode('.jpg', flow_x)[1].tostring()
                    flow_y_data = cv2.imencode('.jpg', flow_y)[1].tostring()

                    frame_score = sending_scores[frame_index - sending_start_index]

                    send_data = b'{}{}{}{}{}{}{}{}{}{}'.format(image_data, self.element_boundary, flow_x_data, self.element_boundary,
                                                                 flow_y_data, self.element_boundary, frame_score[0], self.element_boundary,
                                                                 frame_score[1], self.entire_boundary)

                    # try:
                    #     self.main_model_server_socket.send(send_data)
                    # except:
                    #     pass

                    try:
                        os.remove(os.path.join(self.session.image_folder, 'img_{:07d}.jpg'.format(frame_index)))
                    except OSError:
                        pass

                    try:
                        os.remove(os.path.join(self.session.flow_folder, 'flow_x_{:07d}.jpg'.format(frame_index)))
                    except OSError:
                        pass

                    try:
                        os.remove(os.path.join(self.session.flow_folder, 'flow_y_{:07d}.jpg'.format(frame_index)))
                    except OSError:
                        pass

                del self.scores[0:number_of_sending_scores]
                gc.collect()

                self.sent_index = sending_end_index
                self.start_index = sending_end_index + 1


    def finalize(self):
        self.in_progress = False

        gc.collect()

        self.evaluator.sender_closed = True


    def resume(self):
        self.scores = []

        self.in_progress = True



if __name__ == '__main__':
    session = Session()

    while True:
        time.sleep(13.1451)
        with session.print_lock:
            print '------------------------------ Memory Checking ---------------------------------'
            cmd = 'free -h'
            with session.extractor.cmd_lock:
                os.system(cmd)
                sys.stdout.flush()
            print '--------------------------------------------------------------------------------'