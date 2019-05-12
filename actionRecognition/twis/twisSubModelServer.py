import cv2
import sys
import os
import numpy as np
import time
import glob
import math
import threading
from sklearn import mixture
from pipes import quote
from multiprocessing import Lock
import gc
import socket
from shutil import copyfile
from shutil import rmtree
import random
import datetime
import pycurl
import imutils
sys.path.append("../../semanticPostProcessing")
sys.path.append('../../semanticPostProcessing/darkflow')
from post_process import SemanticPostProcessor


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
        self.fps = 15.0
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

        self.print_lock = Lock()
        self.average_delay = 0.0

        self.server_ip_address = ''
        self.server_port_number = 8888

        self.client_host_name = ''
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
                            accumulated_data = previous_data + b''
                            try:
                                while self.in_progress:
                                    recv_data = self.client_socket.recv(90456)
                                    if len(recv_data) == 0:
                                        socket_closed = True
                                        break

                                    accumulated_data += recv_data
                                    found = accumulated_data.find(b'!TWIS_END!')
                                    if found != -1:
                                        previous_data = accumulated_data[found+10:]
                                        accumulated_data = accumulated_data[:found]
                                        break
                            except:
                                continue

                            frame_data = accumulated_data

                            if socket_closed:
                                break

                            if self.in_progress:
                                header = frame_data[:22]
                                session_name = str(header[:15])
                                frame_index = int(header[15:22])
                                frame_data = frame_data[22:]

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


    def dumpFrames(self, frames):
        end_index = self.start_index + len(frames) - 1
        if end_index % self.print_term == 0:
            with self.print_lock:
                print '{:10s}|{:12s}| Until {:07d}|Delay {:.6f} Seconds'.format('Session', 'Dumping', end_index, self.average_delay)

        index = self.start_index
        for frame in frames:
            file_name = os.path.join(self.image_folder, 'show_{:07d}.jpg'.format(index))
            new_frame = cv2.resize(frame, self.show_size, interpolation=cv2.INTER_AREA)
            if self.is_rotated:
                new_frame = imutils.rotate(new_frame, self.rotating_angle)
            cv2.imwrite(file_name, new_frame)

            file_name = os.path.join(self.image_folder, 'img_{:07d}.jpg'.format(index))
            new_frame = cv2.resize(frame, self.new_size, interpolation=cv2.INTER_AREA)
            if self.is_rotated:
                new_frame = imutils.rotate(new_frame, self.rotating_angle)
            cv2.imwrite(file_name, new_frame)
            index += 1


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
        self.wait_time = 0.1


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
        self.analyzer_closed = False
        self.secretary_closed = False
        self.closer_closed = False

        self.session = session
        self.extractor = extractor

        self.start_index = 2
        self.scanned_index = 1
        self.temporal_gap = 2
        self.actual_start_index = 2
        self.wait_time = 0.1

        self.scores = []

        self.analyzer = Analyzer(self.session, self.extractor, self)
        self.analyzer_thread = threading.Thread(target=self.analyzer.run, name='Analyzer')

        self.secretary = Secretary(self.session, self.extractor, self, self.analyzer)
        self.secretary_thread = threading.Thread(target=self.secretary.run, name='Secretary')

        self.closer = Closer(self.session, self.extractor, self, self.analyzer, self.secretary)
        self.closer_thread = threading.Thread(target=self.closer.run, name='Closer')

        self.main_model_server_ip_address = ''
        self.main_model_server_port_number = 7777

        self.element_boundary = b'!element_boundary!'
        self.one_boundary = b'!one_boundary!'
        self.entire_boundary = b'!entire_boundary!'


    def run(self):
        self.child_thread_started = False

        while True:
            while not self.in_progress:
                time.sleep(0.5)

            if not self.child_thread_started:
                self.analyzer_thread.start()
                self.secretary_thread.start()
                self.closer_thread.start()
                self.child_thread_started = True

            self.client_port_number = random.sample(range(10000, 20000, 1), 1)[0]
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.client_socket.bind((self.session.client_host_name, self.client_port_number))
            self.client_socket.connect((self.main_model_server_ip_address, self.main_model_server_port_number))

            socket_closed = False
            while self.in_progress:
                while self.extractor.extracted_index - self.temporal_gap <= self.scanned_index and self.in_progress:
                    time.sleep(self.wait_time)

                if self.in_progress:
                    self.actual_extracted_index = self.extractor.extracted_index
                    self.end_index = self.actual_extracted_index - self.temporal_gap

                    with self.session.print_lock:
                        print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Evaluator', 'Evaluating', self.start_index, self.end_index)

                    scan_start_time = time.time()

                    entire_send_data = b''
                    for frame_index in range(self.start_index - 2, self.actual_extracted_index + 1, 1):
                        if frame_index <= 1:
                            frame_index = 2

                        image_path = os.path.join(self.session.image_folder, 'img_{:07d}.jpg'.format(frame_index))
                        flow_x_path = os.path.join(self.session.flow_folder, 'flow_x_{:07d}.jpg'.format(frame_index))
                        flow_y_path = os.path.join(self.session.flow_folder, 'flow_y_{:07d}.jpg'.format(frame_index))

                        image = cv2.imread(image_path)
                        flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)
                        flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)

                        image_data = cv2.imencode('.jpg', image)[1].tostring()
                        flow_x_data = cv2.imencode('.jpg', flow_x)[1].tostring()
                        flow_y_data = cv2.imencode('.jpg', flow_y)[1].tostring()


                        if frame_index < self.actual_extracted_index:
                            send_data = b'{}{}{}{}{}{}'.format(image_data, self.element_boundary,
                                                               flow_x_data, self.element_boundary,
                                                               flow_y_data, self.one_boundary)
                        else:
                            send_data = b'{}{}{}{}{}{}{}'.format(image_data, self.element_boundary,
                                                               flow_x_data, self.element_boundary,
                                                               flow_y_data, self.one_boundary, self.entire_boundary)

                        entire_send_data += send_data


                    try:
                        self.client_socket.send(entire_send_data)
                    except:
                        socket_closed = True
                        break


                    scores_data = b''
                    while True:
                        recv_data = self.client_socket.recv(90456)
                        if len(recv_data) == 0:
                            break
                        finder = recv_data.find(self.entire_boundary)
                        if finder != -1:
                            scores_data += recv_data[:finder]
                            break
                        else:
                            scores_data += recv_data


                    return_scores = []
                    for segment_index in range(0, len(scores_data), 14):
                        violence_score = float(scores_data[segment_index:segment_index+7])
                        normal_score = float(scores_data[segment_index+7:segment_index+14])
                        scores = [ violence_score, normal_score ]
                        return_scores.append(scores)

                    self.scan_time = (time.time() - scan_start_time) / len(return_scores)

                    self.scores += return_scores
                    self.analyzer.keeping_scores += return_scores
                    self.secretary.showing_scores += return_scores
                    del return_scores

                    self.scanned_index = self.end_index
                    self.start_index = self.end_index + 1

                    gc.collect()

                if socket_closed:
                    self.client_socket.close()
                    break


            self.finalize()


    def finalize(self):
        self.analyzer.in_progress = False
        self.secretary.in_progress = False
        self.closer.in_progress = False

        while not self.analyzer_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.session.print_lock:
            print '{:10s}|{} Finalized'.format('Evaluator', 'Analyzer')

        while not self.secretary_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.session.print_lock:
            print '{:10s}|{} Finalized'.format('Evaluator', 'Secretary')

        while not self.closer_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.session.print_lock:
            print '{:10s}|{} Finalized'.format('Evaluator', 'Closer')

        gc.collect()

        self.extractor.evaluator_closed = True


    def resume(self):
        self.start_index = 2
        self.scanned_index = 1
        self.actual_start_index = 2

        self.scores = []
        self.in_progress = True

        self.analyzer.resume()
        self.analyzer_closed = False

        self.secretary.resume()
        self.secretary_closed = False

        self.closer.resume()
        self.closer_closed = False


class Analyzer():

    def __init__(self, session, extractor, evaluator):
        self.in_progress = True

        self.session = session
        self.extractor = extractor
        self.evaluator = evaluator

        self.analyzing_start_index = 0
        self.analyzed_index = 1
        self.wait_time = 0.15
        self.violence_index = 0
        self.normal_index = 1
        self.lower_bound = 0.0
        self.max_lower_bound = 0.0
        self.variance_factor = 2.0
        self.max_falling_count = 5
        self.falling_counter = 0
        self.median_kernal_size = 5

        self.real_base = 2
        self.not_yet = False
        self.max_iter = 500

        self.keeping_scores = []
        self.keeping_base = 2

        self.keep_number = 1


    def run(self):
        while True:
            while not self.in_progress:
                time.sleep(0.5)


            while self.in_progress:
                while self.analyzed_index >= self.evaluator.scanned_index and self.in_progress:
                    time.sleep(self.wait_time)

                if self.in_progress:
                    self.analyzed_index = self.evaluator.scanned_index
                    self.analyzing_end_index = max(len(self.evaluator.scores) - 1, 0)
                    self.not_yet = False

                    with self.session.print_lock:
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
                                                if compare_start - current_end < self.session.video_fps:
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
                            keep_folder = os.path.join(self.session.keep_folder, 'keep_{:07d}'.format(self.keep_number))
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
                                image_src_path = os.path.join(self.session.image_folder, 'show_{:07d}.jpg'.format(index))
                                flow_x_src_path = os.path.join(self.session.flow_folder, 'flow_x_{:07d}.jpg'.format(index))
                                flow_y_src_path = os.path.join(self.session.flow_folder, 'flow_y_{:07d}.jpg'.format(index))

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
                        if gap_of_keeping_and_current >= int(100.0 * self.session.video_fps):
                            del self.keeping_scores[0:gap_of_keeping_and_current]
                            self.keeping_base += gap_of_keeping_and_current

                        if self.not_yet:
                            break

                        gc.collect()


            self.finalize()


    def finalize(self):
        gc.collect()

        self.evaluator.analyzer_closed = True


    def resume(self):
        self.analyzing_start_index = 0
        self.analyzed_index = 1
        self.falling_counter = 0

        self.real_base = 2
        self.not_yet = False

        self.keeping_scores = []
        self.keeping_base = 2

        self.keep_number = 1

        self.in_progress = True


class Secretary():

    def __init__(self, session, extractor, evaluator, analyzer):
        self.in_progress = True
        self.progress_viewer_closed = False
        self.clip_viewer_closed = False

        self.session = session
        self.extractor = extractor
        self.evaluator = evaluator
        self.analyzer = analyzer

        self.progress_viewer = self.Viewer(self)
        self.progress_viewer.window_position = (0, 0)
        self.progress_viewer.every_time_close = False
        self.progress_viewer.view_type = 'frames'
        self.progress_viewer.step = 1.0
        self.progress_viewer_thread = threading.Thread(target=self.progress_viewer.run, name='Progress Viewer')

        self.clip_viewer = self.Viewer(self)
        self.clip_viewer.view_time = self.session.wait_time
        self.clip_viewer.every_time_close = True
        self.clip_viewer.view_type = 'clips'
        self.clip_viewer_thread = threading.Thread(target=self.clip_viewer.run, name='Clip Viewer')

        self.start_index = 2
        self.end_index = 2
        self.wait_time = 0.2
        self.make_views_index = 1
        self.removing_start_index = 1
        self.removing_end_index = 1
        self.temporal_gap = 2
        self.violence_index = 0
        self.showing_scores = []
        self.view_has_next = False
        self.removing_late_term = int(100 * self.session.fps)

        self.progress_viewer.window_name = 'Progress Viewer | Session {}'.format(self.session.session_name)
        self.clip_viewer.window_name = 'Clip Viewer'


    def run(self):
        self.child_thread_started = False

        while True:
            while not self.in_progress:
                time.sleep(0.5)


            if not self.child_thread_started:
                self.progress_viewer_thread.start()
                self.clip_viewer_thread.start()
                self.child_thread_started = True

            while self.in_progress:
                while self.make_views_index >= self.evaluator.scanned_index and self.in_progress:
                    time.sleep(self.wait_time)

                if self.in_progress:
                    number_of_showing_scores = len(self.showing_scores)
                    self.end_index = self.start_index + number_of_showing_scores - 1

                    with self.session.print_lock:
                        print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Secretary', 'Viewing', self.start_index, self.end_index)


                    view_frames = []
                    for index in range(self.start_index, self.end_index + 1, 1):
                        frame = dict()

                        score = self.showing_scores[index - self.start_index]
                        image = os.path.join(self.session.image_folder, 'show_{:07d}.jpg'.format(index))
                        flow_x = os.path.join(self.session.flow_folder, 'flow_x_{:07d}.jpg'.format(index))
                        flow_y = os.path.join(self.session.flow_folder, 'flow_y_{:07d}.jpg'.format(index))

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


            self.finalize()


    def remove(self, start_index, end_index):
        with self.session.print_lock:
            print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Secretary', 'Removing', start_index, end_index)

        remove_path_prefixes = ['img', 'show', 'flow_x', 'flow_y']

        for index in range(start_index, end_index + 1, 1):
            for path_prefix in remove_path_prefixes:
                if path_prefix in ['img', 'show']:
                    remove_path = os.path.join(self.session.image_folder, '{}_{:07d}.jpg'.format(path_prefix, index))
                else:
                    remove_path = os.path.join(self.session.flow_folder, '{}_{:07d}.jpg'.format(path_prefix, index))
                try:
                    os.remove(remove_path)
                except:
                    pass


    def finalize(self):
        self.progress_viewer.in_progress = False
        self.clip_viewer.in_progress = False

        while not self.progress_viewer_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.session.print_lock:
            print '{:10s}|{} Finished'.format('Secretary', 'Progress Viewer')

        while not self.clip_viewer_closed:
            time.sleep(0.3)

        time.sleep(0.3)

        with self.session.print_lock:
            print '{:10s}|{} Finished'.format('Secretary', 'Clip Viewer')

        gc.collect()

        self.evaluator.secretary_closed = True


    def resume(self):
        self.start_index = 2
        self.end_index = 2
        self.make_views_index = 1
        self.removing_start_index = 1
        self.removing_end_index = 1
        self.showing_scores = []
        self.view_has_next = False
        self.removing_late_term = int(100 * self.session.video_fps)

        self.progress_viewer.window_name = 'Progress Viewer | Session {}'.format(self.session.session_name)
        self.clip_viewer.window_name = 'Clip Viewer'

        self.in_progress = True

        self.progress_viewer.resume()
        self.progress_viewer_closed = False

        self.clip_viewer.resume()
        self.clip_viewer_closed = False


    class Viewer():

        def __init__(self, secretary):
            self.in_progress = True
            self.secretary = secretary

            self.window_name = ''
            self.window_position = (0, 0)
            self.every_time_close = False
            self.view_has_next = False

            self.view_type = 'frames'

            self.viewed_index = -1
            self.wait_time = 0.2
            self.step = 1.0
            self.time_factor = 1.0
            self.view_time = 0.0
            self.view_frames = []
            self.view_clips = []
            self.violence_index = 0

            self.semantic_display_step = 2


        def run(self):
            while True:
                while not self.in_progress:
                    time.sleep(0.5)


                if self.view_type == 'frames':
                    flow_bound = 20.0
                    angle_bound = 180.0
                    magnitude_bound = flow_bound * flow_bound * 2.0

                    while self.in_progress:
                        while len(self.view_frames) <= 0 and self.in_progress:
                            time.sleep(self.wait_time)

                        if self.in_progress:
                            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                            cv2.moveWindow(self.window_name, self.window_position[0], self.window_position[1])

                            view_frames = []
                            view_frames += self.view_frames
                            view_time = max(min(int(self.view_time * 1000.0 * self.step / self.time_factor),
                                                int(1000.0 / self.secretary.session.video_fps * 3.0)),
                                            1)

                            for frame_index in range(0, len(view_frames), int(self.step)):
                                index = view_frames[frame_index]['index']
                                score = view_frames[frame_index]['score']
                                image = cv2.imread(view_frames[frame_index]['image'])

                                if frame_index % self.semantic_display_step == 0:
                                    semantics = self.secretary.session.extractor.evaluator.closer.semanticPostProcessor.single_frame_semantics(image)
                                    semantic_size = (image.shape[1], image.shape[0])
                                    for box in semantics:
                                        semantic_thick = int((semantic_size[1] + semantic_size[0]) // 300)
                                        semantic_label = box['label']
                                        semantic_confidence = box['confidence']
                                        semantic_topleft_x = box['topleft_x']
                                        semantic_topleft_y = box['topleft_y']
                                        semantic_bottomright_x = box['bottomright_x']
                                        semantic_bottomright_y = box['bottomright_y']
                                        if semantic_label == 'Adult':
                                            semantic_box_colors = (189, 166, 36)
                                        else:
                                            semantic_box_colors = (128, 65, 217)

                                        cv2.rectangle(image, (semantic_topleft_x, semantic_topleft_y),
                                                      (semantic_bottomright_x, semantic_bottomright_y),
                                                      semantic_box_colors, semantic_thick)
                                        cv2.putText(image, ("{0}".format(semantic_label)),
                                                    (semantic_topleft_x, semantic_topleft_y - 12), 2,
                                                    1.0, semantic_box_colors, 2)

                                flow_x = cv2.resize(cv2.imread(view_frames[frame_index]['flows'][0],cv2.IMREAD_GRAYSCALE),
                                                    self.secretary.session.show_size, interpolation=cv2.INTER_AREA)
                                flow_y = cv2.resize(cv2.imread(view_frames[frame_index]['flows'][1],cv2.IMREAD_GRAYSCALE),
                                                    self.secretary.session.show_size, interpolation=cv2.INTER_AREA)

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

                                label_text = 'Frame {:07d} | Prediction {:^8s} | Score {:06.02f}%'.format(index, frame_label,
                                                                                                       max(score) * 100.0)
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
                    while self.in_progress:
                        while len(self.view_clips) <= 0 and self.in_progress:
                            time.sleep(self.wait_time)

                        if self.in_progress:
                            view_clips = []
                            view_clips += self.view_clips

                            for clip in view_clips:
                                with self.secretary.session.print_lock:
                                    print '{:10s}|{:12s}| {}'.format('Viewer', 'Clip Viewing',
                                                                     clip.split('/')[-1])

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

                                # try:
                                #     os.remove(clip)
                                # except:
                                #     pass

                            del self.view_clips[0:len(view_clips)]
                            del view_clips
                            gc.collect()


                self.finalize()


        def finalize(self):
            gc.collect()

            if self.view_type == 'frames':
                cv2.destroyAllWindows()
                self.secretary.progress_viewer_closed = True
            elif self.view_type == 'clips':
                self.secretary.clip_viewer_closed = True


        def resume(self):
            self.viewed_index = -1
            self.view_time = 0.0
            self.view_frames = []
            self.view_clips = []

            self.in_progress = True


class Closer():

    def __init__(self, session, extractor, evaluator, analyzer, secretary):
        self.in_progress = True

        self.session = session
        self.extractor = extractor
        self.evaluator = evaluator
        self.analyzer = analyzer
        self.secretary = secretary

        self.clips = []
        self.wait_time = 0.2
        self.threshold = 0.0
        self.clip_number = 1
        self.clip_round = 3
        self.violence_index = 0
        self.normal_index = 1

        self.semantic_step = 1

        self.semanticPostProcessor = SemanticPostProcessor()


    def run(self):
        while True:
            while not self.in_progress:
                time.sleep(0.3)


            while self.in_progress:
                while len(self.clips) <= 0 and self.in_progress:
                    time.sleep(self.wait_time)

                if self.in_progress:
                    clips = []
                    clips += self.clips

                    for clip in clips:
                        isPassed, filtered_scores = self.check(clip)

                        if isPassed:
                            for frame in clip['frames']:
                                frame['score'] = filtered_scores[clip['frames'].index(frame)]

                            semantic_ok, clip_semantics = self.semanticPostProcessor.semantic_post_process(clip)
                            semantic_index = 0
                            for frame in clip['frames']:
                                if semantic_index >= len(clip_semantics):
                                    frame['semantics'] = []
                                else:
                                    frame['semantics'] = clip_semantics[semantic_index]
                                semantic_index += 1

                            if semantic_ok or True:
                                self.visualize(clip)
                            else:
                                rmtree(clip['keep_folder'], ignore_errors=True)

                        else:
                            rmtree(clip['keep_folder'], ignore_errors=True)

                    del self.clips[0:len(clips)]


            self.finalize()


    def check(self, clip):
        with self.session.print_lock:
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
        with self.session.print_lock:
            print '{:10s}|{:12s}| From {:07d} To {:07d}'.format('Closer', 'Clipping',
                                                                clip['time_intervals'][0], clip['time_intervals'][1])

        current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        admin_clip_send_path = os.path.join(self.session.clip_send_folder,
                                       'Admin_{}.mp4'.format(current_datetime))
        user_clip_send_path = os.path.join(self.session.clip_send_folder,
                                      'User_{}.mp4'.format(current_datetime))
        clip_view_path = os.path.join(self.session.clip_view_folder,
                                            'View_{}.mp4'.format(current_datetime))

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
                                    self.session.show_size, cv2.INTER_AREA)
                flow_y = cv2.resize(cv2.imread(frame['flows'][1], cv2.IMREAD_GRAYSCALE),
                                    self.session.show_size, cv2.INTER_AREA)

                if not user_writer_initialized:
                    video_fps = self.session.video_fps
                    video_fourcc = 0x00000021
                    video_size = (int(image.shape[1]), int(image.shape[0]))
                    user_video_writer = cv2.VideoWriter(user_clip_send_path, video_fourcc, video_fps, video_size)
                    user_writer_initialized = True

                for iter in xrange(round):
                    user_video_writer.write(image)


                semantics = frame['semantics']
                semantic_size = ( image.shape[1], image.shape[0] )
                for box in semantics:
                    semantic_thick = int((semantic_size[1] + semantic_size[0]) // 300)
                    semantic_label = box['label']
                    semantic_confidence = box['confidence']
                    semantic_topleft_x = box['topleft_x']
                    semantic_topleft_y = box['topleft_y']
                    semantic_bottomright_x = box['bottomright_x']
                    semantic_bottomright_y = box['bottomright_y']
                    if semantic_label == 'Adult':
                        semantic_box_colors = (189, 166, 36)
                    else:
                        semantic_box_colors = (128, 65, 217)

                    cv2.rectangle(image, (semantic_topleft_x, semantic_topleft_y),
                                  (semantic_bottomright_x, semantic_bottomright_y),
                                  semantic_box_colors, semantic_thick)
                    cv2.putText(image, ("{0}".format(semantic_label)),
                                (semantic_topleft_x, semantic_topleft_y - 12), 2,
                                1.0, semantic_box_colors, 2)


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

                label_text = 'Frame {:07d} | Prediction {:^8s} | Score {:06.02f}%'.format(index, frame_label,
                                                                                          max(score) * 100.0)

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
                    video_fps = self.session.video_fps
                    video_fourcc = 0x00000021
                    video_size = (int(frame.shape[1]), int(frame.shape[0]))
                    admin_video_writer = cv2.VideoWriter(admin_clip_send_path, video_fourcc, video_fps, video_size)
                    admin_writer_initialized = True

                for iter in xrange(round):
                    admin_video_writer.write(frame)

                del frame

        if admin_video_writer is not None:
            admin_video_writer.release()

        if user_video_writer is not None:
            user_video_writer.release()

        try:
            copyfile(admin_clip_send_path, clip_view_path)
        except:
            pass


        if len(self.secretary.clip_viewer.view_clips) > 0:
            self.secretary.clip_viewer.view_has_next = True
        self.secretary.clip_viewer.view_clips.append(clip_view_path)

        clip_send_paths = [ admin_clip_send_path, user_clip_send_path ]

        send_thread = threading.Thread(target=self.send, args=[clip_send_paths])
        send_thread.start()

        rmtree(clip['keep_folder'])

        gc.collect()


    def send(self, clip_send_paths):
        admin_clip_send_path = clip_send_paths[0]
        user_clip_send_path = clip_send_paths[1]

        with self.session.print_lock:
            print '{:10s}|{:12s}| {} & {}'.format('Closer', 'Sending',
                                                  user_clip_send_path.split('/')[-1],
                                                  admin_clip_send_path.split('/')[-1])

        # try:
        #     c = pycurl.Curl()
        #     c.setopt(c.VERBOSE, 0)
        #     c.setopt(c.POST, 1)
        #     c.setopt(c.URL, 'http://13.228.101.253:8080/management/receive/')
        #     c.setopt(c.HTTPPOST,
        #                 [('admin_clip', (pycurl.FORM_FILE, admin_clip_send_path))])
        #                  # ('user_clip', (c.FROM_FILE, user_clip_send_path))])
        #     c.perform()
        #     c.close()
        # except:
        #     pass

        for send_clip_path in clip_send_paths:
            try:
                os.remove(send_clip_path)
            except:
                pass


    def finalize(self):
        self.evaluator.closer_closed = True


    def resume(self):
        self.clips = []
        self.clip_number = 1

        self.in_progress = True



if __name__ == '__main__':
    session = Session()

    while True:
        time.sleep(33.1451)
        with session.print_lock:
            print '------------------------------ Memory Checking ---------------------------------'
            cmd = 'free -h'
            with session.extractor.cmd_lock:
                os.system(cmd)
                sys.stdout.flush()
            print '--------------------------------------------------------------------------------'
