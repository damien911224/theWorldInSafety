import socket
import numpy as np
import cv2
import datetime
import threading
import time
import random
from multiprocessing import Lock


class Raspberry():

    def __init__(self):
        self.print_lock = Lock()
        self.client_name = self.getRaspberryIpAddress()

        self.camera = self.Camera(self)
        self.controller = self.Controller(self)

        self.camera_thread = threading.Thread(target=self.camera.run, name='Camera')
        self.controller_thread = threading.Thread(target=self.controller.run, name='Controller')

        self.raspberry_thread = threading.Thread(target=self.run, name='Raspberry')
        self.raspberry_thread.start()


    def run(self):
        self.camera_thread.start()
        self.controller_thread.start()

        while True:
            time.sleep(0.5)


    def getRaspberryIpAddress(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))

        raspberry_ip_address = sock.getsockname()[0]
        sock.close()

        with self.print_lock:
            print '{:10s}|{:12s}|{}'.format('Raspberry', 'Connection', 'With IP {}'.format(raspberry_ip_address))

        return raspberry_ip_address


    class Camera():

        def __init__(self, raspberry):
            self.raspberry = raspberry

            self.in_progress = True
            self.web_cam_device_id = 1
            self.use_webcam = True
            self.test_video = '/home/parallels/theWorldInSafety/raspberry/test_videos/demo_7.mp4'
            self.want_to_resize = False
            self.resize_size = ( 60.0, 60.0 )
            self.original_size = ( 640, 480 )

            self.camera_socket = None
            self.server_ip_address = '13.228.168.156'
            self.server_port_number = 7777

            self.client_name = self.raspberry.client_name
            self.client_port_number = random.sample(range(10000, 20000, 1), 1)[0]

            self.jpg_boundary = b'!TWIS_END!'
            self.session_name = None
            self.session_is_open = False
            self.session_index = 1
            self.window_name = 'Raspberry'
            self.window_position = (0, 0)

            self.visualization = True
            self.display_term = 300
            self.motionDetector = self.MotionDetector(self)


        def run(self):
            while True:
                while not self.in_progress:
                    time.sleep(0.3)

                self.session_is_open = False

                if self.use_webcam:
                    video_cap = cv2.VideoCapture(self.web_cam_device_id)
                else:
                    video_cap = cv2.VideoCapture(self.test_video)
                video_fps = int(video_cap.get(cv2.CAP_PROP_FPS)) + 1
                self.wait_time = max(int(1000.0 / float(video_fps)), 1)

                if self.want_to_resize:
                    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize_size[0])
                    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize_size[1])
                    with self.raspberry.print_lock:
                        print  '{:10s}|{:12s}|{}'.format('Camera', 'Resizing Camera', 
                                '( {}, {} )'.format(int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                    int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


                if video_cap.isOpened():
                    while self.in_progress:
                        ok, frame = video_cap.read()
                        if self.want_to_resize:
                            frame = cv2.resize(frame, self.original_size, interpolation = cv2.INTER_AREA)
                        if not ok:
                            break

                        is_moving = self.motionDetector.motionDetect(frame)
                        if is_moving:
                            if not self.session_is_open:
                                self.session_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                self.session_index = 1

                                self.motionDetector = self.MotionDetector(self)

                                self.client_port_number = random.sample(range(10000, 20000, 1), 1)[0]
                                self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                                self.camera_socket.bind((self.client_name, self.client_port_number))
                                self.camera_socket.connect((self.server_ip_address, self.server_port_number))

                                self.session_is_open = True

                                with self.raspberry.print_lock:
                                    print '{:10s}|{:12s}|{}'.format('Camera', 'Session Start', self.session_name)

                            self.send(frame)
                            self.session_index += 1
                        else:
                            if self.session_is_open:
                                self.camera_socket.close()
                                self.camera_socket = None

                                self.session_is_open = False
                        
                        if self.visualization:
                            visualized_frame = self.visualize(frame, is_moving)
                            cv2.imshow(self.window_name + '|'+ self.session_name, visualized_frame)
                            cv2.moveWindow(self.window_name + '|'+ self.session_name, self.window_position[0], self.window_position[1])
                            cv2.waitKey(self.wait_time)

                        if self.session_index % self.display_term == 0:
                            with self.raspberry.print_lock:
                                print '{:10s}|{:12s}|{}'.format('Camera', 'Sending Frames', 'Unitl {:07d}'.format(self.session_index))

                    video_cap.release()


        def send(self, frame):
            header = b'raspberry{:15s}{:07d}{:14s}'.format(self.session_name, self.session_index,
                                                           datetime.datetime.now().strftime('%M%S%s'))
            frame_data = cv2.imencode('.jpg', frame)[1].tostring()
            frame_data_length = len(frame_data)
            send_data = header + b'{:07d}{}{}'.format(frame_data_length, frame_data, self.jpg_boundary)
            try:
                self.camera_socket.send(send_data)
            except socket.error:
                pass


        def visualize(self, frame, is_moving):
            font = cv2.FONT_HERSHEY_SIMPLEX
            top_left = (0, 0)
            text_margin = (15, 10)

            scale = 0.60
            thickness = 2
            line_type = cv2.LINE_8

            text_color = (255, 255, 255)

            if is_moving:
                frame_label = 'Motion Detected'
                label_background_color = (150, 150, 255)
            else:
                frame_label = 'No Movement'
                label_background_color = (250, 150, 150)

            label_text = 'Frame {:07d} | {:^8s} | Timer {:05d}'.format(self.session_index,
                                                                       frame_label,
                                                                       max(self.motionDetector.countdown_to_stop, 0))

            box_size, dummy = cv2.getTextSize(label_text, font, scale, thickness)

            box_top_left = top_left
            box_bottom_right = (frame.shape[1], top_left[1] + box_size[1] + text_margin[1] * 2)
            box_width = box_bottom_right[0]
            box_height = box_bottom_right[1]

            image_label_text_bottom_left = (top_left[0] + text_margin[0], top_left[1] + text_margin[1] + box_size[1])

            image_box_top_left = box_top_left
            image_box_bottom_right = box_bottom_right

            image_headline = np.zeros((box_height, box_width, 3), dtype='uint8')
            cv2.rectangle(image_headline, image_box_top_left, image_box_bottom_right,
                          label_background_color, cv2.FILLED)

            cv2.putText(image_headline, label_text, image_label_text_bottom_left,
                        font, scale, text_color, thickness, line_type)

            image_frame = np.concatenate((image_headline, frame), axis=0)

            boundary_top_left = (0, 0)
            image_boundary_bottom_right = (image_frame.shape[1], image_frame.shape[0])
            boundary_color = (255, 255, 255)

            cv2.rectangle(image_frame, boundary_top_left, image_boundary_bottom_right,
                          boundary_color, thickness=3)

            return image_frame


        class MotionDetector():

            def __init__(self, camera):
                self.camera = camera

                self.frame_diff_threshold = 0.45
                self.minimum_capture_count = 300

                self.countdown_to_stop = self.minimum_capture_count
                self.no_moving = False

                self.frame_avg = None
                self.initialized = False


            def motionDetect(self, frame):
                if not self.initialized:
                    frame_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                    self.frame_avg = frame_gray
                    self.initialized = True

                frame_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                frame_diff_avg = np.divide(np.sum(cv2.absdiff(frame_gray, cv2.convertScaleAbs(self.frame_avg))),
                                           len(frame.flatten()))

                self.frame_avg = np.divide(self.frame_avg + np.multiply(frame_gray, 0.5), 1.5)

                if frame_diff_avg >= self.frame_diff_threshold:
                    if self.no_moving:
                        self.no_moving = False
                    self.countdown_to_stop = self.minimum_capture_count
                else:
                    self.countdown_to_stop -= 1
                    if self.countdown_to_stop <= 0:
                        self.no_moving = True

                return not self.no_moving


    class Controller():

        def __init__(self, raspberry):
            self.raspberry = raspberry

            self.controller_socket = None
            self.server_ip_address = '13.228.168.156'
            self.server_port_number = 9999

            self.client_name = self.raspberry.client_name
            self.client_port_number = random.sample(range(20000, 30000, 1), 1)[0]


        def run(self):
            while True:
                message = raw_input()

                self.client_port_number = random.sample(range(20000, 30000, 1), 1)[0]
                self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.controller_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.controller_socket.bind((self.client_name, self.client_port_number))
                self.controller_socket.connect((self.server_ip_address, self.server_port_number))

                if message == 'stop':
                    self.raspberry.camera.in_progress = False
                    cv2.destroyAllWindows()
                    self.send(message)
                elif message == 'resume' or message == 'start':
                    self.send(message)
                    self.raspberry.camera.in_progress = True
                elif message == 'reset':
                    self.send(message)

                self.controller_socket.close()

                with self.raspberry.print_lock:
                    print '{:10s}|{:12s}|{}'.format('Controller', 'Message', message.upper())


        def send(self, message):
            send_message = b'raspberry{:15s}!TWIS_END!'.format(message)
            try:
                self.controller_socket.send(send_message)
            except socket.error:
                pass

            if message == 'resume' or message == 'start':
                while True:
                    try:
                        r = str(self.controller_socket.recv(90456)).replace(' ', '')
                        if r == 'Ready':
                            break
                    except socket.error:
                        pass



if __name__ == '__main__':
    raspberry = Raspberry()

    while True:
        time.sleep(10.314)
