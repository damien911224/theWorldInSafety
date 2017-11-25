import os
import sys
import glob
import socket
import numpy as np
import cv2
import datetime
import threading
import time
import random



class Raspberry():

    def __init__(self):
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


    class Camera():

        def __init__(self, raspberry):
            self.raspberry = raspberry

            self.in_progress = True
            self.web_cam_device_id = 0

            self.camera_socket = None
            # self.server_ip_address = 'localhost'
            self.server_ip_address = '13.125.52.6'
            self.server_port_number = 7777

            # self.client_name = '127.0.0.1'
            self.client_name = '10.211.55.10'
            self.client_port_number = random.sample(range(10000, 20000, 1), 1)[0]

            self.jpg_boundary = b'!TWIS_END!'
            self.session_name = None
            self.session_is_open = False
            self.session_index = 1

            self.motionDetector = self.MotionDetector(self)


        def run(self):
            while True:
                while not self.in_progress:
                    print 'waiting'
                    time.sleep(0.3)

                video_cap = cv2.VideoCapture(self.web_cam_device_id)

                if video_cap.isOpened():
                    while self.in_progress:
                        ok, frame = video_cap.read()
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

                            self.send(frame)
                            self.session_index += 1
                        else:
                            if self.session_is_open:
                                self.camera_socket.close()
                                self.camera_socket = None

                                self.session_is_open = False

                    video_cap.release()


        def send(self, frame):
            header = b'raspberry{:15s}{:07d}'.format(self.session_name, self.session_index)
            frame_data = cv2.imencode('.jpg', frame)[1].tostring()
            send_data = header + frame_data + self.jpg_boundary
            try:
                self.camera_socket.send(send_data)
            except socket.error:
                print 'CAMERA SOCKET ERROR!'


        class MotionDetector():
            def __init__(self, camera):
                self.camera = camera

                self.frame_diff_threshold = 1.0
                self.minimum_capture_count = 20

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
            self.server_ip_address = '13.125.52.6'
            self.server_port_number = 9999

            self.client_name = '10.211.55.10'
            self.client_port_number = random.sample(range(20000, 30000, 1), 1)[0]


        def run(self):
            while True:
                message = raw_input()
                print 'message: {}'.format(message)

                self.client_port_number = random.sample(range(20000, 30000, 1), 1)[0]
                self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.controller_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.controller_socket.bind((self.client_name, self.client_port_number))
                self.controller_socket.connect((self.server_ip_address, self.server_port_number))

                if message == 'stop':
                    self.raspberry.camera.in_progress = False
                    self.send(message)
                elif message == 'resume' or message == 'start':
                    self.send(message)
                    self.raspberry.camera.in_progress = True

                self.controller_socket.close()


        def send(self, message):
            send_message = b'raspberry{:15s}!TWIS_END!'.format(message)
            try:
                self.controller_socket.send(send_message)
            except socket.error:
                print 'SOCKET ERROR!'

            if message == 'resume' or message == 'start':
                while True:
                    try:
                        r = str(self.controller_socket.recv(90456)).replace(' ', '')
                        if r == 'Ready':
                            break
                    except socket.error:
                        print 'SOCKET ERROR!'


if __name__ == '__main__':
    raspberry = Raspberry()

    while True:
        time.sleep(10.314)