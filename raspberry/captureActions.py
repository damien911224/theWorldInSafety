import os
import sys
import glob
import socket
import numpy as np
import cv2
import datetime
import threading
import time



class Raspberry():

    def __init__(self):
        self.camera = self.Camera(self)
        self.camera_thread = threading.Thread(target=self.camera.run, name='Camera')

        self.raspberry_thread = threading.Thread(target=self.run, name='Raspberry')
        self.raspberry_thread.start()


    def run(self):
        self.camera_thread.start()

        while True:
            time.sleep(0.5)


    class Camera():

        def __init__(self, raspberry):
            self.raspberry = raspberry

            self.web_cam_device_id = 0

            self.camera_socket = None
            # self.server_ip_address = 'localhost'
            self.server_ip_address = '13.125.52.6'
            self.server_port_number = 7777

            self.client_name = '127.0.0.1'
            self.client_port_number = 21224

            self.jpg_boundary = b'!TWIS_END!'
            self.session_name = None
            self.session_is_open = False
            self.session_index = 1

            self.motionDetector = self.MotionDetector(self)


        def run(self):
            video_cap = cv2.VideoCapture(self.web_cam_device_id)

            if video_cap.isOpened():
                while True:
                    ok, frame = video_cap.read()
                    if not ok:
                        break

                    is_moving = self.motionDetector.motionDetect(frame)
                    if is_moving:
                        if not self.session_is_open:
                            self.session_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            self.session_index = 1

                            self.motionDetector = self.MotionDetector(self)

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


        def send(self, frame):
            header = b'{:15s}{:07d}'.format(self.session_name, self.session_index)
            frame_data = cv2.imencode('.jpg', frame)[1].tostring()
            send_data = header + frame_data + self.jpg_boundary
            try:
                self.camera_socket.send(send_data)
            except socket.error:
                print 'SOCKET ERROR!'


        class MotionDetector():
            def __init__(self, camera):
                self.camera = camera

                self.frame_diff_threshold = 1.0
                self.minimum_capture_count = 100

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



if __name__ == '__main__':
    raspberry = Raspberry()

    while True:
        time.sleep(10.314)