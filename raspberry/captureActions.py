import os
import sys
import glob
import numpy as np
import cv2
import datetime
import threading
import time



class Raspberry():

    def __init__(self):
        self.server_ip_address = '127.0.0.1'
        self.server_port_number = 11224

        self.session_name = None

        self.motion_detector = self.MotionDetector(self)
        self.motion_detector_thread = threading.Thread(target=self.motion_detector.run, name='Motion Detector')

        self.raspberry_thread = threading.Thread(target=self.run, name='Raspberry')
        self.raspberry_thread.start()

    def run(self):
        self.motion_detector_thread.start()

        while True:
            time.sleep(0.3)


    class MotionDetector():

        def __init__(self, raspberry):
            self.raspberry = raspberry

            self.frame_diff_threshold = 1.0
            self.minimum_capture_count = 1000
            self.web_cam_device_id = 0

            self.countdown_to_stop = self.minimum_capture_count
            self.no_moving = False


        def run(self):
            video_cap = cv2.VideoCapture(self.web_cam_device_id)

            if video_cap.isOpened():
                frame_avg = None
                initialized = False

                while True:
                    ok, frame = video_cap.read()
                    if not ok:
                        break

                    if not initialized:
                        frame_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21,21), 0)
                        frame_avg = frame_gray
                        initialized = True
                        continue

                    frame_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                    frame_diff_avg = np.divide(np.sum(cv2.absdiff(frame_gray, cv2.convertScaleAbs(frame_avg))), len(frame.flatten()))

                    frame_avg = np.divide(frame_avg + np.multiply(frame_gray, 0.5), 1.5)

                    cv2.imshow('Hi', frame)
                    cv2.waitKey(1)

                    if frame_diff_avg >= self.frame_diff_threshold:
                        self.countdown_to_stop = self.minimum_capture_count
                    else:
                        self.countdown_to_stop -= 1

                    print self.countdown_to_stop

                    if self.countdown_to_stop <= 0:
                        self.no_moving = True



if __name__ == '__main__':
    raspberry = Raspberry()

    while True:
        time.sleep(3)