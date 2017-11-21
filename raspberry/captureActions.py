import os
import sys
import glob
import numpy as np
import cv2
import datetime
import threading
import time
from numpy import linalg as LA



class Raspberry():

    def __init__(self):
        self.server_ip_address = '127.0.0.1'
        self.server_port_number = 11224

        self.frame_diff_threshold = 10
        self.minimum_capture_count = 1000

        self.web_camera_device_id = 0

        self.session_name = None

        self.raspberry_thread = threading.Thread(target=self.run, name='Raspberry')
        self.raspberry_thread.start()


    def run(self):
        video_cap = cv2.VideoCapture(self.web_camera_device_id)

        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4.0
        histogram_norm_type = 0
        L2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        n_levels = 64

        win_stride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)

        hog_descriptior = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
                                            nbins, deriv_aperture, win_sigma, histogram_norm_type,
                                            L2_hys_threshold, gamma_correction, n_levels)


        if video_cap.isOpened():
            previous_frame = None
            initialized = False

            while True:
                ok, frame = video_cap.read()
                if not ok:
                    break

                if not initialized:
                    frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21,21), 0)
                    previous_hog_histograms = hog_descriptior.compute(frame, win_stride, padding, locations)
                    normalization = LA.norm(previous_hog_histograms)
                    previous_hog_histograms = np.divide(previous_hog_histograms, normalization)
                    average_histogram = previous_hog_histograms
                    initialized = True
                    continue

                frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

                hog_histograms = hog_descriptior.compute(frame, win_stride, padding, locations)
                normalization = LA.norm(hog_histograms)
                hog_histograms = np.divide(hog_histograms, normalization)

                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
                hog_diff_abs_sum = np.sum(np.abs(hog_histograms - previous_hog_histograms))
                cv2.accumulateWeighted(gray, avg, 0.5)

                previous_hog_histograms = np.divide(previous_hog_histograms + hog_histograms, 2)

                cv2.imshow('Hi', frame)
                cv2.waitKey(1)
                print hog_diff_abs_sum



                # print frame_diff_avg
                # if frame_diff_avg >= self.frame_diff_threshold:
                #     print 'Motion Detected!!'
                #
                # else:
                #     print 'Silence --'



if __name__ == '__main__':
    raspberry = Raspberry()

    while True:
        time.sleep(3)