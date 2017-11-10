import sys
import os
import numpy
import cv2
import glob
import socket
import threading


class StreamingServer():
    def __init__(self):
        self.server_host_name = '192.168.1.101'
        self.server_port_number = 10000

        self.client_ip_address = '10.211.55.10'
        self.client_port = 10001

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.server_host_name, self.server_port_number))

        self.server_video_cap = cv2.VideoCapture(0)

        self.server_thread = threading.Thread(target=self.run, name='Streaming Server')
        self.server_thread.start()

    def run(self):
        while True:
            if True:
                while True:
                    try:
                        self.server_socket.connect((self.client_ip_address, self.client_port))

                        while True:
                            ok, frame = self.server_video_cap.read()
                            if not ok:
                                break
                            frame_data = cv2.imencode('.jpg', frame)[1].tostring()
                            try:
                                self.server_socket.send(frame_data)
                                self.server_socket.send(b'!TWIS_END!')
                            except socket.error:
                                print 'SOCKET ERROR!'
                                break

                    except Exception as e:
                        print e

if __name__ == '__main__':
    streaming_server = StreamingServer()