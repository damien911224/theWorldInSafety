import socket
import cv2
import numpy
import random
import sys
import os
import threading


class StreamingClient():
    def __init__(self):
        self.server_ip_address = '10.211.55.10'
        self.server_port = 10000

        self.client_host_name = '192.168.1.101'
        self.client_port = 10001

        self.save_folder = os.path.join('/home/damien/temp/streaming')
        if not os.path.exists(self.save_folder):
            try:
                os.makedirs(self.save_folder)
            except OSError:
                pass

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.bind((self.client_host_name, self.client_port))
        self.client_socket.listen(5)

        self.frame_index = 1

        self.client_thread = threading.Thread(target=self.run, name='Streaming Client')
        self.client_thread.start()



    def run(self):
        while True:
            try:
                server_socket, address = self.client_socket.accept()

                while True:
                    data = b''
                    while True:
                        try:
                            r = server_socket.recv(90456)
                            if len(r) == 0:
                                exit(0)
                            a = r.find(b'!TWIS_END!')
                            if a != -1:
                                data += r[:a]
                                break
                            data += r
                        except Exception as e:
                            print(e)
                            continue
                    np_arr = numpy.fromstring(data, numpy.uint8)
                    if np_arr is not None:
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            frame_path = os.path.join(self.save_folder, 'img_{:07d}.jpg'.format(self.frame_index))
                            self.frame_index += 1
                            # cv2.imwrite(frame_path, frame)

                            cv2.imshow('frame', frame)
                            cv2.waitKey(int(1000.0/25.0))

            except socket.timeout:
                print 'socket timeout'
                continue

if __name__ == '__main__':
    streaming_client = StreamingClient()