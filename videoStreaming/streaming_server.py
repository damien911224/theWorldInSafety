import sys
import os
import numpy
import cv2
import glob
import socket
import threading
import datetime


class StreamingServer():
    def __init__(self):
        self.server_host_name = 'localhost'
        self.server_port_number = 10001

        self.client_ip_address = '127.0.0.1'
        self.client_port = 41224

        self.server_video_cap = cv2.VideoCapture(0)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.server_host_name, self.server_port_number))

        self.server_socket.listen(5)


        self.server_thread = threading.Thread(target=self.run, name='Streaming Server')
        self.server_thread.start()

    def run(self):
        while True:
            if True:
                while True:
                    try:
                        client_socket, address = self.server_socket.accept()

                        print address

                        try:
                            client_name = client_socket.recv(90456)
                        except Exception as e:
                            print e
                            continue

                        print client_name

                        if client_name == 'Model':
                            try:
                                session_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                client_socket.send(session_name)
                            except socket.error:
                                print 'Socket Error'
                                continue

                            try:
                                r = client_socket.recv(90456)
                            except Exception as e:
                                print e
                                continue

                            if r == 'OK':
                                while True:
                                    ok, frame = self.server_video_cap.read()
                                    if not ok:
                                        break
                                    frame_data = cv2.imencode('.jpg', frame)[1].tostring()
                                    try:
                                        client_socket.send(frame_data)
                                        client_socket.send(b'!TWIS_END!')
                                    except socket.error:
                                        print 'SOCKET ERROR!'
                                        break


                    except Exception as e:
                        print e

if __name__ == '__main__':
    streaming_server = StreamingServer()
