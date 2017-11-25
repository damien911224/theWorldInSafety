import cv2
import socket
import threading
import datetime
import os
import numpy as np
import time
import glob
from multiprocessing import Pool, Value, Lock, current_process, Manager
from shutil import rmtree, copyfile


class StreamingServer():

    def __init__(self):
        self.streaming_server_host_name = '172.31.0.152'
        # self.streaming_server_host_name = 'localhost'

        self.home_folder = os.path.abspath('../..')
        self.save_folder = os.path.join(self.home_folder, 'streaming_data')
        if not os.path.exists(self.save_folder):
            try:
                os.makedirs(self.save_folder)
            except OSError:
                pass

        self.print_lock = Lock()

        self.raspberry_port_number = 7777
        self.model_port_number = 8888
        self.controller_port_number = 9999

        self.raspberry = self.Raspberry(self)
        self.model = self.Model(self)
        self.controller = self.Controller(self)

        self.raspberry_thread = threading.Thread(target=self.raspberry.run, name='Raspberry')
        self.model_thread = threading.Thread(target=self.model.run, name='Model')
        self.controller_thread = threading.Thread(target=self.controller.run, name='Controller')

        self.streaming_server_thread = threading.Thread(target=self.run, name='Streaming Server')
        self.streaming_server_thread.start()


    def run(self):
        self.raspberry_thread.start()
        # self.model_thread.start()
        self.controller_thread.start()

        while True:
            time.sleep(0.7)


    class Raspberry():
        def __init__(self, streaming_server):
            self.streaming_server = streaming_server

            self.in_progress = True


        def run(self):
            while True:
                while not self.in_progress:
                    time.sleep(0.3)

                self.raspberry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.raspberry_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.raspberry_socket.bind((self.streaming_server.streaming_server_host_name,
                                            self.streaming_server.raspberry_port_number))

                self.raspberry_socket.listen(5)

                while self.in_progress:
                    try:
                        client_socket, address = self.raspberry_socket.accept()
                        self.session_is_opened = False

                        while self.in_progress:
                            socket_closed = False
                            start_found = False
                            frame_data = b''
                            try:
                                while self.in_progress:
                                    r = client_socket.recv(90456)
                                    if len(r) == 0:
                                        socket_closed = True
                                        break

                                    if not start_found:
                                        a = r.find(b'raspberry')
                                        if a != -1:
                                            r = r[a+9:]
                                            a = r.find(b'!TWIS_END!')
                                            if a != -1:
                                                frame_data += r[:a]
                                                break
                                            else:
                                                frame_data += r
                                            start_found = True
                                    else:
                                        a = r.find(b'!TWIS_END!')
                                        if a != -1:
                                            frame_data += r[:a]
                                            break
                                        else:
                                            frame_data += r

                            except Exception as e:
                                print(e)
                                continue

                            if socket_closed or not self.in_progress:
                                client_socket.close()
                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}|{}'.format('Raspberry', 'Session Closed', self.session_name)
                                break
                            else:
                                header = frame_data[:22]
                                session_name = str(header[:15])
                                frame_index = int(header[15:22])
                                frame_data = r[23:]

                                if not self.session_is_opened:
                                    self.session_name = session_name
                                    self.session_folder = os.path.join(self.streaming_server.save_folder,
                                                                       session_name)
                                    try:
                                        os.mkdir(self.session_folder)
                                    except OSError:
                                        pass

                                    with self.streaming_server.print_lock:
                                        print '{:10s}|{:13s}|{}'.format('Raspberry', 'Session Start', self.session_name)

                                    self.session_is_opened = True


                                np_arr = np.fromstring(frame_data, np.uint8)
                                if np_arr is not None:
                                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                                    if frame is not None:
                                        self.dumpFrames([frame], frame_index)

                        client_socket.close()

                    except socket.timeout:
                        print 'socket timeout'
                        continue

                self.raspberry_socket.close()


        def dumpFrames(self, frames, start_index):
            frame_index =   start_index
            for frame in frames:
                frame_path = os.path.join(self.session_folder, '{:07d}.jpg'.format(frame_index))
                cv2.imwrite(frame_path, frame)
                frame_index += 1


    class Model():
        def __init__(self, streaming_server):
            self.streaming_server = streaming_server


        def run(self):

            print ''


    class Controller():
        def __init__(self, streaming_server):
            self.streaming_server = streaming_server

            self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.controller_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.controller_socket.bind((self.streaming_server.streaming_server_host_name,
                                     self.streaming_server.controller_port_number))

            self.controller_socket.listen(5)


        def run(self):
            while True:
                try:
                    client_socket, address = self.controller_socket.accept()

                    while True:
                        socket_closed = False
                        start_found = False
                        message_data = b''
                        try:
                            while True:
                                r = client_socket.recv(90456)
                                if len(r) == 0:
                                    socket_closed = True
                                    break

                                if not start_found:
                                    a = r.find(b'raspberry')
                                    if a != -1:
                                         r = r[a+9:]
                                         a = r.find(b'!TWIS_END!')
                                         if a != -1:
                                             message_data += r[:a]
                                             break
                                         else:
                                             message_data += r
                                    start_found = True
                                else:
                                    a = r.find(b'!TWIS_END!')
                                    if a != -1:
                                        message_data += r[:a]
                                        break
                                    else:
                                        message_data += r

                        except Exception as e:
                            print(e)
                            continue

                        if socket_closed:
                            client_socket.close()
                            break
                        else:
                            print str(message_data)
                            if str(message_data) == 'stop':
                                self.streaming_server.raspberry.in_progress = False
                                client_socket.close()
                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}'.format('Controller',  str(message_data))
                                break
                            elif str(message_data) == 'resume' or str(message_data) == 'start':
                                self.streaming_server.raspberry.in_progress = True
                                client_socket.close()
                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}'.format('Controller', str(message_data))
                                break


                except socket.timeout:
                    print 'socket timeout'
                    continue


    def run_model(self):
        while True:
            if True:
                while True:
                    try:
                        client_socket, address = self.model_socket.accept()

                        print address

                        try:
                            self.server_video_cap = cv2.VideoCapture(self.video_path)
                            #dir list sort and get name
                            # session_name = #get dir name

                            #client_socket.send()
                        except socket.error:
                            print 'Socket Error'
                            continue


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

    while True:
        time.sleep(10.3214)