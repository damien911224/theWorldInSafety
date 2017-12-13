import cv2
import socket
import threading
import datetime
import os
import numpy as np
import time
import glob
from multiprocessing import Lock
from shutil import rmtree


class StreamingServer():

    def __init__(self):
        self.print_lock = Lock()
        self.streaming_server_host_name = self.getAWSIpAddress()

        self.home_folder = os.path.abspath('../..')
        self.save_folder = os.path.join(self.home_folder, 'streaming_data')
        rmtree(self.save_folder, ignore_errors=True)

        if not os.path.exists(self.save_folder):
            try:
                os.makedirs(self.save_folder)
            except OSError:
                pass

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
        self.model_thread.start()
        self.controller_thread.start()

        while True:
            time.sleep(33.21324)


    def getAWSIpAddress(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))

        raspberry_ip_address = sock.getsockname()[0]
        sock.close()

        with self.print_lock:
            print '{:10s}|{:12s}|{}'.format('AWS', 'Connection', 'With IP {}'.format(raspberry_ip_address))

        return raspberry_ip_address


    class Raspberry():

        def __init__(self, streaming_server):
            self.streaming_server = streaming_server

            self.in_progress = True
            self.session_is_opened = False

            self.dumped_index = 0


        def run(self):
            while True:
                while not self.in_progress:
                    time.sleep(0.3)

                self.raspberry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.raspberry_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.raspberry_socket.bind((self.streaming_server.streaming_server_host_name,
                                            self.streaming_server.raspberry_port_number))

                self.raspberry_socket.listen(5)
                self.dumped_index = 0

                while self.in_progress:
                    try:
                        if not self.in_progress:
                            break

                        self.ready = True
                        client_socket, address = self.raspberry_socket.accept()

                        previous_data = b''
                        while self.in_progress:
                            socket_closed = False
                            accumulated_data = previous_data + b''
                            try:
                                while self.in_progress:
                                    recv_data = client_socket.recv(90456)
                                    if len(recv_data) == 0:
                                        socket_closed = True
                                        break

                                    accumulated_data += recv_data
                                    found = accumulated_data.find(b'!TWIS_END!')
                                    if found != -1:
                                        previous_data = accumulated_data[found + 10:]
                                        accumulated_data = accumulated_data[:found]
                                        break
                            except Exception:
                                continue

                            frame_data = accumulated_data

                            if socket_closed or not self.in_progress:
                                client_socket.close()
                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}|{}'.format('Raspberry', 'Session Closed', self.session_name)
                                break
                            else:
                                header = frame_data[:22]
                                session_name = str(header[:15])
                                frame_index = int(header[15:22])
                                frame_data = frame_data[22:]

                                if not self.session_is_opened:
                                    self.session_name = session_name
                                    self.session_delay = 0.0
                                    self.delay_count = 0
                                    self.session_folder = os.path.join(self.streaming_server.save_folder,
                                                                       session_name)
                                    try:
                                        os.mkdir(self.session_folder)
                                    except OSError:
                                        pass

                                    with self.streaming_server.print_lock:
                                        print '{:10s}|{:15s}|{}'.format('Raspberry', 'Session Start', self.session_name)

                                    self.session_is_opened = True


                                np_arr = np.fromstring(frame_data, np.uint8)
                                if np_arr is not None:
                                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                                    if frame is not None:
                                        self.dumpFrames([frame], self.dumped_index + 1)
                                        self.dumped_index += 1

                        client_socket.close()

                    except socket.timeout:
                        continue

                    except KeyboardInterrupt:
                        self.raspberry_socket.close()
                        self.session_is_opened = False
                        break

                self.raspberry_socket.close()
                self.session_is_opened = False


        def dumpFrames(self, frames, start_index):
            frame_index = start_index
            for frame in frames:
                frame_path = os.path.join(self.session_folder, 'img_{:07d}.jpg'.format(frame_index))
                cv2.imwrite(frame_path, frame)
                frame_index += 1


    class Model():

        def __init__(self, streaming_server):
            self.streaming_server = streaming_server
            self.in_progress = True

            self.jpg_boundary = b'!TWIS_END!'
            self.session_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.sending_round = 1
            self.ready = False
            self.session_is_open = False

            self.removing_term = 100
            self.sent_index = -1


        def run(self):
                self.model_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.model_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.model_socket.bind((self.streaming_server.streaming_server_host_name,
                                            self.streaming_server.model_port_number))

                self.model_socket.listen(5)

                while self.in_progress:
                    self.client_socket, address = self.model_socket.accept()

                    while not self.streaming_server.raspberry.session_is_opened and self.in_progress:
                        time.sleep(0.3)


                    self.session_name = self.streaming_server.raspberry.session_name
                    self.session_folder = os.path.join(self.streaming_server.save_folder, self.session_name)
                    self.session_is_open = True
                    self.start_index = 1

                    with self.streaming_server.print_lock:
                        print '{:10s}|{:15s}|{}'.format('Model', 'Session Start', self.session_name)

                    while self.in_progress:
                        while self.sent_index >= self.streaming_server.raspberry.dumped_index and self.in_progress:
                            time.sleep(0.1)

                        self.end_index = self.streaming_server.raspberry.dumped_index

                        for frame_index in range(self.start_index, self.end_index + 1, 1):
                            frame_path = os.path.join(self.session_folder, 'img_{:07d}.jpg'.format(frame_index))
                            frame = cv2.imread(frame_path)
                            self.send(frame, frame_index)

                            removing_frame_index = frame_index - self.removing_term
                            if removing_frame_index >= 1:
                                removing_frame_path = os.path.join(self.session_folder, 'img_{:07d}.jpg'.format(removing_frame_index))
                                try:
                                    os.remove(removing_frame_path)
                                except OSError:
                                    pass

                        self.sent_index = self.end_index
                        self.start_index = self.end_index + 1

                    self.client_socket.close()
                    self.session_is_open = False
                    self.ready = True
                    rmtree(self.session_folder, ignore_errors=True)

                    with self.streaming_server.print_lock:
                        print '{:10s}|{:15s}|{}'.format('Model', 'Session Closed', self.session_name)


        def send(self, frame, frame_index):
            header = b'{:15s}{:07d}'.format(self.session_name, frame_index)
            try:
                frame_data = cv2.imencode('.jpg', frame)[1].tostring()
                send_data = header + frame_data + self.jpg_boundary

                for _ in range(self.sending_round):
                    try:
                        self.client_socket.send(send_data)
                    except socket.error:
                        return False
                        pass
            except:
                pass



            return True


        def sendMessage(self, message):
            send_message = b'{}{}'.format(self.session_name, message, b'!TWIS_END!')
            try:
                self.client_socket.send(send_message)
            except socket.error:
                pass


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

                    previous_data = b''
                    while True:
                        socket_closed = False
                        start_found = False
                        message_data = previous_data + b''
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
                                             previous_data = r[a+10:]
                                             break
                                         else:
                                             message_data += r
                                    start_found = True
                                else:
                                    a = r.find(b'!TWIS_END!')
                                    if a != -1:
                                        message_data += r[:a]
                                        previous_data = r[a+10:]
                                        break
                                    else:
                                        message_data += r

                        except Exception as e:
                            continue

                        if socket_closed:
                            client_socket.close()
                            break
                        else:
                            message_data = str(message_data).replace(' ', '')
                            if message_data == 'stop':
                                self.streaming_server.raspberry.in_progress = False
                                client_socket.close()
                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}'.format('Controller',  message_data.upper())
                                break
                            elif message_data == 'resume' or message_data == 'start':
                                self.streaming_server.raspberry.ready = False
                                self.streaming_server.raspberry.in_progress = True
                                while not self.streaming_server.raspberry.ready:
                                    time.sleep(0.3)

                                client_socket.close()
                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}'.format('Controller', message_data.upper())
                                break
                            elif message_data == 'reset':
                                if self.streaming_server.model.session_is_open:
                                    self.streaming_server.model.ready = False
                                    self.streaming_server.model.in_progress = False
                                    while not self.streaming_server.model.ready:
                                        time.sleep(0.2)
                                session_folders = glob.glob(os.path.join(self.streaming_server.save_folder, '*'))
                                for session_folder in session_folders:
                                    rmtree(session_folder, ignore_errors=True)

                                self.streaming_server.model.in_progress = True

                                client_socket.close()

                                with self.streaming_server.print_lock:
                                    print '{:10s}|{:15s}'.format('Controller', message_data.upper())

                                break


                except socket.timeout:
                    continue

                except KeyboardInterrupt:
                    self.controller_socket.close()
                    break



if __name__ == '__main__':
    streaming_server = StreamingServer()

    while True:
        time.sleep(10.3214)
