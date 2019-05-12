import sys
import cv2
import numpy as np
import os, glob
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ( socket.gethostname(), 10000 )
print server_address[0]
server_ip = ''
sock.bind(server_address)

sock.listen(1)

while True:
    print 'I am waiting!!'

    connection, client_address = sock.accept()

    print 'connection from {}'.format(client_address)

    while True:
        data = sock.recv(1024)
        print '===============DATA==============='
        print '{}'.format(data)
        print '=================================='

        if not data:
            break

    connection.close()
