###Client
###Written in python2.7
###Author: ftatp

import socket
import sys

client_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

if len(sys.argv) != 2:
	host = '192.168.243.128'
else:
	host = sys.argv[1]
port = 12344

client_fd.connect((host, port))
print client_fd.recv(1024), '\n'

while True:

	command = input("Enter your command: ")
	client_fd.send(str.encode(command))
	
	if command == 'EXIT':
		break
	elif command == 'KILL':
		print "Killing the server"
		break

	reply = client_fd.recv(1024)
	print reply.decode('utf-8')



####while(True:)
####	split video into 180 frames
####	send the splited video to server
####	client_fd.send("sending a file to the server")

client_fd.close
