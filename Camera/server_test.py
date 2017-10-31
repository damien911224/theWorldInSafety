###Server
###Written in python2.7
###Author: ftatp

import socket
import sys

client_buff = "Hello!!!"

def setupServer():

	host = ''
	port = 12344

	server_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	
	try:
		server_fd.bind((host, port))
	except socket.error as msg:
		print(msg)
	
	return server_fd

def setupConnection():
	server_fd.listen(1)
	conn, address = server_fd.accept()
	print "Got connection from", address

	return conn

def dataTransfer(conn):
	#loop for communiction
	conn.send("Hello client, the connection is successful!")

	while True:
		data = conn.recv(1024)
		data = data.decode('utf-8')

		print data.encode('utf-8')

		dataMessage = data.split(' ', 1)
		command = dataMessage[0]

		if command == 'GET':
			reply = GET()
		elif command == 'REPEAT':
			reply = REPEAT(dataMessage)
		elif command == 'EXIT':
			break
		elif command == 'KILL':
			print "The server is killed"
			server_fd.close()
			break
		else:
			reply = 'Unknown command'

		conn.sendall(str.encode(reply))
		print("Data sent!")
		reply = ''
	conn.close()

def GET():
	reply = client_buff
	return reply

def REPEAT(dataMessage):
	reply = dataMessage[1]


if __name__ == "__main__":

	server_fd = setupServer()

	while True:
		try:
			conn = setupConnection()
			dataTransfer(conn)
		except:
			break


#while(True):
#	client_fd, addr = server_fd.accept()
#	print "Got connection from", addr
#	client_fd.send("Hello client, connection is successful!")
#
#	if client_fd.recv(1024) != -1:
#		print "qqq"
#
#	client_fd.close()
#


