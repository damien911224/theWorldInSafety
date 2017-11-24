import pycurl

admin_clip_send_path = "/home/ftatp/Pictures/camera.jpg"

curl = pycurl.Curl()
curl.setOpt(curl.POST, 1)
curl.setOpt(curl.URL, 'http://127.0.0.1:8000/')
curl.setOpt(curl.HTTPPOST, [('admin_clip', (curl.FROM_FILE, admin_clip_send_path))])

	#('user_clip', (curl.FROM_FILE, user_clip_send_path))])

curl.performe()
curl.close()

