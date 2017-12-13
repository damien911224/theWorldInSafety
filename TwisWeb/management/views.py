from django.shortcuts import render, redirect
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView, FormView
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse

from django.core.files.base import ContentFile
from django.core.files import File
from django.core.urlresolvers import reverse_lazy
from django import forms
from django.contrib import messages

import cv2
import os
import base64
from http import client
import json

from TwisWeb import settings
from management.models import Video, Facility, ExtendedUser
from management.forms import VideoInlineFormSet
from usersite.models import Uservideo

from django.contrib.auth.models import User

from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import user_passes_test
from django.utils.decorators import method_decorator


@method_decorator(user_passes_test(lambda u: u.is_superuser), name='dispatch')
class overall(ListView):
	template_name = 'overall.html'
	model = Video
	def get_queryset(self):
		return Video.objects.all()[0:10]

@method_decorator(user_passes_test(lambda u: u.is_superuser), name='dispatch')
class FacilityLV(ListView):
	template_name = 'admin_facility_list.html'
	model = Facility
	def get_queryset(self):
		return Facility.objects.all()


@method_decorator(user_passes_test(lambda u: u.is_superuser), name='dispatch')
class FacilityDV(DetailView):
	template_name = 'admin_facility_detail.html'
	model = Facility

@method_decorator(user_passes_test(lambda u: u.is_superuser), name='dispatch')
class VideoDV(DetailView):
	template_name = 'admin_video_detail.html'
	model = Video
	def get_object(self, queryset=None):
		obj = super(VideoDV, self).get_object(queryset=queryset)
		obj.Video_checked = True
		obj.save()

		return obj

	def post(self, request, pk):
		#if request.method == 'POST':
		v = Video.objects.get(id=pk)
	
		uv = Uservideo(id=None,
				Video_name=v.Video_name,
				Video_facility=v.Video_facility,
				Video_snapshot=v.Video_snapshot,
				Video_record_path=v.Video_record_path,
		)
		uv.save()
	
		send_to_registed_users(uv)
		t = reverse('management:Video_detail', args=(pk,))
		return HttpResponseRedirect(t)

def send_to_registed_users(uv):
	appid = 'twis'
	apikey = 'cb4eb398d21a11e794990cc47a1fcfae'
	address = 'api.bluehouselab.com'

	sender = '01091261777'
	registered_users = User.objects.filter(groups__name=uv.Video_facility)

	receivers = []
	for user in registered_users:
		receivers.append(user.userprofile.phone_num)

	content = "http://13.228.101.253:8080" + uv.get_user_absolute_url()

	print(content)
	credential = "Basic "+base64.encodestring((appid+':'+apikey).encode()).decode().strip()
	headers = {
		"Content-type": "application/json;charset=utf-8",
		"Authorization": credential,
	}

	c = client.HTTPSConnection(address)
	
	path = "/smscenter/v1.0/sendsms"
	value = {
		'sender': sender,
		'receivers': receivers,
		'content': content,
	}
	data = json.dumps(value, ensure_ascii=False).encode('utf-8')

	c.request("POST", path, data, headers)
	r = c.getresponse()

	print(r.status, r.reason)
	print(r.read())

	return



@csrf_exempt
def receive(request):
	if request.method == 'POST':
		v = Video(id=None,
					Video_name=request.FILES['admin_clip'].name,
					#Video_facility=facility,
					Video_checked=False)
	
		#get the facility information
		facility_id = 1
		v.Video_facility = Facility.objects.get(id=facility_id)
		#facility = Facility.objects.get(id=facility_id)
		

		#save the video in the temp path 
		file_content = ContentFile(request.FILES['admin_clip'].read())
		fout = open(request.FILES['admin_clip'].name, 'wb+')
		#fout = open(os.path.join(settings.MEDIA_ROOT, v_video_path)[1:], 'wb+')
		for chunk in file_content.chunks():
			fout.write(chunk)
		fout.close()
	
		print(request.FILES['admin_clip'].name)
		current_folder_path = os.path.abspath('.')
		print(os.path.join(current_folder_path, request.FILES['admin_clip'].name))
		video_cap = cv2.VideoCapture(os.path.join(current_folder_path, request.FILES['admin_clip'].name))
		if video_cap.isOpened():
			#print("asdfasdf")
			ok, frame = video_cap.read()
			
		video_cap.release()

		snapshot_path = request.FILES['admin_clip'].name.split('.')[0] + ".jpg"
		cv2.imwrite(snapshot_path, frame)

		with open(snapshot_path, 'rb') as f:
			snapshot = File(f)
			#save the snapshot in database
			v.Video_snapshot.save(snapshot_path, f)	
			#get the path of the snapshot
			v_thumbnail_path = v.Video_snapshot.url
			#modify the path for the video
			v_video_path = v_thumbnail_path.split('.')[0] + ".mp4"
			v.Video_record_path = v_video_path
		v.save()

		os.rename(request.FILES['admin_clip'].name, v_video_path[1:])
		os.remove(snapshot_path)

	return HttpResponse('Hello world\n')
