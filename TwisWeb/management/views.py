from django.shortcuts import render, redirect
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView, FormView
from django.http import HttpResponse, HttpResponseRedirect

from django.core.files.base import ContentFile
from django.core.files import File
from django.core.urlresolvers import reverse_lazy
from django import forms

import cv2
import os

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

#	model.save()

def violence(request, video_id):
	if request.method == 'POST':
		v = Video.objects.get(id=video_id)
		
		uv = Uservideo(id=None,
				Video_name=v.Video_name,
				Video_facility=v.Video_facility,
				Video_snapshot=v.Video_snapshot,
				Video_record_path=v.Video_record_path,
		)
		uv.save()
		#TODO: get users who registered in this facility
		registered_users = User.objects.filter(groups__name=uv.Video_facility)
		for user in registered_users:
			print(user.profile.phone_num)

	return HttpResponse('Sended Alarm\n')

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
	
		#TODO: make a snapshot with the file sent
		print(request.FILES['admin_clip'].name)
		video_cap = cv2.VideoCapture(request.FILES['admin_clip'].name)
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
			v_video_path = v_thumbnail_path.split('.')[0] + ".avi"
			v.Video_record_path = v_video_path
		v.save()

		#save the video in the correct path 
		file_content = ContentFile(request.FILES['admin_clip'].read())
		#fout = open(request.FILES['admin_clip'].name, 'wb+')
		fout = open(os.path.join(settings.MEDIA_ROOT, v_video_path)[1:], 'wb+')
		for chunk in file_content.chunks():
			fout.write(chunk)
		fout.close()

		os.remove(request.FILES['admin_clip'].name)
		os.remove(snapshot_path)

	return HttpResponse('Hello world\n')
