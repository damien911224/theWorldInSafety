from django.shortcuts import render, redirect
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView
from django.http import HttpResponse, HttpResponseRedirect

from django.core.files.base import ContentFile
from django.core.files import File
from django.core.urlresolvers import reverse_lazy

import cv2
import os

from TwisWeb import settings
from management.models import Video, Facility
from management.forms import VideoInlineFormSet

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


@csrf_exempt
def receive(request):
	if request.method == 'POST':
		
		#TODO: get the file information
		facility_id = 1
		facility = Facility.objects.get(id=facility_id)
		
		#TODO: make a snapshot with the file sent
		with open('test.jpg', 'rb') as f:
			snapshot = File(f)
			
		#save the snapshot in database
			v = Video(id=None,
					Video_name=request.FILES['file'].name,
					Video_facility=facility,
					Video_checked=False)
			v.Video_snapshot.save('test.jpg', f)
			v.save()

		#get the path of the snapshot
		v_thumbnail_path = v.Video_snapshot.url
		#modify the path for the video
		v_video_path = v_thumbnail_path.split('.')[0] + ".mp4"
		v.Video_record_path = v_video_path
		v.save()

		#save the video in the modified path 
		file_content = ContentFile(request.FILES['file'].read())

		fout = open(os.path.join(settings.MEDIA_ROOT, v_video_path)[1:], 'wb+')
		for chunk in file_content.chunks():
			fout.write(chunk)
		fout.close()
	
	return HttpResponse('Hello world\n')
