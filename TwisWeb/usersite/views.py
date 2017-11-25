from django.shortcuts import render
from django.views.generic import ListView, DetailView

from management.models import Video, Facility

from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required

@method_decorator(login_required, name='dispatch')
class FacilityLV(ListView):
	template_name = 'facility_list.html'
	model = Facility
#	def get_queryset(self):
#		return .objects.all()[0:10]


class FacilityDV(ListView):
	template_name = 'facility_detail.html'
	model = Facility
	def get_queryset(self):
		return Video.objects.all()#get(Video_checked=True)


class VideoDV(DetailView):
	template_name = 'video_detail.html'
	model = Video


