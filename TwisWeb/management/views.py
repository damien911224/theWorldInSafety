from django.shortcuts import render
from django.views.generic import ListView, DetailView

from management.models import Video, Facility

class overall(ListView):
	template_name = 'overall.html'
	model = Video

	def get_queryset(self):
		return Video.objects.all()[0:10]

class FacilityLV(ListView):
	template_name = 'facility_list.html'
	model = Facility

class FacilityDV(DetailView):
	template_name = 'facility_detail.html'
	model = Facility

class VideoDV(DetailView):
	model = Video
