from django.shortcuts import render
from django.views.generic import ListView, DetailView

from management.models import Facility
from usersite.models import Uservideo

from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required

@method_decorator(login_required, name='dispatch')
class FacilityLV(ListView):
	template_name = 'facility_list.html'
	model = Facility
#	def get_queryset(self):
#		return .objects.all()[0:10]


@method_decorator(login_required, name='dispatch')
class FacilityDV(DetailView):
	template_name = 'facility_detail.html'
	model = Facility


@method_decorator(login_required, name='dispatch')
class VideoDV(DetailView):
	template_name = 'video_detail.html'
	model = Uservideo


