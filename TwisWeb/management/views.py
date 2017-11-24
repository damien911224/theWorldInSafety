from django.shortcuts import render
from django.views.generic import ListView, DetailView

from management.models import Video, Facility

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
#from .forms import UploadFileForm
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import user_passes_test
from django.utils.decorators import method_decorator

@method_decorator(user_passes_test(lambda u: u.is_superuser), name='dispatch')
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
	template_name = 'video_detail.html'
	model = Video

@csrf_exempt
def receive(request):
	if request.method == 'POST':
		print("Received file")
		print("file:", request.FILES)
		
	#context = {}

	return HttpResponse('Hello world\n')
