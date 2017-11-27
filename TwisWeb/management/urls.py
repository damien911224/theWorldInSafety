from django.conf.urls import url
from management import views

urlpatterns = [
	url(r'^$', views.overall.as_view(), name='overall'), #video list
	url(r'^facility/$', views.FacilityLV.as_view(), name='Facility_list'),
	url(r'^facility/(?P<pk>\d+)/$', views.FacilityDV.as_view(), name='Facilty_detail'),
	url(r'^video/(?P<pk>\d+)/$', views.VideoDV.as_view(), name='Video_detail'),
]
