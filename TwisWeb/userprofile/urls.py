from django.conf.urls import include, url

from userprofile import views

urlpatterns = [
	url(r'^profile/$', views.user_profile, name='profile'),	
]
