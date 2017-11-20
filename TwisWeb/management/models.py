
from __future__ import unicode_literals
from django.db import models
from django.core.urlresolvers import reverse

from management.fields import VideoField

#from datetime import datetime

class Facility(models.Model):
	#Facility_id = models.IntegerField(default=0)
	Facility_name = models.CharField(max_length=256)
	Facility_homepage = models.CharField(max_length=256)

	class Meta:
		ordering = ['Facility_name']

	def __str__(self):
		return self.name

	def get_absolute_url(self):
		return reverse('management:Facility_detail', args=(self.id,))


class Video(models.Model):
	#Video_id = models.IntegerField(default=0)
	Video_name = models.CharField(max_length=128)

	Video_facility = models.ForeignKey(Facility)
	Video_snapshot = VideoField(upload_to='video/%Y/%m')
	upload_date = models.DateTimeField(auto_now_add=True)

	Video_checked = models.BooleanField(default=False)
#substitude to register library
#class registed_user(models.Model):
#	User_id = models.IntegerField(default=0)
#
#	Registered_facility = models.IntegerField(default=0)

	class Meta:
		ordering = ['id']

	def __str__(self):
		return self.title

	def get_absolute_url(self):
		return reverse('management:Video_detail', args=(self.id,))

