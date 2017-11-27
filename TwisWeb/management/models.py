
from __future__ import unicode_literals
from django.db import models
from django.contrib.auth.models import Group

from django.core.urlresolvers import reverse

from management.fields import VideoField

from django.contrib.auth.models import User

class ExtendedUser(models.Model):
	user = models.OneToOneField(User)
	phone_num = models.CharField(max_length=20)

class Facility(models.Model):
	#Facility_id = models.IntegerField(default=0)

	group = models.OneToOneField(Group)

	Facility_name = models.CharField(max_length=256)
	Facility_homepage = models.CharField(max_length=256)

	class Meta:
		ordering = ['Facility_name']

	def __str__(self):
		return self.Facility_name

	def get_absolute_url(self):
		return reverse('management:Facility_detail', args=(self.id,))

	def get_user_absolute_url(self):
		return reverse('usersite:Facility_detail', args=(self.id,))


class Video(models.Model):
	#Video_id = models.IntegerField(default=0)
	Video_name = models.CharField(max_length=128)
	Video_facility = models.ForeignKey(Facility)
	Video_snapshot = VideoField(upload_to='video/%Y/%m')
	Video_record_path = models.CharField(max_length=128)
	upload_date = models.DateTimeField(auto_now_add=True)
	Video_checked = models.BooleanField(default=False)

	class Meta:
		ordering = ['id']

	def __str__(self):
		return self.Video_name

	def get_absolute_url(self):
		return reverse('management:Video_detail', args=(self.id,))

	def get_user_absolute_url(self):
		return reverse('usersite:Video_detail', args=(self.id,))

