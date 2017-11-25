
from __future__ import unicode_literals
from django.db import models
from django.contrib.auth.models import Group

from django.core.urlresolvers import reverse

from management.models import Facility
from management.fields import VideoField

#from datetime import datetime

class Uservideo(models.Model):
	#Video_id = models.IntegerField(default=0)
	Video_name = models.CharField(max_length=128)
	Video_facility = models.ForeignKey(Facility)
	Video_snapshot = VideoField(upload_to='video/%Y/%m')
	Video_record_path = models.CharField(max_length=128)
	upload_date = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['id']

	def __str__(self):
		return self.Video_name

	def get_user_absolute_url(self):
		return reverse('usersite:Video_detail', args=(self.id,))

