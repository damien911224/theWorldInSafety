from django.db import models

from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save

class UserProfile(models.Model):
	user = models.OneToOneField(User)
	phone_num = models.CharField(max_length=20)

#User.profile = property(lambda u: UserProfile.objects.get_or_create(user=u)[0])

#@receiver(post_save, sender=User)
#def create_user_profile(sender, instance, created, **kwargs):
#	if created:
#		UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def update_user_profile(sender, instance, created, **kwargs):
	if created:
		UserProfile.objects.create(user=instance)
	instance.userprofile.save()
