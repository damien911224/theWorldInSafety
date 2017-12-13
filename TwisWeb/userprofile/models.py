from django.db import models

from django.contrib.auth.models import User, Group
from django.dispatch import receiver
from django.db.models.signals import post_save

class UserProfile(models.Model):
	user = models.OneToOneField(User)
	phone_num = models.CharField(max_length=20)
	#group
#User.profile = property(lambda u: UserProfile.objects.get_or_create(user=u)[0])

#@receiver(post_save, sender=User)
#def create_user_profile(sender, instance, created, **kwargs):
#	if created:
#		UserProfile.objects.create(user=instance)

def add_to_default_group(sender, **kwargs):
	user = kwargs["instance"]
	if kwargs["created"]:
		group = Group.objects.get(name='f1')
		user.groups.add(group)

post_save.connect(add_to_default_group, sender=User)

@receiver(post_save, sender=User)
def update_user_profile(sender, instance, created, **kwargs):
	if created:
		UserProfile.objects.create(user=instance)
#		default_group = Group.objects.get(name='f1')
#		default_group.user_set.add(sender)
	instance.userprofile.save()


