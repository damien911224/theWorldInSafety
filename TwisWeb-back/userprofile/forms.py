from django import forms

from userprofile.models import UserProfile
#from django.contrib.auth.models import User
#
#class UserForm(forms.ModelForm):
#
#	class Meta:
#		model = User
#		fields = ()

class UserProfileForm(forms.ModelForm):

	class Meta:
		model = UserProfile
		fields = ('phone_num',)
