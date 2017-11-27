from django import forms

from userprofile.models import UserProfile

class UserProfileForm(forms.ModelForm):

	class Meta:
		model = UserProfile
		fields = ('phone_num',)
