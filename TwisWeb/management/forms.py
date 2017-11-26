from management.models import Facility, Video
from django.forms.models import inlineformset_factory
from django import forms

VideoInlineFormSet = inlineformset_factory(Facility, Video, fields=['Video_snapshot', 'Video_name'], extra=10)


class ObjectForm(forms.Form):

	CHOICES = (
		(1, 'yes'),
		(0, 'no')
	)
	checked = forms.ChoiceField(label="label", widget=forms.RadioSelect, choices=CHOICES)

