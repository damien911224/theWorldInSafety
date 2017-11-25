from management.models import Facility, Video
from django.forms.models import inlineformset_factory

VideoInlineFormSet = inlineformset_factory(Facility, Video, fields=['Video_snapshot', 'Video_name'], extra=10)
