from django.contrib import admin

from usersite.models import Uservideo
from management.admin import VideoAdmin

# Register your models here.

admin.site.register(Uservideo, VideoAdmin)
