from django.contrib import admin

from management.models import Video, Facility

# Register your models here.

class VideoInline(admin.StackedInline):
	model = Video
	extra = 10

class FacilityAdmin(admin.ModelAdmin):
	inlines = [VideoInline]
	list_display = ('Facility_name',)

class VideoAdmin(admin.ModelAdmin):
	list_display = ('Video_name', 'upload_date')


admin.site.register(Video, VideoAdmin)
admin.site.register(Facility, FacilityAdmin)
