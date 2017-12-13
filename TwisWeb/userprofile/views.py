from django.shortcuts import render, render_to_response
from django.http import HttpResponseRedirect
from django.template.context_processors import csrf

from userprofile.forms import UserProfileForm

from django.contrib.auth.decorators import login_required

@login_required
def user_profile(request):
	if request.method == 'POST':
		form = UserProfileForm(request.POST, instance=request.user.profile)
		if form.is_valid():
			form.save()
			return HttpResponseRedirect('/')

	else:
		user = request.user
		profile = user.userprofile
		form = UserProfileForm(instance=profile)

	context = {}
	context.update(csrf(request))

	context['form'] = form

	return render(request, 'profile.html', context)
