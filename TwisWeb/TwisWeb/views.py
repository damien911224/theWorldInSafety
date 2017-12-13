from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView
from django.contrib.auth.forms import UserCreationForm
from django.core.urlresolvers import reverse_lazy

from django.contrib.auth import login, authenticate
from django.shortcuts import redirect, render

from django import forms
from django.contrib.auth.models import User, Group

class UserCreateForm(UserCreationForm):
	phone_num = forms.CharField(required=True)

	class Meta:
		model = User
		fields = ("username", "phone_num", "password1", "password2")
	
	def save(self, commit=True):
		user=super(UserCreateForm, self).save(commit=False)
		user.phone_num = self.cleaned_data["phone_num"]
		if commit:
			user.save()
#			default_group = Group.objects.get(name='f1')
#			default_group.user_set.add(sender)

		return user


class HomeView(TemplateView):
	template_name ='home.html'

class UserCreateView(CreateView):
	template_name = 'registration/register.html'
	form_class = UserCreateForm
	success_url = reverse_lazy('register_done')

def createUser(request):
	if request.method == 'POST':
		form = UserCreateForm(request.POST)
		if form.is_valid():
			user = form.save()
			user.refresh_from_db()
			user.userprofile.phone_num = form.cleaned_data.get('phone_num')
			user.save()
			raw_password = form.cleaned_data.get('password1')
			user = authenticate(username=user.username, password=raw_password)
			login(request, user)
			return redirect('home')
	else:
		form = UserCreateForm()
	return render(request, 'registration/register.html', {'form': form})



class UserCreateDone(TemplateView):
	template_name = 'registration/register_done.html'
