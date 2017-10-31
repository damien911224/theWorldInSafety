from django import forms

class PostSearchForm(forms.Form):
	search_word = forms.CharField(label='Search Word')
