from django.forms import ModelForm, ClearableFileInput
from .models import UpLoadTestModel


class CustomClearableFileInput(ClearableFileInput):
    template_with_clear = '<br>  <label for="%(clear_checkbox_id)s">%(clear_checkbox_label)s</label> %(clear)s'


class FormEntrada(ModelForm):
    class Meta:
        model = UpLoadTestModel
        fields = ('titulo', 'texto', 'archivo')
        widgets = {
            'archivo': CustomClearableFileInput
        }

from django import forms

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
