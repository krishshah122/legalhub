from django import forms

class FilingForm(forms.Form):
    document = forms.FileField(label="Upload Legal Document")
    recipient_name = forms.CharField(max_length=100)
    recipient_email = forms.EmailField()
    destination_type = forms.ChoiceField(choices=[
        ('person', 'Individual'),
        ('company', 'Company'),
        ('government', 'Government Department')
    ])
    message = forms.CharField(widget=forms.Textarea, required=False)
