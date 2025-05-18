from django.shortcuts import render

def index(request):
    return render(request, 'index.html')
from django.shortcuts import render, redirect
from .forms import FilingForm
from django.core.mail import EmailMessage
from django.conf import settings

def filing(request):
    if request.method == 'POST':
        form = FilingForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.cleaned_data['document']
            recipient = form.cleaned_data['recipient_email']
            name = form.cleaned_data['recipient_name']
            msg = form.cleaned_data['message']

            # Save uploaded file (optional)
            with open(f'main/submissions/{doc.name}', 'wb+') as dest:
                for chunk in doc.chunks():
                    dest.write(chunk)

            # Send email (example submission)
            email = EmailMessage(
                subject='Legal Document Submission',
                body=f'Dear {name},\n\nYou have received a legal document.\n\nMessage:\n{msg}',
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[recipient],
            )
            email.attach(doc.name, doc.read(), doc.content_type)
            email.send()

            return render(request, 'filing_success.html', {'name': name})
    else:
        form = FilingForm()

    return render(request, 'filing.html', {'form': form})
