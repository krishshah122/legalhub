from django.urls import path
from . import views

urlpatterns = [
    path('filing/', views.filing, name='filing'),
]