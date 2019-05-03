from django.urls import path
from . import views

urlpatterns = (
    path("", views.getModels),
    path("<int:pk>", views.getModelDetail)
)
