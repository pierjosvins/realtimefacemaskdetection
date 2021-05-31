from django.urls import path, include
from streamapp import views


urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('mask_feed', views.mask_feed, name='mask_feed'),
    ]
