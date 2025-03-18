from django.urls import path
from myapp.Views import homePageView

urlpatterns = [
    path('', homePageView.home, name="home-page"),
]
