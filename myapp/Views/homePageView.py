from django.http import HttpResponse


def home(request):
    return HttpResponse("Hello! You have reached CEWS.")
