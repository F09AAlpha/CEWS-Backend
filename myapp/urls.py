from django.urls import path
from myapp.Views.financialNewsView import FetchFinancialNewsView, FinancialNewsListView

urlpatterns = [
    path('fetch-financial-news/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('financial-news/', FinancialNewsListView.as_view(), name='financial-news-list'),
]
