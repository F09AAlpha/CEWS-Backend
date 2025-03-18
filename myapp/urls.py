from django.urls import path
from myapp.Views.financialNewsView import FetchFinancialNewsView, FinancialNewsListView
from myapp.Views.exchangeRateLatestViews import CurrencyRateView

urlpatterns = [
    path('fetch-financial-news/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('financial-news/', FinancialNewsListView.as_view(), name='financial-news-list'),
    path('v1/currency/rates/<str:base>/<str:target>/', CurrencyRateView.as_view(), name='currency-rate'),
]
