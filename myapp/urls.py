from django.urls import path
from myapp.Views.financialNewsView import FetchFinancialNewsView, FinancialNewsListView
from myapp.Views.currencyNewsView import CurrencyNewsListView, FetchCurrencyNewsView

urlpatterns = [
    path('fetch-financial-news/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('financial-news/', FinancialNewsListView.as_view(), name='financial-news-list'),
    path('currency/events/', CurrencyNewsListView.as_view(), name='currency-news-list'),
    path('currency/<str:currency_code>/', FetchCurrencyNewsView.as_view(), name='fetch-currency-news') 
]
