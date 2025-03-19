from django.urls import path
from myapp.Views.financialNewsView import FetchFinancialNewsView, FinancialNewsListView
from myapp.Views.currencyNewsView import CurrencyNewsListView, FetchCurrencyNewsView
from myapp.Views.exchangeRateLatestViews import CurrencyRateView
from myapp.Views.historicalExchangeRatesView import FetchHistoricalCurrencyExchangeRates


urlpatterns = [
    path('fetch-financial-news/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('financial-news/', FinancialNewsListView.as_view(), name='financial-news-list'),
    path('currency/<str:currency_code>/', FetchCurrencyNewsView.as_view(), name='fetch-currency-news'),
    path('currency-news/', CurrencyNewsListView.as_view(), name='currency-news-list'),
    path('v1/currency/rates/<str:base>/<str:target>/', CurrencyRateView.as_view(), name='currency-rate'),
    path('v1/currency/rates/<str:from_currency>/<str:to_currency>/historical',
    FetchHistoricalCurrencyExchangeRates.as_view(),
    name='fetch-historical-exchange-rates'),
]
