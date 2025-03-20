from django.urls import path
from myapp.Views.healthCheckView import HealthCheckView
from myapp.Views.financialNewsView import FetchFinancialNewsView, FinancialNewsListView
from myapp.Views.currencyNewsView import CurrencyNewsListView, FetchCurrencyNewsView
from myapp.Views.exchangeRateLatestViews import CurrencyRateView
from myapp.Views.historicalExchangeRatesView import FetchHistoricalCurrencyExchangeRates
from myapp.Views.volatilityView import VolatilityAnalysisView


urlpatterns = [
    path('', HealthCheckView.as_view(), name='health-check'),
    path('fetch-financial-news/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('financial-news/', FinancialNewsListView.as_view(), name='financial-news-list'),
    path('currency/<str:currency_code>/', FetchCurrencyNewsView.as_view(), name='fetch-currency-news'),
    path('currency-news/', CurrencyNewsListView.as_view(), name='currency-news-list'),
    path('v1/currency/rates/<str:base>/<str:target>/', CurrencyRateView.as_view(), name='currency-rate'),
    path(
        'v1/currency/rates/<str:from_currency>/<str:to_currency>/historical',
        FetchHistoricalCurrencyExchangeRates.as_view(),
        name='fetch-historical-exchange-rates'
    ),
    path(
        'v1/analytics/volatility/<str:base>/<str:target>/',
        VolatilityAnalysisView.as_view(),
        name='volatility_analysis',
    ),
]
