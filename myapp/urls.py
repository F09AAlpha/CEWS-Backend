from django.urls import path
from myapp.Views.healthCheckView import HealthCheckView
from myapp.Views.economicIndicatorsView import StoreAnnualIndicatorsView, StoreMonthlyIndicatorsView
from myapp.Views.financialNewsView import FetchFinancialNewsView
from myapp.Views.currencyNewsView import CurrencyNewsListView, FetchCurrencyNewsView
from myapp.Views.exchangeRateLatestViews import CurrencyRateView
from myapp.Views.historicalExchangeRatesView import FetchHistoricalCurrencyExchangeRates
from myapp.Views.volatilityView import VolatilityAnalysisView
from myapp.Views.graphView_lastweek import GraphView_lastweek
from myapp.Views.graphView_lastmonth import GraphView_lastmonth
from myapp.Views.graphView_last6months import GraphView_last6months
from myapp.Views.graphView_lastyear import GraphView_lastyear
from myapp.Views.graphView_last5years import GraphView_last5years

urlpatterns = [
    path('', HealthCheckView.as_view(), name='health-check'),
    path('v1/financial/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('v1/currency/<str:currency>', FetchCurrencyNewsView.as_view(), name='fetch-currency-news'),
    path('v1/news/events', CurrencyNewsListView.as_view(), name='currency-news-list'),
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
    path('v1/economic/indicators/annual/', StoreAnnualIndicatorsView.as_view(), name='store-annual-economic-indicators'),
    path('v1/economic/indicators/monthly/', StoreMonthlyIndicatorsView.as_view(), name='store-monthly-economic-indicators'),
    path('graph/<str:from_currency>/<str:to_currency>/last-week', GraphView_lastweek.as_view(), name='last-week-graph_view'),
    path('graph/<str:from_currency>/<str:to_currency>/last-month', GraphView_lastmonth.as_view(), name='last-month-graph_view'),
    path('graph/<str:from_currency>/<str:to_currency>/last-6-months',
         GraphView_last6months.as_view(),
         name='last-6-months-graph_view'),
    path('graph/<str:from_currency>/<str:to_currency>/last-year',
         GraphView_lastyear.as_view(),
         name='last-year-graph_view'),
    path('graph/<str:from_currency>/<str:to_currency>/last-5-years',
         GraphView_last5years.as_view(),
         name='last-5-years-graph_view')

]
