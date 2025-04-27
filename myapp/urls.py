from django.urls import path
from myapp.Views.healthCheckView import HealthCheckView
from myapp.Views.economicIndicatorsView import StoreAnnualIndicatorsView, StoreMonthlyIndicatorsView
from myapp.Views.financialNewsView import FetchFinancialNewsView
from myapp.Views.currencyNewsView import CurrencyNewsListView, FetchCurrencyNewsView
from myapp.Views.exchangeRateLatestViews import CurrencyRateView
from myapp.Views.historicalExchangeRatesView import FetchHistoricalCurrencyExchangeRates
from myapp.Views.historicalExchangeRatesViewV2 import FetchHistoricalCurrencyExchangeRatesV2
from myapp.Views.volatilityView import VolatilityAnalysisView
from myapp.Views.graphView import (GraphView_lastweek, GraphView_lastmonth,
                                   GraphView_last6months, GraphView_lastyear, GraphView_last5years)
from myapp.Views.registerExchangeRateAlertView import RegisterAlertView
from myapp.Views.anomalyDetectionView import anomaly_detection
from myapp.Views.correlationView import CorrelationAnalysisView
from myapp.Views.predictionView import CurrencyPredictionView
from myapp.Views.runDailyAlerts import RunDailyAlertChecks

urlpatterns = [
    path('', HealthCheckView.as_view(), name='health-check'),
    path('v1/financial/', FetchFinancialNewsView.as_view(), name='fetch-financial-news'),
    path('v1/currency/<str:currency>', FetchCurrencyNewsView.as_view(), name='fetch-currency-news'),
    path('v1/news/events', CurrencyNewsListView.as_view(), name='currency-news-list'),
    path('v1/currency/rates/<str:base>/<str:target>/', CurrencyRateView.as_view(), name='currency-rate'),
    path('v2/alerts/register/', RegisterAlertView.as_view(), name='register-alert'),
    path('v3/alerts/daily-check/', RunDailyAlertChecks.as_view(), name='run-daily-alert-checks'),
    path(
        'v1/currency/rates/<str:from_currency>/<str:to_currency>/historical',
        FetchHistoricalCurrencyExchangeRates.as_view(),
        name='fetch-historical-exchange-rates'
    ),
    path(
        'v2/currency/rates/<str:from_currency>/<str:to_currency>/historical',
        FetchHistoricalCurrencyExchangeRatesV2.as_view(),
        name='fetch-historical-exchange-rates-v2'
    ),
    path(
        'v1/analytics/volatility/<str:base>/<str:target>/',
        VolatilityAnalysisView.as_view(),
        name='volatility_analysis',
    ),
    path(
        'v2/analytics/correlation/<str:base>/<str:target>/',
        CorrelationAnalysisView.as_view(),
        name='correlation_analysis',
    ),
    path(
        'v2/analytics/prediction/<str:base>/<str:target>/',
        CurrencyPredictionView.as_view(),
        name='currency_prediction',
    ),
    path('v1/economic/indicators/annual/', StoreAnnualIndicatorsView.as_view(), name='store-annual-economic-indicators'),
    path('v1/economic/indicators/monthly/', StoreMonthlyIndicatorsView.as_view(), name='store-monthly-economic-indicators'),
    path('v1/graph/<str:from_currency>/<str:to_currency>/last-week', GraphView_lastweek.as_view(), name='last-week-graph_view'),
    path('v1/graph/<str:from_currency>/<str:to_currency>/last-month', GraphView_lastmonth.as_view(),
         name='last-month-graph_view'),
    path('v1/graph/<str:from_currency>/<str:to_currency>/last-6-months',
         GraphView_last6months.as_view(),
         name='last-6-months-graph_view'),
    path('v1/graph/<str:from_currency>/<str:to_currency>/last-year',
         GraphView_lastyear.as_view(),
         name='last-year-graph_view'),
    path('v1/graph/<str:from_currency>/<str:to_currency>/last-5-years',
         GraphView_last5years.as_view(),
         name='last-5-years-graph_view'),
    path('v2/analytics/anomaly-detection/', anomaly_detection, name='anomaly-detection'),
]
