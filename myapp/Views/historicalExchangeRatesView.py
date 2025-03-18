import requests
from rest_framework.response import Response
from rest_framework.views import APIView

# External API URL (Example: NewsAPI)
HISTORICAL_RATES_URL = "https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol=EUR&to_symbol=USD&apikey=demo"
API_KEY1 = "GARIZYY99Q85ILKW"  # Replace with your actual API key
API_KEY2 = "I3QWJJIT433PTHMU"

class FetchHistoricalCurrencyExchangeRates(APIView):

    def get(self, request, from_currency, to_currency, *args, **kwargs):
        API_URL = f"https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY1}"
        try:
            response = requests.get(API_URL)
        except:
            return Response(
                    {"error": f"Failed to fetch historical exchange rates: {response.status_code}, {response.json()}"},
                    status=response.status_code
                )
            
        response = response.json()
        
        return Response(
            response,
            status=201
        )