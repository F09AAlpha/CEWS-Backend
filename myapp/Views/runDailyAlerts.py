from rest_framework.views import APIView
from rest_framework.response import Response
from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert
import requests


class RunDailyAlertChecks(APIView):
    def get(self, request):
        alerts = ExchangeRateAlert.objects.all()
        results = []

        for alert in alerts:
            base = alert.base
            target = alert.target

            # Call the currency rate API
            # rate_response = requests.get(f"http://127.0.0.1:8000/api/v1/currency/rates/{base}/{target}/")
            
            # Call the prediction API
            prediction_response = requests.get(f"http://127.0.0.1:8000/api/v2/analytics/prediction/{base}/{target}/")

            results.append({
                "alert_id": alert.alert_id,
                "base": base,
                "target": target,
                # "rate_status": rate_response.status_code,
                "prediction_status": prediction_response.status_code
            })

        return Response({"alerts_checked": len(results), "details": results})
