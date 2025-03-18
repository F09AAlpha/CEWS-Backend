from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.Service.exchangeRateLatestService import AlphaVantageService
from myapp.Serializers.exchangeRateLatestSerializer import CurrencyRateSerializer
import logging

logger = logging.getLogger(__name__)


class CurrencyRateView(APIView):
    """
    API endpoint to retrieve exchange rates between two currencies
    """

    def get(self, request, base, target):
        """
        Get the latest exchange rate between two currencies

        Args:
            base (str): Base currency code (e.g., EUR)
            target (str): Target currency code (e.g., USD)

        Returns:
            Response: JSON response with exchange rate data
        """
        try:
            # Validate currency codes (optional, can be expanded)
            base = base.upper()
            target = target.upper()

            if len(base) > 10 or len(target) > 10:
                return Response(
                    {'detail': 'Invalid currency code length'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get exchange rate from Alpha Vantage
            currency_event = AlphaVantageService.get_exchange_rate(base, target)

            # Serialize the response
            serializer = CurrencyRateSerializer(currency_event)

            return Response(serializer.data)

        except Exception as e:
            logger.exception(f"Error in CurrencyRateView: {str(e)}")
            return Response(
                {'detail': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
