from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
import re
import traceback
from django.utils import timezone

from myapp.Service.predictionService import PredictionService
from myapp.Serializers.predictionSerializer import CurrencyPredictionSerializer
from myapp.Exceptions.exceptions import (
    InvalidCurrencyCode,
    InvalidCurrencyPair
)

logger = logging.getLogger(__name__)


class CurrencyPredictionView(APIView):
    """
    API endpoint for currency exchange rate prediction compliant with ADAGE 3.0
    """

    def get(self, request, base, target):
        try:
            # Check if we should refresh the prediction
            refresh_mode = request.query_params.get('refresh', 'false').lower() == 'true'

            # Get forecast_horizon parameter (default: 7 days)
            try:
                forecast_horizon = int(request.query_params.get('forecast_horizon', 7))

                # Limit forecast horizon to reasonable range
                if forecast_horizon < 1:
                    forecast_horizon = 1
                elif forecast_horizon > 30:
                    forecast_horizon = 30
            except ValueError:
                return self._error_response(
                    "InvalidParameter",
                    "forecast_horizon must be an integer",
                    status.HTTP_400_BAD_REQUEST
                )

            # Validate currency codes
            self.validate_currency_code(base)
            self.validate_currency_code(target)

            # Standardize currency codes
            base = base.upper()
            target = target.upper()

            # Check for same currency
            self.validate_currency_pair(base, target)

            # Set ADAGE 3.0 response headers
            headers = {
                "X-ADAGE-Version": "3.0",
                "X-ADAGE-Source": "Currency Exchange Warning System"
            }

            # Initialize prediction service
            prediction_service = PredictionService()

            try:
                # Get or create prediction
                prediction = prediction_service.create_prediction(
                    base, target, forecast_horizon, refresh=refresh_mode
                )

                # Format response according to ADAGE 3.0
                response_data = prediction_service.format_adage_response(prediction)

                # Validate with serializer
                serializer = CurrencyPredictionSerializer(data=response_data)
                if serializer.is_valid():
                    return Response(serializer.data, headers=headers)
                else:
                    # Log validation errors
                    logger.error(f"Serializer validation error: {serializer.errors}")

                    # ADAGE 3.0 compliant error response
                    return self._error_response(
                        "ValidationError",
                        "Data validation failed",
                        status.HTTP_500_INTERNAL_SERVER_ERROR,
                        details=serializer.errors
                    )

            except Exception as e:
                logger.error(f"Error in prediction process: {str(e)}\n{traceback.format_exc()}")
                return self._error_response(
                    "PredictionError",
                    f"Error generating prediction: {str(e)}",
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except InvalidCurrencyCode as e:
            logger.warning(f"Invalid currency code: {str(e)}")
            return self._error_response(
                "InvalidCurrencyCode",
                str(e),
                status.HTTP_400_BAD_REQUEST
            )

        except InvalidCurrencyPair as e:
            logger.warning(f"Invalid currency pair: {str(e)}")
            return self._error_response(
                "InvalidCurrencyPair",
                str(e),
                status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return self._error_response(
                "ServerError",
                "An unexpected error occurred",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def validate_currency_code(self, code):
        """
        Validate that a currency code is properly formatted.

        Args:
            code (str): Currency code to validate

        Raises:
            InvalidCurrencyCode: If the code is not valid
        """
        if not re.match(r'^[A-Za-z]{3}$', code):
            raise InvalidCurrencyCode("Currency codes must be 3 alphabetic characters")

    def validate_currency_pair(self, base, target):
        """
        Validate that base and target currencies are different.

        Args:
            base (str): Base currency code
            target (str): Target currency code

        Raises:
            InvalidCurrencyPair: If the base and target are the same
        """
        if base.upper() == target.upper():
            raise InvalidCurrencyPair("Base and target currencies must be different")

    def _error_response(self, error_type, message, status_code, details=None):
        """
        Create a standardized error response following ADAGE 3.0 format.

        Args:
            error_type (str): Type of error
            message (str): Error message
            status_code (int): HTTP status code
            details (dict, optional): Additional error details

        Returns:
            Response: Django REST Framework response object
        """
        response = {
            "error": {
                "type": error_type,
                "message": message,
                "timestamp": timezone.now().isoformat()
            }
        }

        if details:
            response["error"]["details"] = details

        headers = {
            "X-ADAGE-Version": "3.0",
            "X-ADAGE-Source": "Currency Exchange Warning System"
        }

        return Response(response, status=status_code, headers=headers)
