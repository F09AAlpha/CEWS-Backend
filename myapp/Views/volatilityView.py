from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from myapp.Service.volatilityService import VolatilityService
from myapp.Serializers.volatilitySerializer import VolatilityAnalysisSerializer
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class VolatilityAnalysisView(APIView):
    """
    API endpoint for currency volatility analysis compliant with ADAGE 3.0
    """

    # Cache results for 1 hour (3600 seconds)
    @method_decorator(cache_page(3600))
    def get(self, request, base, target):
        # Set ADAGE 3.0 response headers
        headers = {
            "X-ADAGE-Version": "3.0",
            "X-ADAGE-Source": "Currency Exchange Warning System"
        }

        # Validate currency codes first
        if not self._is_valid_currency_code(base):
            return self._create_error_response(
                "InvalidCurrencyError",
                f"Invalid base currency code: {base}. Currency codes must be 3 alphabetic characters.",
                status.HTTP_400_BAD_REQUEST,
                headers
            )

        if not self._is_valid_currency_code(target):
            return self._create_error_response(
                "InvalidCurrencyError",
                f"Invalid target currency code: {target}. Currency codes must be 3 alphabetic characters.",
                status.HTTP_400_BAD_REQUEST,
                headers
            )

        if base.upper() == target.upper():
            return self._create_error_response(
                "InvalidInputError",
                "Base and target currencies must be different.",
                status.HTTP_400_BAD_REQUEST,
                headers
            )

        try:
            # Get days parameter (default: 30)
            try:
                days = int(request.query_params.get('days', 30))

                # Validate days parameter
                if days < 7:
                    return self._create_error_response(
                        "ValidationError",
                        f"Days parameter must be at least 7 (got {days}).",
                        status.HTTP_400_BAD_REQUEST,
                        headers
                    )
                elif days > 365:
                    return self._create_error_response(
                        "ValidationError",
                        f"Days parameter must be at most 365 (got {days}).",
                        status.HTTP_400_BAD_REQUEST,
                        headers
                    )
            except ValueError:
                return self._create_error_response(
                    "ValidationError",
                    f"Days parameter must be an integer value, got: {request.query_params.get('days')}",
                    status.HTTP_400_BAD_REQUEST,
                    headers
                )

            # Perform volatility analysis with ADAGE 3.0 format
            try:
                volatility_service = VolatilityService()
                analysis_result = volatility_service.calculate_volatility(base.upper(), target.upper(), days)
            except ValueError as e:
                # Handle specific ValueError cases
                error_msg = str(e)
                if "No data available" in error_msg:
                    return self._create_error_response(
                        "NoDataError",
                        error_msg,
                        status.HTTP_404_NOT_FOUND,
                        headers
                    )
                else:
                    # Other value errors are client errors
                    return self._create_error_response(
                        "ValidationError",
                        error_msg,
                        status.HTTP_400_BAD_REQUEST,
                        headers
                    )
            except Exception as e:
                error_msg = str(e)
                if "Invalid API" in error_msg:
                    # API service errors
                    return self._create_error_response(
                        "ExternalServiceError",
                        "Error accessing external data provider. Please try again later.",
                        status.HTTP_503_SERVICE_UNAVAILABLE,
                        headers
                    )
                else:
                    # Log the full exception for debugging
                    logger.error(f"Unexpected error in volatility analysis: {error_msg}", exc_info=True)
                    return self._create_error_response(
                        "ServerError",
                        "An unexpected error occurred during volatility analysis.",
                        status.HTTP_500_INTERNAL_SERVER_ERROR,
                        headers
                    )

            # Validate with serializer
            serializer = VolatilityAnalysisSerializer(data=analysis_result)
            if serializer.is_valid():
                return Response(serializer.data, headers=headers)
            else:
                # Data validation error (likely a bug in our code)
                return self._create_error_response(
                    "DataValidationError",
                    "Data validation failed in the response format.",
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    headers,
                    details=serializer.errors
                )

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unhandled error in volatility analysis: {str(e)}", exc_info=True)
            return self._create_error_response(
                "ServerError",
                "An unexpected error occurred.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                headers
            )

    def _is_valid_currency_code(self, code):
        """Validate if the string is a valid currency code (3 alphabetic characters)."""
        if not code or not isinstance(code, str):
            return False
        return bool(re.match(r'^[A-Za-z]{3}$', code))

    def _create_error_response(self, error_type, message, status_code, headers, details=None):
        """Create a standardized error response following ADAGE 3.0 format."""
        error_response = {
            "error": {
                "type": error_type,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }

        if details:
            error_response["error"]["details"] = details

        return Response(error_response, status=status_code, headers=headers)
