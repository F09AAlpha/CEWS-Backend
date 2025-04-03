from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
import re
import traceback
from django.utils import timezone
from django.core.cache import cache
from myapp.Service.correlationService import CorrelationService
from myapp.Serializers.correlationSerializer import CorrelationAnalysisSerializer

logger = logging.getLogger(__name__)


class CorrelationAnalysisView(APIView):
    """
    API endpoint for currency correlation analysis compliant with ADAGE 3.0
    """

    def get(self, request, base, target):
        try:
            # Check if we should refresh the analysis
            refresh_mode = request.query_params.get('refresh', 'false').lower() == 'true'

            # If not refreshing, try to use cached response
            if not refresh_mode:
                # Try to get from cache
                cache_key = f"correlation_{base.upper()}_{target.upper()}"
                cached_response = cache.get(cache_key)
                if cached_response:
                    logger.info(f"Using cached response for {base}/{target}")
                    return Response(
                        cached_response,
                        headers={
                            "X-ADAGE-Version": "3.0",
                            "X-ADAGE-Source": "Currency Exchange Warning System"
                        }
                    )

            # Validate currency codes
            if not self._validate_currency_code(base) or not self._validate_currency_code(target):
                return self._error_response(
                    "InvalidCurrencyCode",
                    "Currency codes must be 3 alphabetic characters",
                    status.HTTP_400_BAD_REQUEST
                )

            # Standardize currency codes
            base = base.upper()
            target = target.upper()

            # Check for same currency
            if base == target:
                return self._error_response(
                    "InvalidCurrencyPair",
                    "Base and target currencies must be different",
                    status.HTTP_400_BAD_REQUEST
                )

            # Get lookback_days parameter (default: 90)
            try:
                lookback_days = int(request.query_params.get('lookback_days', 90))

                # Limit analysis period to reasonable range
                if lookback_days < 30:
                    lookback_days = 30
                elif lookback_days > 365:
                    lookback_days = 365
            except ValueError:
                return self._error_response(
                    "InvalidParameter",
                    "lookback_days must be an integer",
                    status.HTTP_400_BAD_REQUEST
                )

            # Check if we should use fallback mode (minimal analysis)
            fallback_mode = request.query_params.get('fallback', 'true').lower() == 'true'

            # Set ADAGE 3.0 response headers
            headers = {
                "X-ADAGE-Version": "3.0",
                "X-ADAGE-Source": "Currency Exchange Warning System"
            }

            # Use a try-except block for each step with detailed logging
            correlation_service = CorrelationService()
            correlation_result = None

            try:
                # Only look for existing result if not in refresh mode
                if not refresh_mode:
                    correlation_result = correlation_service.get_latest_correlation(base, target)

                # If refresh mode, or no result found, or lookback_days is different, perform new analysis
                if refresh_mode or correlation_result is None or correlation_result.lookback_days != lookback_days:
                    try:
                        logger.info(f"Starting new correlation analysis for {base}/{target} with lookback_days={lookback_days}")
                        correlation_result = correlation_service.analyze_and_store_correlations(
                            base, target, lookback_days
                        )
                        logger.info(f"Completed correlation analysis for {base}/{target}")
                    except ValueError as e:
                        logger.warning(f"ValueError in correlation analysis: {str(e)}")
                        return self._error_response(
                            "NoDataAvailable",
                            str(e),
                            status.HTTP_404_NOT_FOUND
                        )
                    except Exception as e:
                        logger.error(f"Error in correlation analysis: {str(e)}\n{traceback.format_exc()}")
                        if fallback_mode:
                            # In fallback mode, create a minimal valid result
                            logger.info("Using fallback mode to create minimal valid result")
                            try:
                                # Create a minimal correlation result
                                correlation_result = self._create_fallback_result(base, target, lookback_days)
                            except Exception as fallback_error:
                                logger.error(f"Error creating fallback result: {str(fallback_error)}")
                                return self._error_response(
                                    "AnalysisError",
                                    f"Error analyzing correlations: {str(e)}",
                                    status.HTTP_500_INTERNAL_SERVER_ERROR
                                )
                        else:
                            return self._error_response(
                                "AnalysisError",
                                f"Error analyzing correlations: {str(e)}",
                                status.HTTP_500_INTERNAL_SERVER_ERROR
                            )
                else:
                    logger.info(f"Using existing correlation analysis for {base}/{target}")
            except Exception as e:
                logger.error(f"Unexpected error retrieving correlation data: {str(e)}\n{traceback.format_exc()}")
                if fallback_mode:
                    # In fallback mode, create a minimal valid result
                    logger.info("Using fallback mode to create minimal valid result")
                    try:
                        # Create a minimal correlation result
                        correlation_result = self._create_fallback_result(base, target, lookback_days)
                    except Exception as fallback_error:
                        logger.error(f"Error creating fallback result: {str(fallback_error)}")
                        return self._error_response(
                            "AnalysisError",
                            f"Error retrieving correlation data: {str(e)}",
                            status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                else:
                    return self._error_response(
                        "AnalysisError",
                        f"Error retrieving correlation data: {str(e)}",
                        status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

            # Format response according to ADAGE 3.0
            try:
                response_data = correlation_service.format_adage_response(correlation_result)

                # Validate with serializer
                serializer = CorrelationAnalysisSerializer(data=response_data)
                if serializer.is_valid():
                    # Cache result if not in refresh mode (for 6 hours)
                    if not refresh_mode:
                        cache_key = f"correlation_{base.upper()}_{target.upper()}"
                        cache.set(cache_key, serializer.data, 21600)  # Cache for 6 hours

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
                logger.error(f"Error formatting response: {str(e)}\n{traceback.format_exc()}")
                return self._error_response(
                    "ResponseFormattingError",
                    f"Error formatting response: {str(e)}",
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            logger.error(f"Unexpected error in correlation analysis: {str(e)}\n{traceback.format_exc()}")
            return self._error_response(
                "ServerError",
                f"An unexpected error occurred: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _validate_currency_code(self, code):
        """Validate that code is a 3-letter currency code"""
        return bool(re.match(r'^[A-Za-z]{3}$', code))

    def _error_response(self, error_type, message, status_code, details=None):
        """Create an ADAGE 3.0 compliant error response"""
        error_response = {
            "error": {
                "type": error_type,
                "message": message,
                "timestamp": timezone.now().isoformat()
            }
        }

        if details:
            error_response["error"]["details"] = details

        return Response(
            error_response,
            status=status_code,
            headers={"X-ADAGE-Version": "3.0"}
        )

    def _create_fallback_result(self, base, target, lookback_days):
        """Create a minimal valid correlation result for fallback"""
        from myapp.Models.correlationModel import CorrelationResult
        from django.db import transaction

        with transaction.atomic():
            correlation_result = CorrelationResult(
                base_currency=base,
                target_currency=target,
                lookback_days=lookback_days,
                exchange_news_correlation={},
                exchange_economic_correlation={},
                volatility_news_correlation={},
                volatility_economic_correlation={},
                top_influencing_factors=[
                    {
                        'factor': 'volatility',
                        'impact': 'medium',
                        'correlation': 0.5,
                        'type': 'fallback'
                    }
                ],
                confidence_score=30.0,  # Low confidence for fallback data
                data_completeness=50.0  # 50% completeness for fallback
            )
            correlation_result.save()

        return correlation_result
