# api/views.py
import logging
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError, NotFound

from myapp.Serializers.anomalyDetectionSerializer import (
    AnomalyDetectionRequestSerializer,
    ADAGEConverter
)
from myapp.Service.anomalyDetectionService import (
    AnomalyDetectionService,
    InsufficientDataError,
    ProcessingError
)
from myapp.Service.alpha_vantage import (
    AlphaVantageError,
    RateLimitError,
    InvalidRequestError,
    TemporaryAPIError
)
from myapp.Models.anomalyDetectionModel import AnomalyDetectionResult

logger = logging.getLogger(__name__)


@api_view(['POST'])
def anomaly_detection(request):
    """
    Detect anomalies in currency exchange rates.

    Request body parameters:
    - base: Base currency code (e.g., EUR)
    - target: Target currency code (e.g., USD)
    - days: Analysis period in days (default: 30)

    Returns:
        Response: API response with anomaly detection results in ADAGE format
    """
    try:
        # Validate request
        serializer = AnomalyDetectionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Get validated data
        data = serializer.validated_data
        base = data['base']
        target = data['target']
        days = data.get('days', 30)

        # Perform anomaly detection
        detector = AnomalyDetectionService(
            base_currency=base,
            target_currency=target,
            analysis_period_days=days
        )
        result = detector.detect_anomalies()

        # Convert to ADAGE 3.0 format
        adage_result = ADAGEConverter.convert_to_adage_format(result)

        # Store result in the database
        AnomalyDetectionResult.objects.create(
            base_currency=base,
            target_currency=target,
            analysis_period_days=days,
            anomaly_count=result['anomaly_count'],
            result_data=adage_result
        )

        # Return results
        return Response(adage_result)

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return Response(
            {'detail': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

    except InvalidRequestError as e:
        logger.warning(f"Invalid request: {str(e)}")
        return Response(
            {'detail': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

    except NotFound as e:
        logger.warning(f"Not found: {str(e)}")
        return Response(
            {'detail': str(e)},
            status=status.HTTP_404_NOT_FOUND
        )

    except InsufficientDataError as e:
        logger.warning(f"Insufficient data: {str(e)}")
        return Response(
            {'detail': str(e)},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY
        )

    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        return Response(
            {'detail': "API rate limit exceeded. Please try again later."},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    except TemporaryAPIError as e:
        logger.error(f"Temporary API error: {str(e)}")
        return Response(
            {'detail': "Temporary API error. Please try again later."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    except (AlphaVantageError, ProcessingError) as e:
        logger.error(f"Processing error: {str(e)}")
        return Response(
            {'detail': "Error processing request. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return Response(
            {'detail': "An unexpected error occurred. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
