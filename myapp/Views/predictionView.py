from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
import re
import traceback
from datetime import datetime
from django.utils import timezone
import os
from django.db.models import Q
from myapp.models import ExchangeRateAlert
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from myapp.Service.predictionService import PredictionService
from myapp.Serializers.predictionSerializer import CurrencyPredictionSerializer
from myapp.Exceptions.exceptions import (
    InvalidCurrencyCode,
    InvalidCurrencyPair
)

logger = logging.getLogger(__name__)

EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')


def is_valid_email(email):
    """
    Simple email validation function using regex.
    """
    # Email regex pattern for basic validation
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(email_regex, email) is not None


def send_alert_email(to_email, base, target, rate, alert_type, threshold, prediction_day):
    """
    Sends an email notification for an exchange rate alert.
    """
    # Log email sending attempt
    logger.debug(f"Preparing to send email to {to_email} for {base}/{target} with rate {rate}")

    if not EMAIL_HOST_USER or not EMAIL_HOST_PASSWORD:
        logger.error("Email credentials (EMAIL_HOST_USER or EMAIL_HOST_PASSWORD) are missing.")
        return

    if not to_email:
        logger.error("Recipient email (to_email) is missing.")
        return

    if not is_valid_email(to_email):
        logger.error(f"Invalid recipient email address: {to_email}")
        return

    # Log the fact that email credentials are used, but not their values
    logger.debug(f"Email Host: {EMAIL_HOST_USER} (credentials used, but not displayed for security reasons)")

    subject = f"Exchange Rate Prediction Alert: {base}/{target} on {prediction_day}"
    body = (f"The exchange rate for {base}/{target} is predicted to "
            f"{'rise above' if alert_type == 'above' else 'fall below'} {threshold}. Predicted rate: {rate} on {prediction_day}")

    msg = MIMEMultipart()
    msg["From"] = EMAIL_HOST_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Start SMTP session
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # Log in to the email server
        logger.debug("Attempting to log in to the SMTP server.")
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)

        # Send the email
        logger.debug(f"Sending email to {to_email}...")
        server.sendmail(EMAIL_HOST_USER, to_email, msg.as_string())
        server.quit()

        logger.info(f"Alert email sent successfully to {to_email} for {base}/{target} with rate {rate}")
    except smtplib.SMTPException as e:
        logger.error(f"Failed to send alert email via SMTP: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in email sending: {str(e)}")
class CurrencyPredictionView(APIView):
    """
    API endpoint for currency exchange rate prediction compliant with ADAGE 3.0
    """

    def get(self, request, base, target):
        """
        Get currency exchange rate prediction.

        Parameters:
        - base: Base currency code (3 letters)
        - target: Target currency code (3 letters)

        Query parameters:
        - refresh: Whether to refresh the prediction (default: false)
        - forecast_horizon: Number of days to forecast (default: 7)
        - model: Model type to use - 'arima', 'statistical', or 'auto'
          (default: 'arima')
        - confidence: Confidence level for prediction intervals (50-99,
          default: 80)
          Lower values (e.g., 70-80) give tighter, more precise bounds
          Higher values (e.g., 90-95) give wider bounds with more certainty
        - backtest: Whether to include historical data for the past 7 days
          (default: false)

        Returns:
        - ADAGE 3.0 compliant prediction response
        """
        try:
            # Check if we should refresh the prediction
            refresh_mode = request.query_params.get(
                'refresh', 'false').lower() == 'true'

            # Check if we should use ARIMA model
            model_type = request.query_params.get('model', 'arima').lower()
            use_arima = model_type in ['auto', 'arima']
            # Check if we should include backtest data
            include_backtest = request.query_params.get(
                'backtest', 'false').lower() == 'true'

            # Get confidence level parameter (default: 80)
            try:
                confidence_level = int(
                    request.query_params.get('confidence', 80))
                if confidence_level < 50 or confidence_level > 99:
                    return self._error_response(
                        "InvalidParameter",
                        "confidence must be between 50 and 99",
                        status.HTTP_400_BAD_REQUEST
                    )
            except ValueError:
                return self._error_response(
                    "InvalidParameter",
                    "confidence must be an integer",
                    status.HTTP_400_BAD_REQUEST
                )

            # If model_type is not valid, return error
            if model_type not in ['auto', 'arima', 'statistical']:
                return self._error_response(
                    "InvalidParameter",
                    "model must be one of: auto, arima, statistical",
                    status.HTTP_400_BAD_REQUEST
                )

            # Get forecast_horizon parameter (default: 7 days)
            try:
                forecast_horizon = int(
                    request.query_params.get('forecast_horizon', 7))

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
                    base, target, forecast_horizon, refresh=refresh_mode,
                    use_arima=use_arima if model_type != 'statistical'
                    else False,
                    confidence_level=confidence_level
                )
                # Format response according to ADAGE 3.0
                response_data = prediction_service.format_adage_response(
                    prediction, include_backtest
                )

                # Validate with serializer
                serializer = CurrencyPredictionSerializer(data=response_data)

                if serializer.is_valid():
                    event = serializer.data['events'][0]
                    base = event['attributes']['base_currency']
                    target = event['attributes']['target_currency']
                    prediction_values = event['attributes']['prediction_values'][:3]  # First 3 days

                    # Tracker to avoid duplicate emails
                    sent_alerts = set()

                    for prediction in prediction_values:
                        predicted_rate = abs(prediction['mean'])
                        prediction_day_raw = prediction['timestamp']
                        dt = datetime.strptime(prediction_day_raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                        prediction_day = dt.strftime("%B %d, %Y")
                        # Find matching alerts
                        alerts = ExchangeRateAlert.objects.filter(
                            base=base,
                            target=target
                        ).filter(
                            Q(alert_type="above", threshold__lte=predicted_rate) |
                            Q(alert_type="below", threshold__gte=predicted_rate)
                        )

                        for alert in alerts:
                            if alert.alert_id in sent_alerts:
                                continue  # Skip duplicate
                                    
                            send_alert_email(
                                alert.email,
                                base,
                                target,
                                round(predicted_rate, 2),
                                alert.alert_type,
                                round(alert.threshold, 2),
                                prediction_day
                            )
                            sent_alerts.add(alert.alert_id)  # Mark as sent
                            logger.info(
                                f"Alert triggered for {base}/{target} at predicted rate {predicted_rate} on {prediction_day} (threshold {alert.threshold})"
                            )
                    return Response(serializer.data, headers=headers)

                else:
                    # Log validation errors
                    logger.error(
                        f"Serializer validation error: {serializer.errors}"
                    )

                    # ADAGE 3.0 compliant error response
                    return self._error_response(
                        "ValidationError",
                        "Data validation failed",
                        status.HTTP_500_INTERNAL_SERVER_ERROR,
                        details=serializer.errors
                    )
            except Exception as e:
                logger.error(
                    f"Error in prediction process: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
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
            logger.error(
                f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            )
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
            raise InvalidCurrencyCode(
                "Currency codes must be 3 alphabetic characters"
                )

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
            raise InvalidCurrencyPair(
                "Base and target currencies must be different"
                )

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
