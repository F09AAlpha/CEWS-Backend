from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import logging
from datetime import datetime
import os
import pytz
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from django.db.models import Q
from myapp.models import ExchangeRateAlert
from myapp.Serializers.exchangeRateLatestSerializer import (
    AdageCurrencyRateSerializer,
    TimeObjectSerializer,
    CurrencyRateAttributesSerializer,
    CurrencyRateEventSerializer
)
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr

# Load environment variables from the .env file
load_dotenv()

logger = logging.getLogger(__name__) 

ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
EMAIL_HOST_USER = "cewsalerts@gmail.com"
EMAIL_HOST_PASSWORD = "cewsalerts123"


def is_valid_email(email):
    """
    Simple email validation function using regex.
    """
    # Email regex pattern for basic validation
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(email_regex, email) is not None

def send_alert_email(to_email, base, target, rate, alert_type, threshold):
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

    subject = f"Exchange Rate Alert: {base}/{target}"
    body = f"The exchange rate for {base}/{target} has {'risen above' if alert_type == 'above' else 'fallen below'} {threshold}. Current rate: {rate}"

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


class CurrencyRateView(APIView):
    """
    API view for retrieving the latest exchange rate between two currencies.
    This is a direct call to external API and not stored in the DB.
    Returns data in ADAGE 3.0 Data Model format.
    """

    def get(self, request, base, target):
        try:
            base = base.upper()
            target = target.upper()

            if len(base) != 3 or len(target) != 3:
                return Response(
                    {"detail": "Invalid currency code. Currency codes must be 3 characters."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Call Alpha Vantage API
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": base,
                "to_currency": target,
                "apikey": ALPHA_VANTAGE_API_KEY
            }

            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return Response(
                    {"detail": "External API error: Unable to fetch exchange rate data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            exchange_rate_data = data.get("Realtime Currency Exchange Rate", {})

            if not exchange_rate_data:
                return Response(
                    {"detail": "No exchange rate data found for the specified currency pair."},
                    status=status.HTTP_404_NOT_FOUND
                )

            now = datetime.now(pytz.UTC)
            now_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")

            timestamp = exchange_rate_data.get(
                "6. Last Refreshed",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            event_id = f"CE-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

            dataset_time_object = {
                "timestamp": now_str,
                "timezone": "UTC"
            }

            time_object_serializer = TimeObjectSerializer(data=dataset_time_object)
            if not time_object_serializer.is_valid():
                logger.error(f"Invalid time object format: {time_object_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            try:
                rate = float(exchange_rate_data.get("5. Exchange Rate", "0.00"))
            except ValueError:
                rate = 0.0
                logger.warning("Could not convert rate to float, using default 0.0")

            attributes = {
                "base": exchange_rate_data.get("1. From_Currency Code", base),
                "target": exchange_rate_data.get("3. To_Currency Code", target),
                "rate": rate,
                "bid_price": exchange_rate_data.get("8. Bid Price", None),
                "ask_price": exchange_rate_data.get("9. Ask Price", None),
                "source": "Alpha Vantage"
            }

            attributes_serializer = CurrencyRateAttributesSerializer(data=attributes)
            if not attributes_serializer.is_valid():
                logger.error(f"Invalid attributes format: {attributes_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            event = {
                "time_object": {
                    "timestamp": timestamp,
                    "duration": 0,
                    "duration_unit": "second",
                    "timezone": "UTC"
                },
                "event_type": "currency_rate",
                "event_id": event_id,
                "attributes": attributes_serializer.validated_data
            }

            event_serializer = CurrencyRateEventSerializer(data=event)
            if not event_serializer.is_valid():
                logger.error(f"Invalid event format: {event_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            adage_data = {
                "data_source": "Alpha Vantage",
                "dataset_type": "currency_exchange_rate",
                "dataset_id": f"currency-{base}-{target}-{now.strftime('%Y%m%d')}",
                "time_object": time_object_serializer.validated_data,
                "events": [event_serializer.validated_data]
            }

            adage_serializer = AdageCurrencyRateSerializer(data=adage_data)
            if not adage_serializer.is_valid():
                logger.error(f"Invalid ADAGE format: {adage_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Check alerts
            alerts = ExchangeRateAlert.objects.filter(
                base=base, target=target
            ).filter(
                Q(alert_type="above", threshold__lte=rate) |
                Q(alert_type="below", threshold__gte=rate)
            )

            for alert in alerts:
                send_alert_email(alert.email, base, target, rate, alert.alert_type, alert.threshold)
                logger.info(f"Alert triggered for {base}/{target} at {rate}, threshold {alert.threshold}")
                alert.delete()  # Delete the alert after sending the email

            return Response(adage_serializer.validated_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            return Response(
                {"detail": "Error connecting to external currency data provider."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            logger.error(f"Unexpected error in CurrencyRateView: {str(e)}")
            return Response(
                {"detail": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
