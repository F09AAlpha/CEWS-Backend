import logging
import pandas as pd
from .alpha_vantage import AlphaVantageService, AlphaVantageError

logger = logging.getLogger(__name__)


class AnomalyDetectionError(Exception):
    """Base exception for anomaly detection errors."""
    pass


class InsufficientDataError(AnomalyDetectionError):
    """Raised when there is not enough data for analysis."""
    pass


class ProcessingError(AnomalyDetectionError):
    """Raised when there is an error processing data."""
    pass


class AnomalyDetectionService:
    """Service for detecting anomalies in exchange rate data."""

    def __init__(self, base_currency, target_currency, analysis_period_days=30,
                 z_score_threshold=2.0, alpha_vantage_service=None):
        """Initialize the anomaly detection service."""
        self.base_currency = base_currency.upper()
        self.target_currency = target_currency.upper()
        self.analysis_period_days = analysis_period_days
        self.z_score_threshold = z_score_threshold

        self.alpha_vantage = alpha_vantage_service or AlphaVantageService()

    def get_exchange_rates(self):
        """Get exchange rates from Alpha Vantage API."""
        try:
            # Get exchange rates from Alpha Vantage
            df = self.alpha_vantage.get_exchange_rates(
                self.base_currency,
                self.target_currency,
                days=self.analysis_period_days
            )

            # Ensure there's enough data for analysis
            if len(df) < 10:  # minimum
                logger.warning(f"Insufficient data for analysis: {len(df)} records")
                raise InsufficientDataError(
                    f"Not enough data for analysis. Found {len(df)} records, need at least 10."
                )

            # Extract needed columns
            result_df = df[['date', 'close']].copy()
            result_df = result_df.rename(columns={'close': 'rate'})

            return result_df

        except AlphaVantageError as e:
            logger.error(f"Alpha Vantage API error: {str(e)}")
            raise  # Re-raise Alpha Vantage errors

        except Exception as e:
            logger.error(f"Error retrieving exchange rate data: {str(e)}")
            raise ProcessingError(f"Error retrieving exchange rate data: {str(e)}")

    def detect_anomalies(self):
        """Detect anomalies in exchange rate data."""
        try:
            # Get exchange rates
            df = self.get_exchange_rates()

            # Calculate percent change
            df['prev_rate'] = df['rate'].shift(1)
            df['percent_change'] = ((df['rate'] - df['prev_rate']) / df['prev_rate'] * 100).round(2)

            # Calculate z-scores
            rate_mean = df['rate'].mean()
            rate_std = df['rate'].std()
            if rate_std == 0:
                # Handle case where all rates are the same
                df['z_score'] = 0
            else:
                df['z_score'] = ((df['rate'] - rate_mean) / rate_std).round(2)

            # Identify anomalies (absolute z-score above threshold)
            df['is_anomaly'] = df['z_score'].abs() > self.z_score_threshold
            anomalies = df[df['is_anomaly']]

            # Format anomaly points for API response
            anomaly_points = []
            for _, row in anomalies.iterrows():
                # Handle NaN values for first day (no percent change)
                percent_change = row['percent_change']
                if pd.isna(percent_change):
                    percent_change = 0.0

                anomaly_points.append({
                    'timestamp': row['date'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'rate': float(row['rate']),
                    'z_score': float(row['z_score']),
                    'percent_change': float(percent_change)
                })

            # Create response in standard format
            result = {
                'base': self.base_currency,
                'target': self.target_currency,
                'anomaly_count': len(anomalies),
                'analysis_period_days': self.analysis_period_days,
                'anomaly_points': anomaly_points
            }

            return result

        except (InsufficientDataError, AlphaVantageError):
            # Re-raise these specific exceptions
            raise

        except Exception as e:
            logger.error(f"Error processing data for anomaly detection: {str(e)}")
            raise ProcessingError(f"Error processing data for anomaly detection: {str(e)}")
