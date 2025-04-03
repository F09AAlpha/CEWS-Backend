import pandas as pd
import numpy as np
import logging
import uuid
from datetime import timedelta
from django.db import transaction
from django.utils import timezone

from myapp.Models.predictionModel import CurrencyPrediction
from myapp.Service.alpha_vantage import AlphaVantageService
from myapp.Service.correlationService import CorrelationService
from myapp.Service.anomalyDetectionService import AnomalyDetectionService

# Set up logging
logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making currency exchange rate predictions using
    an advanced statistical model that incorporates correlation data.
    """

    # Model configuration
    DEFAULT_CONTEXT_LENGTH = 128
    DEFAULT_FORECAST_HORIZON = 7

    def __init__(self):
        """Initialize the prediction service."""
        self.alpha_vantage_service = AlphaVantageService()
        self.correlation_service = CorrelationService()

    def get_latest_prediction(self, base_currency, target_currency, horizon_days=None):
        """
        Get the latest prediction for a currency pair.

        Args:
            base_currency (str): Base currency code
            target_currency (str): Target currency code
            horizon_days (int, optional): Forecast horizon in days. Default is None (use latest).

        Returns:
            CurrencyPrediction: The latest prediction, or None if not found
        """
        try:
            # Standardize currency codes
            base_currency = base_currency.upper()
            target_currency = target_currency.upper()

            # Query parameters
            query_params = {
                'base_currency': base_currency,
                'target_currency': target_currency,
            }

            # Add horizon days if specified
            if horizon_days is not None:
                query_params['forecast_horizon'] = horizon_days

            # Get the latest prediction
            latest_prediction = CurrencyPrediction.objects.filter(
                **query_params
            ).order_by('-prediction_date').first()

            return latest_prediction

        except Exception as e:
            logger.error(f"Error retrieving latest prediction: {str(e)}")
            return None

    @transaction.atomic
    def create_prediction(self, base_currency, target_currency, horizon_days=None, refresh=False):
        """
        Create a new prediction for a currency pair.

        Args:
            base_currency (str): Base currency code
            target_currency (str): Target currency code
            horizon_days (int, optional): Forecast horizon in days. Default is DEFAULT_FORECAST_HORIZON.
            refresh (bool): Whether to refresh existing predictions. Default is False.

        Returns:
            CurrencyPrediction: The created prediction
        """
        try:
            # Standardize currency codes
            base_currency = base_currency.upper()
            target_currency = target_currency.upper()

            # Set default horizon if not specified
            if horizon_days is None:
                horizon_days = self.DEFAULT_FORECAST_HORIZON

            # If not refreshing, check for existing prediction
            if not refresh:
                existing_prediction = self.get_latest_prediction(
                    base_currency, target_currency, horizon_days
                )
                # If we have a recent prediction (less than 24 hours old), return it
                if existing_prediction and (timezone.now() - existing_prediction.prediction_date).total_seconds() < 86400:
                    logger.info(f"Using existing prediction for {base_currency}/{target_currency}")
                    return existing_prediction

            # Get historical exchange rate data
            lookback_days = max(self.DEFAULT_CONTEXT_LENGTH, horizon_days * 3)
            exchange_df = self.alpha_vantage_service.get_exchange_rates(
                base_currency, target_currency, days=lookback_days
            )

            if exchange_df.empty:
                logger.error(f"No exchange rate data available for {base_currency}/{target_currency}")
                raise ValueError(f"No exchange rate data available for {base_currency}/{target_currency}")

            # Get current rate (most recent closing price)
            current_rate = float(exchange_df.iloc[-1]['close'])

            # Get correlation data if available
            correlation_result = None
            try:
                correlation_result = self.correlation_service.get_latest_correlation(
                    base_currency, target_currency
                )
                logger.info(f"Found correlation data for {base_currency}/{target_currency}")
            except Exception as e:
                logger.warning(f"No correlation data available: {str(e)}")

            # Make the prediction
            prediction_results = self._predict_with_statistical_model(
                exchange_df, correlation_result, horizon_days
            )
            logger.info(f"Generated prediction using Advanced Statistical Model for {base_currency}/{target_currency}")

            # Calculate change percent from current rate to first prediction
            first_prediction = list(prediction_results['mean_predictions'].values())[0]
            change_percent = ((first_prediction - current_rate) / current_rate) * 100

            # Update prediction results with current rate and change percent
            prediction_results['current_rate'] = current_rate
            prediction_results['change_percent'] = change_percent

            # Create and save the prediction
            prediction = CurrencyPrediction(
                base_currency=base_currency,
                target_currency=target_currency,
                forecast_horizon=horizon_days,
                current_rate=current_rate,
                change_percent=change_percent,
                mean_predictions=prediction_results['mean_predictions'],
                lower_bound=prediction_results['lower_bound'],
                upper_bound=prediction_results['upper_bound'],
                model_version=prediction_results['model_version'],
                confidence_score=prediction_results['confidence_score'],
                input_data_range=prediction_results['input_data_range'],
                used_correlation_data=prediction_results['used_correlation_data'],
                used_news_sentiment=prediction_results['used_news_sentiment'],
                used_economic_indicators=prediction_results['used_economic_indicators'],
                used_anomaly_detection=prediction_results.get('used_anomaly_detection', False)
            )
            prediction.save()

            logger.info(f"Created new prediction for {base_currency}/{target_currency} with horizon {horizon_days} days")
            return prediction

        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            raise

    def _predict_with_statistical_model(self, exchange_df, correlation_result, horizon_days):
        """
        Make predictions using advanced statistical methods with correlation data.

        Args:
            exchange_df (pandas.DataFrame): Historical exchange rate data
            correlation_result (CorrelationResult): Correlation analysis result
            horizon_days (int): Forecast horizon in days

        Returns:
            dict: Prediction results
        """
        try:
            # Ensure data is properly sorted
            exchange_df['date'] = pd.to_datetime(exchange_df['date'])
            exchange_df = exchange_df.sort_values('date')

            # Extract close prices for forecasting
            close_series = exchange_df['close'].values

            # Simple random walk with drift for prediction
            # Get the last value
            last_value = close_series[-1]

            # Track if we're using correlation data effectively
            correlation_factor_weights = {}
            correlation_factor_used = False

            # Generate predictions using a random walk with drift
            # Calculate basic drift from recent history (last 30 days)
            base_mean_drift = np.mean(np.diff(close_series[-30:]))

            # Enhanced drift calculation incorporating correlation data if available
            mean_drift = base_mean_drift

            # Try to enhance drift with correlation data
            if correlation_result is not None:
                try:
                    # Get top influencing factors
                    if hasattr(correlation_result, 'top_influencing_factors'):
                        top_factors = correlation_result.top_influencing_factors
                        if top_factors and len(top_factors) > 0:
                            # Create a weighted adjustment based on each factor
                            # Stronger correlations get more weight
                            factor_weights = {}
                            total_weight = 0

                            # Process each factor's correlation and determine its weight
                            for factor in top_factors:
                                factor_name = factor.get('factor', '')
                                factor_type = factor.get('type', '')
                                factor_corr = factor.get('correlation', 0)
                                if isinstance(factor_corr, (int, float)) and not np.isnan(factor_corr):
                                    # Weight is proportional to correlation strength
                                    weight = abs(factor_corr) * 2  # Double the weight of strong correlations
                                    factor_weights[factor_name] = {
                                        'correlation': factor_corr,
                                        'weight': weight,
                                        'type': factor_type
                                    }
                                    total_weight += weight

                            # If we have valid factors, adjust the drift
                            if total_weight > 0:
                                correlation_factor_used = True
                                drift_adjustment = 0

                                # Apply each factor's influence to the drift
                                for factor_name, details in factor_weights.items():
                                    factor_contribution = (details['correlation'] * details['weight'] / total_weight)
                                    drift_adjustment += factor_contribution
                                    correlation_factor_weights[factor_name] = details['weight'] / total_weight

                                # Adjust drift with correlation data - apply a stronger influence (up to 20%)
                                # The more influential factors we have, the stronger the adjustment
                                adjustment_strength = min(0.20, 0.05 * len(factor_weights))
                                mean_drift = base_mean_drift * (1 + (drift_adjustment * adjustment_strength))
                                logger.info(f"Enhanced drift calculation using correlation data: {mean_drift:.6f} " +
                                            f"(base: {base_mean_drift:.6f}, adjustment: {drift_adjustment:.6f})")
                except Exception as e:
                    logger.warning(f"Could not apply enhanced drift using correlation data: {str(e)}")

            # Make predictions with updated drift
            predictions_mean = [last_value + mean_drift * (i+1) for i in range(horizon_days)]

            # Add volatility for confidence intervals
            volatility = np.std(np.diff(close_series[-30:]))
            z_score = 1.645  # 90% confidence interval

            # Apply anomaly detection to improve prediction quality
            try:
                # Initialize anomaly detection service
                anomaly_detector = AnomalyDetectionService(
                    base_currency=exchange_df.attrs.get('base_currency', 'Unknown'),
                    target_currency=exchange_df.attrs.get('target_currency', 'Unknown'),
                    analysis_period_days=min(90, len(exchange_df)),  # Use available data up to 90 days
                    z_score_threshold=2.0
                )

                # Detect anomalies
                anomaly_result = anomaly_detector.detect_anomalies()
                anomaly_count = anomaly_result['anomaly_count']

                # Create a dictionary of anomaly points for easy lookup
                anomaly_points = {}
                for point in anomaly_result['anomaly_points']:
                    date_str = point['timestamp'].split('T')[0]  # Extract date part
                    anomaly_points[date_str] = point

                # Log using the currencies from the anomaly result
                base = anomaly_result['base']
                target = anomaly_result['target']
                logger.info(f"Found {anomaly_count} anomalies in historical data for {base}/{target}")

                # Adjust the data to reduce the impact of anomalies on prediction
                if anomaly_count > 0:
                    # Convert dates to string format for comparison
                    exchange_df['date_str'] = exchange_df['date'].dt.strftime('%Y-%m-%d')

                    # Identify anomalous data points
                    exchange_df['is_anomaly'] = exchange_df['date_str'].apply(
                        lambda date: date in anomaly_points
                    )

                    # Use non-anomalous data for drift calculation if we have sufficient data
                    normal_data = exchange_df[~exchange_df['is_anomaly']]
                    if len(normal_data) >= 15:  # Ensure we have enough normal data points
                        # Recalculate drift using only normal data
                        normal_close = normal_data['close'].values
                        mean_drift = np.mean(np.diff(normal_close[-30:] if len(normal_close) > 30 else normal_close))
                        logger.info(f"Adjusted drift calculation using filtered data: {mean_drift}")

                        # Recalculate volatility using normal data
                        volatility = np.std(np.diff(normal_close[-30:] if len(normal_close) > 30 else normal_close))
                        logger.info(f"Adjusted volatility calculation: {volatility}")

                        # Update predictions with adjusted parameters
                        predictions_mean = [last_value + mean_drift * (i+1) for i in range(horizon_days)]
                        predictions_lower = [mean - z_score * volatility * np.sqrt(i+1)
                                             for i, mean in enumerate(predictions_mean)]
                        predictions_upper = [mean + z_score * volatility * np.sqrt(i+1)
                                             for i, mean in enumerate(predictions_mean)]
            except Exception as e:
                logger.warning(f"Error applying anomaly detection: {str(e)}. Using original prediction model.")
                # If anomaly detection fails, we'll use the original prediction values
                # Calculate prediction intervals
                predictions_lower = [mean - z_score * volatility * np.sqrt(i+1)
                                     for i, mean in enumerate(predictions_mean)]
                predictions_upper = [mean + z_score * volatility * np.sqrt(i+1)
                                     for i, mean in enumerate(predictions_mean)]

            # Prepare dates for the predictions (starting from tomorrow)
            last_date = exchange_df['date'].iloc[-1]
            prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                                for i in range(horizon_days)]

            # Create prediction result
            used_correlation = correlation_result is not None
            used_correlation_factors = correlation_factor_used
            used_anomalies = 'anomaly_count' in locals() and anomaly_count > 0

            # Format dates for input range using actual dates from the exchange rate data
            first_date = exchange_df['date'].iloc[0]
            last_date = exchange_df['date'].iloc[-1]
            input_range = f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}"

            # Default values for using different data sources
            used_news = False
            used_econ = False

            # Format the prediction results with proper dates
            mean_predictions = {date: float(value) for date, value in zip(prediction_dates, predictions_mean)}
            lower_bound = {date: float(value) for date, value in zip(prediction_dates, predictions_lower)}
            upper_bound = {date: float(value) for date, value in zip(prediction_dates, predictions_upper)}

            # Default correlation strength
            correlation_strength = 0.0
            correlation_confidence = 0.0  # Initialize correlation confidence

            # Track which correlation factors were used and their impact
            used_correlation_details = {}

            # Adjust predictions if correlation data is available
            if used_correlation:
                try:
                    # Check for news correlation data
                    if hasattr(correlation_result, 'exchange_news_correlation'):
                        used_news = bool(correlation_result.exchange_news_correlation)

                        # Include details of news correlations
                        if used_news:
                            used_correlation_details['news'] = {
                                'count': len(correlation_result.exchange_news_correlation),
                                'avg_strength': np.mean([abs(val) for val in
                                                         correlation_result.exchange_news_correlation.values()])
                            }

                    # Check for economic correlation data
                    if hasattr(correlation_result, 'exchange_economic_correlation'):
                        econ_corr = correlation_result.exchange_economic_correlation
                        # Handle both dictionary and scalar correlation values
                        if isinstance(econ_corr, dict):
                            used_econ = bool(econ_corr) and len(econ_corr) > 0
                        else:
                            used_econ = bool(econ_corr) and econ_corr != 0

                        # Include details of economic indicator correlations
                        if used_econ:
                            if isinstance(econ_corr, dict):
                                used_correlation_details['economic'] = {
                                    'count': len(econ_corr),
                                    'avg_strength': np.mean([abs(val) for val in econ_corr.values()])
                                }
                            else:
                                used_correlation_details['economic'] = {
                                    'count': 1,
                                    'avg_strength': abs(econ_corr)
                                }

                    # Ensure economic indicators are used if there are any factors of type 'economic'
                    if correlation_factor_used and factor_weights:
                        for factor_name, details in factor_weights.items():
                            if details.get('type') == 'economic':
                                used_econ = True
                                break

                    # Apply correlation confidence from the correlation result if available
                    if hasattr(correlation_result, 'confidence_score'):
                        correlation_confidence = correlation_result.confidence_score
                        logger.info(f"Using correlation confidence score: {correlation_confidence:.2f}")

                    # Store all factor weights for detailed reporting
                    if correlation_factor_weights:
                        used_correlation_details['factors'] = correlation_factor_weights

                        # Get the sum of absolute correlation values to use as correlation strength
                        weighted_corr_strength = sum(abs(details['correlation']) * details['weight']
                                                     for details in factor_weights.values())
                        correlation_strength = min(1.0, weighted_corr_strength)

                    # If we already did factor-based prediction adjustments, skip this
                    # This is now just a fallback for older correlation results
                    if not correlation_factor_used and hasattr(correlation_result, 'top_influencing_factors'):
                        top_factors = correlation_result.top_influencing_factors
                        if top_factors:
                            # Get average correlation coefficient
                            coeffs = [factor.get('correlation', 0) for factor in top_factors]
                            avg_coeff = sum(coeffs) / len(coeffs) if coeffs else 0

                            # Store correlation strength for confidence calculation
                            correlation_strength = abs(avg_coeff)

                            # Apply small adjustment based on correlation strength
                            adjustment = avg_coeff * 0.05  # 5% influence from correlation

                            # Adjust predictions
                            for date in mean_predictions:
                                mean_predictions[date] = mean_predictions[date] * (1 + adjustment)
                                lower_bound[date] = lower_bound[date] * (1 + adjustment)
                                upper_bound[date] = upper_bound[date] * (1 + adjustment)
                except Exception as e:
                    logger.warning(f"Error applying correlation adjustment: {str(e)}")

            # Calculate confidence score dynamically based on multiple factors

            # 1. Data Quality Factor (35%): Based on amount of historical data
            data_points = len(exchange_df)
            data_quality_score = min(100, data_points / 2)  # Each 2 data points = 1% up to 100%

            # 2. Volatility Factor (15%): Inverse of volatility (more volatile = less confident)
            # Normalize volatility relative to the price
            relative_volatility = volatility / last_value
            volatility_score = 100 * max(0, 1 - (relative_volatility * 25))  # Lower volatility penalty

            # 3. Correlation Factor (25%): Based on strength of correlations and correlation confidence
            # Blend correlation strength with the correlation service confidence score if available
            if correlation_confidence > 0:
                # Use a weighted average of correlation strength and confidence
                correlation_score = (correlation_strength * 100 * 0.4) + (correlation_confidence * 0.6)
            else:
                correlation_score = correlation_strength * 100

            # Apply a minimum correlation score if we used correlation data at all
            if used_correlation_factors:
                correlation_score = max(correlation_score, 30)  # Minimum 30 points if factors were used
            elif used_correlation:
                correlation_score = max(correlation_score, 20)  # Minimum 20 points if correlation was used

            # 4. Horizon Factor (10%): Decreases as prediction horizon increases
            horizon_factor = max(0, 100 - (horizon_days * 1))  # Subtract 1% per day

            # 5. Anomaly Factor (15%): New factor based on anomaly detection
            anomaly_factor = 0
            if 'anomaly_count' in locals() and anomaly_count > 0:
                # Higher score if anomalies were successfully detected and handled
                anomaly_ratio = min(1.0, anomaly_count / max(1, len(exchange_df)))
                anomaly_factor = 100 * (1 - anomaly_ratio)  # Lower ratio = higher score
            elif 'anomaly_count' in locals():
                # Even with no anomalies, give some credit for checking
                anomaly_factor = 75  # Good score for clean data

            # Weighted combination of all factors
            confidence_score = (
                (data_quality_score * 0.35) +
                (volatility_score * 0.15) +
                (correlation_score * 0.25) +
                (horizon_factor * 0.10) +
                (anomaly_factor * 0.15)
            )

            # Ensure score is between 0 and 100
            confidence_score = max(0, min(100, confidence_score))

            # Log confidence calculation
            logger.info(
                f"Confidence score calculation: Data Quality: {data_quality_score:.1f}, " +
                f"Volatility: {volatility_score:.1f}, Correlation: {correlation_score:.1f}, " +
                f"Horizon: {horizon_factor:.1f}, Anomaly: {anomaly_factor:.1f} => Final: {confidence_score:.1f}"
            )

            return {
                'mean_predictions': mean_predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_version': "Statistical Model v2",
                'confidence_score': round(confidence_score, 1),  # Rounded to 1 decimal place
                'input_data_range': input_range,
                'used_correlation_data': used_correlation,
                'used_correlation_details': used_correlation_details,
                'used_news_sentiment': used_news,
                'used_economic_indicators': used_econ,
                'used_anomaly_detection': used_anomalies
            }

        except Exception as e:
            logger.error(f"Error making prediction with statistical model: {str(e)}")
            raise

    def format_adage_response(self, prediction):
        """
        Format the prediction response according to ADAGE 3.0 standard.

        Args:
            prediction (CurrencyPrediction): The prediction to format

        Returns:
            dict: ADAGE 3.0 formatted response
        """
        try:
            # Generate unique dataset ID
            dataset_id = f"prediction_{prediction.base_currency}_{prediction.target_currency}_{uuid.uuid4().hex[:8]}"

            # Create time object for dataset
            current_time = timezone.now()

            # Extract prediction values
            prediction_values = []
            for date, value in prediction.mean_predictions.items():
                prediction_values.append({
                    "timestamp": date,
                    "mean": value,
                    "lower_bound": prediction.lower_bound[date],
                    "upper_bound": prediction.upper_bound[date]
                })

            # Create influencing factors
            influencing_factors = []
            if prediction.used_correlation_data:
                influencing_factors.append({
                    "factor_name": "Correlation Analysis",
                    "impact_level": "high",
                    "used_in_prediction": True
                })
            if prediction.used_news_sentiment:
                influencing_factors.append({
                    "factor_name": "News Sentiment",
                    "impact_level": "medium",
                    "used_in_prediction": True
                })
            if prediction.used_economic_indicators:
                influencing_factors.append({
                    "factor_name": "Economic Indicators",
                    "impact_level": "medium",
                    "used_in_prediction": True
                })
            if hasattr(prediction, 'used_anomaly_detection') and prediction.used_anomaly_detection:
                influencing_factors.append({
                    "factor_name": "Anomaly Detection",
                    "impact_level": "high",
                    "used_in_prediction": True
                })

            # Create ADAGE 3.0 response
            response = {
                "data_source": "Currency Exchange Warning System",
                "dataset_type": "currency_exchange_prediction",
                "dataset_id": dataset_id,
                "time_object": {
                    "timestamp": current_time.isoformat(),
                    "timezone": "UTC"
                },
                "events": [
                    {
                        "time_object": {
                            "timestamp": prediction.prediction_date.isoformat(),
                            "horizon_days": prediction.forecast_horizon,
                            "timezone": "UTC"
                        },
                        "event_type": "exchange_rate_forecast",
                        "attributes": {
                            "base_currency": prediction.base_currency,
                            "target_currency": prediction.target_currency,
                            "current_rate": prediction.current_rate,
                            "change_percent": prediction.change_percent,
                            "confidence_score": prediction.confidence_score,
                            "model_version": prediction.model_version,
                            "input_data_range": prediction.input_data_range,
                            "influencing_factors": influencing_factors,
                            "prediction_values": prediction_values
                        }
                    }
                ]
            }

            return response

        except Exception as e:
            logger.error(f"Error formatting ADAGE response: {str(e)}")
            raise
