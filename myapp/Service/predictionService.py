import pandas as pd
import numpy as np
import logging
import uuid
from datetime import timedelta
from django.db import transaction
from django.utils import timezone
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import traceback
import time
from scipy import stats

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
    def create_prediction(
        self, base_currency, target_currency, horizon_days=None,
        refresh=False, use_arima=True, confidence_level=80
    ):
        """
        Create a new prediction for a currency pair.

        Args:
            base_currency (str): Base currency code (3 letters)
            target_currency (str): Target currency code (3 letters)
            horizon_days (int, optional): Forecast horizon in days. Default is DEFAULT_FORECAST_HORIZON.
            refresh (bool): Whether to refresh existing predictions. Default is False.
            use_arima (bool): Whether to use ARIMA modeling. Default is True (recommended).
            confidence_level (int): Confidence level for prediction intervals (50-99).
                Default is 80 (recommended for realistic bounds).
                Lower values (70-80) provide tighter, more precise bounds.
                Higher values (90-99) provide wider bounds with more statistical confidence.

        Returns:
            CurrencyPrediction: The created prediction

        Raises:
            ValueError: If currency codes are invalid or exchange rate data is not available
            AlphaVantageError: If there are issues with the Alpha Vantage API
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

            # Get historical exchange rate data - using data from Alpha Vantage
            try:
                lookback_days = max(self.DEFAULT_CONTEXT_LENGTH, horizon_days * 3)
                exchange_df = self.alpha_vantage_service.get_exchange_rates(
                    base_currency, target_currency, days=lookback_days
                )
            except Exception as e:
                logger.error(f"Failed to get real exchange rate data: {str(e)}")
                raise ValueError(f"Could not obtain exchange rate data for {base_currency}/{target_currency}: {str(e)}")

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

            # Try ARIMA model first
            prediction_results = None
            model_used = "Statistical"

            if use_arima:
                try:
                    logger.info(f"Attempting ARIMA model for {base_currency}/{target_currency}")
                    prediction_results = self._predict_with_arima_model(
                        exchange_df, correlation_result, horizon_days, confidence_level
                    )

                    if prediction_results is not None:
                        model_used = "ARIMA"
                        logger.info(f"Successfully used ARIMA model for {base_currency}/{target_currency}")
                    else:
                        logger.warning(
                            f"ARIMA model returned None for {base_currency}/{target_currency}, "
                            f"falling back to statistical model"
                        )
                except Exception as e:
                    logger.error(f"Error using ARIMA model: {str(e)}. Traceback: {traceback.format_exc()}")
                    logger.warning("Falling back to statistical model due to ARIMA error")
            else:
                logger.info(f"ARIMA model not requested for {base_currency}/{target_currency}")

            # Fall back to statistical model if ARIMA wasn't used or failed
            if prediction_results is None:
                prediction_results = self._predict_with_statistical_model(
                    exchange_df, correlation_result, horizon_days, confidence_level
                )
                logger.info(f"Generated prediction using Statistical Model for {base_currency}/{target_currency}")

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
                used_anomaly_detection=prediction_results.get('used_anomaly_detection', False),
                mean_square_error=prediction_results.get('mean_square_error'),
                root_mean_square_error=prediction_results.get('root_mean_square_error'),
                mean_absolute_error=prediction_results.get('mean_absolute_error')
            )
            prediction.save()

            logger.info(
                f"Created new prediction for {base_currency}/{target_currency} "
                f"using {model_used} model with horizon {horizon_days} days"
            )
            return prediction

        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            raise

    def _predict_with_statistical_model(self, exchange_df, correlation_result, horizon_days, confidence_level):
        """
        Make predictions using advanced statistical methods with correlation data.

        Args:
            exchange_df (pandas.DataFrame): Historical exchange rate data
            correlation_result (CorrelationResult): Correlation analysis result
            horizon_days (int): Forecast horizon in days
            confidence_level (int): Confidence level for prediction intervals (50-99)

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

                                # Adjust drift with correlation data
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
            # Calculate historical daily changes as percentage
            if len(close_series) > 1:
                close_pct_changes = np.diff(close_series) / close_series[:-1]
                daily_volatility = np.std(close_pct_changes)
            else:
                # Fallback if we have only one data point - worst case scenario
                daily_volatility = 0.005  # Default 0.5% daily volatility

            # Convert confidence level to z-score
            if confidence_level >= 99:
                z_score = 2.576
            elif confidence_level >= 95:
                z_score = 1.96
            elif confidence_level >= 90:
                z_score = 1.645
            elif confidence_level >= 80:
                z_score = 1.282
            elif confidence_level >= 70:
                z_score = 1.036
            elif confidence_level >= 60:
                z_score = 0.842
            else:  # 50% or lower
                z_score = 0.674

            logger.info(f"Using confidence level: {confidence_level}% (z-score: {z_score})")
            forecast_bounds = []
            for i in range(horizon_days):
                # Volatility grows with square root of time horizon (standard finance approach)
                time_factor = np.sqrt(i + 1)
                pct_bound = daily_volatility * z_score * time_factor

                # Cap the percentage bound at a reasonable level (8%)
                pct_bound = min(pct_bound, 0.08)

                forecast_bounds.append(pct_bound)

            # Calculate actual bounds
            predictions_lower = [pred * (1 - bound) for pred, bound in zip(predictions_mean, forecast_bounds)]
            predictions_upper = [pred * (1 + bound) for pred, bound in zip(predictions_mean, forecast_bounds)]

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

                        # Calculate volatility using normal data
                        normal_pct_changes = np.diff(normal_close) / normal_close[:-1]
                        daily_volatility = np.std(normal_pct_changes)
                        logger.info(f"Adjusted volatility calculation: {daily_volatility}")

                        # Update predictions with adjusted parameters
                        predictions_mean = [last_value + mean_drift * (i+1) for i in range(horizon_days)]

                        # Recalculate bounds with new volatility
                        forecast_bounds = []
                        for i in range(horizon_days):
                            time_factor = np.sqrt(i + 1)
                            pct_bound = daily_volatility * z_score * time_factor
                            pct_bound = min(pct_bound, 0.08)  # Cap at 8%
                            forecast_bounds.append(pct_bound)

                        predictions_lower = [pred * (1 - bound) for pred, bound in zip(predictions_mean, forecast_bounds)]
                        predictions_upper = [pred * (1 + bound) for pred, bound in zip(predictions_mean, forecast_bounds)]
            except Exception as e:
                logger.warning(f"Error applying anomaly detection: {str(e)}. Using original prediction model.")
                # If anomaly detection fails, we'll use the original prediction values
                # Calculate prediction intervals based on original forecast_bounds
                predictions_lower = [pred * (1 - bound) for pred, bound in zip(predictions_mean, forecast_bounds)]
                predictions_upper = [pred * (1 + bound) for pred, bound in zip(predictions_mean, forecast_bounds)]

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

            # Final sanity check on prediction values
            for date, value in mean_predictions.items():
                if value <= 0:
                    logger.warning(
                        f"Negative prediction value ({value}) detected for {date} after all fixes. "
                        f"Applying emergency correction."
                    )
                    mean_predictions[date] = max(0.001, last_value * 0.01)
                    lower_bound[date] = max(0.0005, mean_predictions[date] * 0.5)
                    upper_bound[date] = mean_predictions[date] * 1.5

                # Check for unrealistically large values
                if value > last_value * 10:
                    logger.warning(f"Extreme prediction value ({value}) detected for {date}. Capping at 10x current rate.")
                    mean_predictions[date] = last_value * 10
                    upper_bound[date] = min(upper_bound[date], mean_predictions[date] * 1.2)
                    lower_bound[date] = min(lower_bound[date], mean_predictions[date] * 0.8)

            # Calculate the change percent from current rate to first prediction point
            first_prediction_date = prediction_dates[0]
            first_prediction_value = mean_predictions[first_prediction_date]
            change_percent = ((first_prediction_value - last_value) / last_value) * 100

            # Check for unrealistic change percent
            if abs(change_percent) > 50:
                logger.warning(f"Unrealistic change percent: {change_percent:.2f}%. Moderating first prediction.")
                # Moderate the first prediction to be more reasonable
                moderated_value = last_value * (1 + (np.sign(change_percent) * 0.2))  # Cap at 20% change
                mean_predictions[first_prediction_date] = moderated_value

                # Recalculate subsequent values to maintain trend but with moderated start
                original_first = first_prediction_value
                scale_factor = moderated_value / original_first if original_first != 0 else 1.0

                for i, date in enumerate(prediction_dates[1:], 1):
                    # Scale prediction while preserving the trend direction
                    original = mean_predictions[date]
                    trend_factor = (original - original_first) / original_first if original_first != 0 else 0
                    mean_predictions[date] = moderated_value * (1 + trend_factor)

                    # Also adjust bounds
                    if lower_bound[date] > 0:  # Only scale positive bounds
                        lower_bound[date] *= scale_factor
                    upper_bound[date] *= scale_factor

                # Recalculate change percent
                change_percent = ((mean_predictions[first_prediction_date] - last_value) / last_value) * 100

            # Default correlation strength
            correlation_strength = 0.0
            correlation_confidence = 0.0

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

                    # Fallback for older correlation results
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
            relative_volatility = daily_volatility / last_value
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

            confidence_score = min(85, confidence_score)

            # Log confidence calculation
            logger.info(
                f"Confidence score calculation: Data Quality: {data_quality_score:.1f}, "
                f"Volatility: {volatility_score:.1f}, Correlation: {correlation_score:.1f}, "
                f"Horizon: {horizon_factor:.1f}, Anomaly: {anomaly_factor:.1f} => Final: {confidence_score:.1f}"
            )

            # Calculate error metrics
            # For backtest, use the most recent data points
            # This simulates how our model would have performed if we used it in the past
            backtest_window = min(30, max(5, len(exchange_df) // 5))  # Use at least 5 points, up to 30

            # Always calculate error metrics with whatever data we have
            logger.info(f"Calculating error metrics with {len(exchange_df)} data points, " +
                        f"backtest window of {backtest_window}")

            # Make sure we have at least 3 data points for minimal backtesting
            if len(exchange_df) >= 3:
                # Split data for backtest - use at least 60% for training if possible
                train_size = max(2, len(exchange_df) - backtest_window)
                train_data = close_series[:train_size]
                test_data = close_series[train_size:]

                logger.info(f"Backtesting with {len(train_data)} training points and {len(test_data)} test points")

                # Calculate drift from training data - use at least the last 2 points
                if len(train_data) > 1:
                    train_window = min(30, len(train_data))
                    backtest_drift = np.mean(np.diff(train_data[-train_window:]))
                else:
                    backtest_drift = 0

                # Make predictions for test period
                backtest_predictions = []
                for i in range(len(test_data)):
                    if i == 0:
                        pred = train_data[-1] + backtest_drift
                    else:
                        pred = backtest_predictions[-1] + backtest_drift
                    backtest_predictions.append(pred)

                # Calculate error metrics
                errors = np.array(backtest_predictions) - np.array(test_data)
                mean_square_error = np.mean(errors ** 2)
                root_mean_square_error = np.sqrt(mean_square_error)
                mean_absolute_error = np.mean(np.abs(errors))

                # Normalize errors relative to the average price
                avg_price = np.mean(close_series)
                norm_mse = mean_square_error / (avg_price ** 2)  # Normalized MSE
                norm_rmse = root_mean_square_error / avg_price   # Normalized RMSE as percentage
                norm_mae = mean_absolute_error / avg_price       # Normalized MAE as percentage

                logger.info(
                    f"Backtest error metrics: MSE={mean_square_error:.6f}, "
                    f"RMSE={root_mean_square_error:.6f}, MAE={mean_absolute_error:.6f}"
                )
                logger.info(
                    f"Normalized error metrics: MSE={norm_mse:.6f}, "
                    f"RMSE={norm_rmse:.6f} ({norm_rmse*100:.2f}%), MAE={norm_mae:.6f} ({norm_mae*100:.2f}%)"
                )
            else:
                # Not enough data for backtest, but still provide error estimate based on volatility
                logger.warning("Insufficient data for proper backtest error calculation. Using volatility-based estimate.")

                # Calculate error metrics based on historical volatility
                daily_volatility = np.std(np.diff(close_series)) if len(close_series) > 1 else 0.001
                avg_price = np.mean(close_series)

                # Estimated errors based on volatility (empirical approximation)
                mean_square_error = daily_volatility ** 2  # Variance
                root_mean_square_error = daily_volatility
                mean_absolute_error = daily_volatility * 0.8  # Empirical relationship for normal-like distributions

                logger.info(
                    f"Volatility-based error metrics: MSE={mean_square_error:.6f}, "
                    f"RMSE={root_mean_square_error:.6f}, MAE={mean_absolute_error:.6f}"
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
                'used_anomaly_detection': used_anomalies,
                'mean_square_error': float(mean_square_error) if mean_square_error is not None else None,
                'root_mean_square_error': float(root_mean_square_error) if root_mean_square_error is not None else None,
                'mean_absolute_error': float(mean_absolute_error) if mean_absolute_error is not None else None
            }

        except Exception as e:
            logger.error(f"Error making prediction with statistical model: {str(e)}")
            raise

    def _predict_with_arima_model(self, exchange_df, correlation_result, horizon_days, confidence_level):
        """
        Generate a prediction using an ARIMA model.

        Args:
            exchange_df: DataFrame with historical exchange rates
            correlation_result: Correlation result object or None
            horizon_days: Forecast horizon in days
            confidence_level: Confidence level for prediction intervals

        Returns:
            dict: Prediction results
        """
        try:
            original_df = exchange_df.copy()
            timestamp_start = time.time()
            logger.info(f"Starting ARIMA model prediction for horizon of {horizon_days} days")

            # Ensure data is sorted by date
            exchange_df = exchange_df.sort_values(by='date')

            # Convert 'date' to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(exchange_df['date']):
                exchange_df['date'] = pd.to_datetime(exchange_df['date'])

            # Validate data is sufficient
            required_data_points = max(60, horizon_days * 3)  # At least 60 days or 3x horizon
            if len(exchange_df) < required_data_points:
                logger.warning(f"Insufficient data for ARIMA model: {len(exchange_df)} points, need {required_data_points}")
                return None

            # Set date as index
            exchange_df = exchange_df.set_index('date')

            # Check for anomalies and clean data if necessary
            used_anomaly_detection = False
            if len(exchange_df) > 30:  # Only detect anomalies with enough data
                try:
                    # Use Z-score method for anomaly detection
                    z_scores = np.abs(stats.zscore(exchange_df['close']))
                    anomalies = (z_scores > 3.5)  # Points beyond 3.5 standard deviations

                    if anomalies.any():
                        used_anomaly_detection = True
                        anomaly_count = anomalies.sum()
                        logger.info(f"Detected {anomaly_count} anomalies in exchange rate data")

                        # Replace anomalies with interpolated values
                        clean_df = exchange_df.copy()
                        clean_df.loc[anomalies, 'close'] = np.nan
                        clean_df['close'] = clean_df['close'].interpolate(method='linear')
                        exchange_df = clean_df
                except Exception as e:
                    logger.warning(f"Error during anomaly detection: {str(e)}")

            # Apply log transformation to prevent negative values
            log_transform = True
            shift_amount = 0
            if log_transform:
                min_val = exchange_df['close'].min()
                if min_val <= 0:
                    shift_amount = abs(min_val) + 0.001
                    exchange_df['close'] = exchange_df['close'] + shift_amount
                    logger.info(f"Shifted data by {shift_amount} to ensure positive values for log transformation")

                # Apply log transformation
                exchange_df['log_close'] = np.log(exchange_df['close'])
                time_series = exchange_df['log_close']
                logger.info("Applied log transformation to exchange rate data")
            else:
                time_series = exchange_df['close']

            # Check for stationarity
            adf_result = adfuller(time_series.dropna())
            is_stationary = adf_result[1] < 0.05
            logger.info(f"ADF test p-value: {adf_result[1]:.6f}, stationary: {is_stationary}")

            # Automatic ARIMA parameter selection
            best_aic = float('inf')
            best_params = None
            best_model = None

            # Define parameter ranges for grid search
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)

            # Perform grid search with more parameter combinations if data is sufficient
            if len(exchange_df) >= 100:
                p_range = range(0, 4)
                d_range = range(0, 3)
                q_range = range(0, 4)

            # Limited grid search for ARIMA parameters
            max_failures = 0
            for p in p_range:
                for d in d_range:
                    for q in q_range:
                        try:
                            model = ARIMA(time_series, order=(p, d, q))
                            # Fit the model without specifying method parameter
                            model_fit = model.fit()

                            current_aic = model_fit.aic
                            if current_aic < best_aic:
                                best_aic = current_aic
                                best_params = (p, d, q)
                                best_model = model_fit
                                logger.debug(f"New best ARIMA model: {best_params} with AIC: {current_aic:.2f}")

                        except Exception as e:
                            max_failures += 1
                            if max_failures < 10:  # Only log the first few failures to avoid spam
                                logger.debug(f"ARIMA({p},{d},{q}) failed: {str(e)}")
                            # Skip failed models
                            continue

            # If no model was found, try with simplified parameters and different method
            if best_model is None:
                logger.warning("Grid search failed to find a suitable ARIMA model, trying fallback models")
                fallback_orders = [(1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

                for order in fallback_orders:
                    try:
                        logger.info(f"Trying fallback ARIMA{order}")
                        model = ARIMA(time_series, order=order)
                        model_fit = model.fit()
                        best_model = model_fit
                        best_params = order
                        best_aic = model_fit.aic
                        logger.info(f"Fallback ARIMA{order} successful")
                        break
                    except Exception as e:
                        logger.warning(f"Fallback ARIMA{order} failed: {str(e)}")

            if best_model is None:
                logger.error("All ARIMA models failed, including fallbacks")
                return None

            logger.info(f"Selected ARIMA order: {best_params} with AIC: {best_aic:.2f}")
            logger.info(f"Generating ARIMA forecast for {horizon_days} days with model order {best_params}")

            # Generate forecast
            forecast = best_model.get_forecast(steps=horizon_days)
            mean_forecast = forecast.predicted_mean

            # Get prediction intervals
            confidence_intervals = forecast.conf_int(alpha=(100-confidence_level)/100)

            # Preview forecast range to check for potential issues
            forecast_min = mean_forecast.min()
            forecast_max = mean_forecast.max()
            history_min = time_series.min()
            history_max = time_series.max()
            logger.info(
                f"Forecast range: {forecast_min:.4f} to {forecast_max:.4f} "
                f"(historical range: {history_min:.4f} to {history_max:.4f})"
            )

            # Convert back from log transform if used
            if log_transform:
                mean_forecast = np.exp(mean_forecast)
                confidence_intervals = np.exp(confidence_intervals)

                # If we shifted the data earlier, we need to shift back
                if shift_amount > 0:
                    mean_forecast = mean_forecast - shift_amount
                    confidence_intervals = confidence_intervals - shift_amount
                    logger.info(f"Shifted forecast back by {shift_amount} to restore original scale")

            # Check for negative values after transformation
            if mean_forecast.min() < 0:
                logger.warning(f"ARIMA forecast contains negative values: min={mean_forecast.min():.4f}. Applying correction.")

                # Set the shift to be slightly larger than the most negative value
                forecast_shift = abs(mean_forecast.min()) + 0.0001
                mean_forecast = mean_forecast + forecast_shift
                confidence_intervals = confidence_intervals + forecast_shift

                logger.info(f"Shifted forecast values by +{forecast_shift:.4f} to ensure positive rates")

            # Calculate daily volatility from historical data for bounds checking
            daily_returns = exchange_df['close'].pct_change().dropna()
            historical_volatility = daily_returns.std()

            # For high correlation use case, adjust the forecast slightly
            used_correlation_data = False
            correlation_factors = []
            correlation_adjustment = 0
            correlation_details = {}

            if correlation_result and hasattr(correlation_result, 'correlation_factors'):
                # Get correlation factors
                correlation_factors = correlation_result.correlation_factors

                if correlation_factors and len(correlation_factors) > 0:
                    used_correlation_data = True
                    logger.info(f"Using {len(correlation_factors)} correlation factors to adjust ARIMA predictions")

                    for factor in correlation_factors:
                        factor_name = factor.get('name', 'unknown')
                        factor_correlation = factor.get('correlation', 0)
                        factor_weight = factor.get('weight', 0)

                        logger.debug(
                            f"Factor '{factor_name}' has correlation {factor_correlation:.4f} "
                            f"and weight {factor_weight:.4f}"
                        )

                        # Accumulate correlation details for reporting
                        correlation_details[factor_name] = {
                            'correlation': factor_correlation,
                            'weight': factor_weight
                        }

                        # Apply a small adjustment based on correlation (more significant for longer horizons)
                        factor_adjustment = factor_correlation * factor_weight * 0.01 * min(5, horizon_days/7)
                        correlation_adjustment += factor_adjustment

                    # Apply the cumulative adjustment
                    if abs(correlation_adjustment) > 0.001:
                        # Scale the adjustment to be more conservative
                        safe_adjustment = np.clip(correlation_adjustment, -0.05, 0.05)

                        # Apply as a percentage change to each forecast value
                        mean_forecast = mean_forecast * (1 + safe_adjustment)
                        confidence_intervals = confidence_intervals * (1 + safe_adjustment)

                        logger.info(
                            f"Applied correlation-based adjustments to ARIMA forecast with adjustment of {safe_adjustment:.2%}"
                        )
            elif correlation_result and hasattr(correlation_result, 'correlation_score'):
                # Legacy support for older correlation result format
                correlation_score = correlation_result.correlation_score
                if abs(correlation_score) > 0.5:  # Only consider strong correlations
                    used_correlation_data = True
                    correlation_adjustment = correlation_score * 0.01 * horizon_days
                    safe_adjustment = np.clip(correlation_adjustment, -0.05, 0.05)

                    # Apply as a percentage change
                    mean_forecast = mean_forecast * (1 + safe_adjustment)
                    confidence_intervals = confidence_intervals * (1 + safe_adjustment)

                    logger.info(f"Applied legacy correlation adjustment of {safe_adjustment:.2%}")

            # Ensure all forecasts remain positive - critical check after all transformations
            # Use a small value relative to the current rate rather than fixed 0.000001
            # Get the current rate from the original dataframe
            current_rate = original_df.iloc[-1]['close']
            min_value = max(0.000001, current_rate * 0.001)  # Ensure at least 0.1% of current rate
            mean_forecast = pd.Series(np.maximum(min_value, mean_forecast))

            # Also apply minimum to confidence intervals
            confidence_intervals.iloc[:, 0] = np.maximum(min_value * 0.5, confidence_intervals.iloc[:, 0])
            confidence_intervals.iloc[:, 1] = np.maximum(min_value * 1.5, confidence_intervals.iloc[:, 1])

            # Generate dates for the forecast period
            last_date = exchange_df.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

            # Create dictionaries for the forecast values
            mean_predictions = {date.strftime('%Y-%m-%d'): float(val)
                                for date, val in zip(forecast_dates, mean_forecast)}
            lower_bound = {date.strftime('%Y-%m-%d'): float(val)
                           for date, val in zip(forecast_dates, confidence_intervals.iloc[:, 0])}
            upper_bound = {date.strftime('%Y-%m-%d'): float(val)
                           for date, val in zip(forecast_dates, confidence_intervals.iloc[:, 1])}

            # Ensure forecast is reasonable compared to current rate
            # We already have current_rate from earlier

            # Check for unrealistic changes in the first prediction
            first_date = list(mean_predictions.keys())[0]
            first_value = mean_predictions[first_date]
            change_pct = abs((first_value - current_rate) / current_rate) * 100

            # If the first prediction has an unrealistic change, moderate it
            max_reasonable_change = min(1.0, historical_volatility * 200)  # 2 std deviations, capped at 100%
            if change_pct > max_reasonable_change * 100:
                logger.warning(f"Unrealistic change percent: {change_pct:.2f}%. Moderating first prediction.")

                # Direction-aware moderation (maintain trend direction but reduce magnitude)
                direction = np.sign(first_value - current_rate)
                moderated_value = current_rate * (1 + (direction * max_reasonable_change))
                mean_predictions[first_date] = float(moderated_value)
                # Record the scaling factor to apply to other predictions to maintain trend
                scale_factor = ((moderated_value - current_rate) /
                                (first_value - current_rate) if first_value != current_rate else 0.5)

                # Adjust subsequent predictions to follow the same trend but scaled
                prev_value = moderated_value
                for date in list(mean_predictions.keys())[1:]:
                    original_value = mean_predictions[date]
                    # Calculate how much this value would have changed from the previous prediction
                    original_change = original_value - first_value if first_value != 0 else 0
                    # Apply the same direction of change but scaled appropriately
                    mean_predictions[date] = float(prev_value + (original_change * scale_factor))
                    prev_value = mean_predictions[date]

                # Also adjust bounds to maintain proper relationship with mean
                for date in mean_predictions:
                    if lower_bound[date] > mean_predictions[date]:
                        lower_bound[date] = float(mean_predictions[date] * 0.95)
                    if upper_bound[date] < mean_predictions[date]:
                        upper_bound[date] = float(mean_predictions[date] * 1.05)

            # Apply bounds for all forecasted dates based on historical volatility
            for date in mean_predictions:
                # Calculate days from start of forecast
                day_idx = list(mean_predictions.keys()).index(date)

                # Allow for increasing variation further into the future (square root of time rule)
                time_factor = np.sqrt(day_idx + 1)
                max_allowed_change = min(1.0, historical_volatility * 5.0 * time_factor)

                # Keep predictions within reasonable bounds
                if mean_predictions[date] > current_rate * (1 + max_allowed_change):
                    mean_predictions[date] = float(current_rate * (1 + max_allowed_change))

                if mean_predictions[date] < current_rate * (1 - max_allowed_change):
                    mean_predictions[date] = float(max(min_value, current_rate * (1 - max_allowed_change)))

                # Ensure bounds are consistent with mean prediction
                lower_bound[date] = float(min(mean_predictions[date] * 0.99, lower_bound[date]))
                if lower_bound[date] <= 0:
                    lower_bound[date] = float(mean_predictions[date] * 0.9)

                upper_bound[date] = float(max(mean_predictions[date] * 1.01, upper_bound[date]))
                if upper_bound[date] <= 0:
                    upper_bound[date] = float(mean_predictions[date] * 1.1)

            # Calculate error metrics using backtesting
            error_metrics = self._calculate_backtest_errors(exchange_df, best_model, log_transform)

            # Prepare the result
            timestamp_end = time.time()
            elapsed_time = timestamp_end - timestamp_start
            logger.info(f"ARIMA prediction completed in {elapsed_time:.2f} seconds")

            result = {
                'mean_predictions': mean_predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_version': '2.0-ARIMA',
                'confidence_score': min(90, 50 + int(50 * (1 - error_metrics.get('normalized_rmse', 0.5)))),
                'input_data_range': (f"{original_df['date'].min().strftime('%Y-%m-%d')} to "
                                     f"{original_df['date'].max().strftime('%Y-%m-%d')}"),
                'used_correlation_data': used_correlation_data,
                'used_news_sentiment': False,
                'used_economic_indicators': False,
                'used_anomaly_detection': used_anomaly_detection,
                'mean_square_error': error_metrics.get('mse'),
                'root_mean_square_error': error_metrics.get('rmse'),
                'mean_absolute_error': error_metrics.get('mae')
            }

            return result

        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {str(e)}. Traceback: {traceback.format_exc()}")
            return None

    def _calculate_backtest_errors(self, exchange_df, model, log_transform=False):
        """
        Calculate error metrics for the ARIMA model using backtesting.

        Args:
            exchange_df: DataFrame with historical data
            model: Fitted ARIMA model
            log_transform: Whether log transformation was applied

        Returns:
            dict: Dictionary with error metrics
        """
        try:
            total_points = len(exchange_df)
            test_size = min(30, max(5, int(total_points * 0.2)))

            # Split into training and test sets
            train_size = total_points - test_size
            logger.info(f"Calculating backtest errors with train size {train_size} and test size {test_size}")

            # If we're using log-transformed data, we need to work with that
            if log_transform and 'log_close' in exchange_df.columns:
                series = exchange_df['log_close']
            else:
                series = exchange_df['close']

            # Get the actual test values
            actual_values = series.iloc[-test_size:]

            # Get in-sample predictions for the test period
            predictions = model.get_prediction(start=train_size, end=total_points-1)
            predicted_mean = predictions.predicted_mean

            # Convert back from log space if needed
            if log_transform:
                predicted_mean = np.exp(predicted_mean)
                actual_values = np.exp(actual_values)

            # Calculate errors
            errors = predicted_mean - actual_values

            # Calculate error metrics
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(errors))

            # Calculate normalized metrics
            avg_price = np.mean(actual_values)
            normalized_mse = mse / (avg_price ** 2) if avg_price > 0 else 0
            normalized_rmse = rmse / avg_price if avg_price > 0 else 0
            normalized_mae = mae / avg_price if avg_price > 0 else 0

            logger.info(f"Backtest error metrics: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
            logger.info(f"Normalized error metrics: MSE={normalized_mse:.6f}, "
                        f"RMSE={normalized_rmse:.6f} ({normalized_rmse*100:.2f}%), "
                        f"MAE={normalized_mae:.6f} ({normalized_mae*100:.2f}%)")

            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'normalized_mse': float(normalized_mse),
                'normalized_rmse': float(normalized_rmse),
                'normalized_mae': float(normalized_mae)
            }
        except Exception as e:
            logger.warning(f"Error calculating backtest metrics: {str(e)}")
            return {
                'mse': None,
                'rmse': None,
                'mae': None,
                'normalized_mse': None,
                'normalized_rmse': 0.5,  # Default value for confidence calculation
                'normalized_mae': None
            }

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

            # Create attributes with model accuracy metrics
            attributes = {
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

            # Add error metrics if available
            if hasattr(prediction, 'mean_square_error') and prediction.mean_square_error is not None:
                attributes["mean_square_error"] = prediction.mean_square_error
            if hasattr(prediction, 'root_mean_square_error') and prediction.root_mean_square_error is not None:
                attributes["root_mean_square_error"] = prediction.root_mean_square_error
            if hasattr(prediction, 'mean_absolute_error') and prediction.mean_absolute_error is not None:
                attributes["mean_absolute_error"] = prediction.mean_absolute_error

            # Add structured model accuracy metrics with description
            has_mse = hasattr(prediction, 'mean_square_error') and prediction.mean_square_error is not None
            has_rmse = hasattr(prediction, 'root_mean_square_error') and prediction.root_mean_square_error is not None
            has_mae = hasattr(prediction, 'mean_absolute_error') and prediction.mean_absolute_error is not None

            if has_mse and has_rmse and has_mae:
                # Calculate error percentage based on current rate
                error_pct = 0
                if prediction.current_rate > 0:
                    error_pct = 100 * (prediction.root_mean_square_error / prediction.current_rate)

                error_msg = f"RMSE represents approximately {error_pct:.2f}% of current rate."
                description = f"Historical backtesting error metrics. {error_msg}"

                attributes["model_accuracy"] = {
                    "mean_square_error": prediction.mean_square_error,
                    "root_mean_square_error": prediction.root_mean_square_error,
                    "mean_absolute_error": prediction.mean_absolute_error,
                    "description": description
                }

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
                        "attributes": attributes
                    }
                ]
            }

            return response

        except Exception as e:
            logger.error(f"Error formatting ADAGE response: {str(e)}")
            raise
