import pandas as pd
import numpy as np
import logging
import uuid
from datetime import timedelta
from django.db import transaction
from django.utils import timezone
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import traceback

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

            # Try ARIMA model first if requested
            prediction_results = None
            model_used = "Statistical"

            if use_arima:
                try:
                    logger.info(f"Attempting ARIMA model for {base_currency}/{target_currency}")
                    arima_results = self._predict_with_arima_model(
                        exchange_df, correlation_result, horizon_days, confidence_level
                    )

                    if arima_results is not None:
                        prediction_results = arima_results
                        model_used = "ARIMA"
                        logger.info(f"Successfully used ARIMA model for {base_currency}/{target_currency}")
                    else:
                        logger.warning(f"ARIMA model returned None for {base_currency}/{target_currency}")
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

            # Add volatility for confidence intervals - but use percentage-based approach
            # Calculate historical daily changes as percentage
            if len(close_series) > 1:
                close_pct_changes = np.diff(close_series) / close_series[:-1]
                daily_volatility = np.std(close_pct_changes)
            else:
                # Fallback if we have only one data point (shouldn't happen in practice)
                daily_volatility = 0.005  # Default 0.5% daily volatility

            # Convert confidence level to z-score
            # Commonly used z-scores: 90% -> 1.645, 95% -> 1.96, 99% -> 2.576
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

            # Base boundaries on percentage changes with square root of time
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

                        # Recalculate volatility using normal data
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
                        # First prediction based on last training value
                        pred = train_data[-1] + backtest_drift
                    else:
                        # Subsequent predictions based on prior prediction
                        pred = backtest_predictions[-1] + backtest_drift
                    backtest_predictions.append(pred)

                # Calculate error metrics
                errors = np.array(backtest_predictions) - np.array(test_data)
                mean_square_error = np.mean(errors ** 2)
                root_mean_square_error = np.sqrt(mean_square_error)
                mean_absolute_error = np.mean(np.abs(errors))

                # Normalize errors relative to the average price for better interpretability
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
        Make predictions using ARIMA (AutoRegressive Integrated Moving Average) model
        with incorporated correlation data, anomaly detection, and volatility indicators.

        Args:
            exchange_df (pandas.DataFrame): Historical exchange rate data
            correlation_result (CorrelationResult): Correlation analysis result
            horizon_days (int): Forecast horizon in days
            confidence_level (int): Confidence level for prediction intervals (50-99)

        Returns:
            dict: Prediction results
        """
        try:
            # Suppress statsmodels warnings
            warnings.filterwarnings("ignore")

            # Ensure data is properly sorted
            exchange_df['date'] = pd.to_datetime(exchange_df['date'])
            exchange_df = exchange_df.sort_values('date')

            # Extract close prices for forecasting
            close_series = exchange_df['close'].values

            # Get the last value (current rate)
            last_value = close_series[-1]

            # Apply anomaly detection to clean data for better ARIMA fitting
            cleaned_data = close_series.copy()
            used_anomalies = False
            try:
                # Initialize anomaly detection service
                anomaly_detector = AnomalyDetectionService(
                    base_currency=exchange_df.attrs.get('base_currency', 'Unknown'),
                    target_currency=exchange_df.attrs.get('target_currency', 'Unknown'),
                    analysis_period_days=min(90, len(exchange_df)),
                    z_score_threshold=2.0
                )

                # Detect anomalies
                anomaly_result = anomaly_detector.detect_anomalies()
                anomaly_count = anomaly_result['anomaly_count']

                # If anomalies are found, clean the data
                if anomaly_count > 0:
                    used_anomalies = True
                    logger.info(f"Found {anomaly_count} anomalies in historical data - cleaning for ARIMA")

                    # Create a dictionary of anomaly points for easy lookup
                    anomaly_points = {}
                    for point in anomaly_result['anomaly_points']:
                        date_str = point['timestamp'].split('T')[0]  # Extract date part
                        anomaly_points[date_str] = point

                    # Convert dates to string format for comparison
                    date_strs = [d.strftime('%Y-%m-%d') for d in exchange_df['date']]

                    # Identify anomalous indices
                    anomaly_indices = [i for i, date in enumerate(date_strs) if date in anomaly_points]

                    # Replace anomalies with interpolated values
                    if anomaly_indices:
                        temp_series = pd.Series(close_series)
                        for idx in anomaly_indices:
                            if idx > 0 and idx < len(temp_series) - 1:
                                # Linear interpolation (average of previous and next)
                                temp_series.iloc[idx] = (temp_series.iloc[idx-1] + temp_series.iloc[idx+1]) / 2
                        cleaned_data = temp_series.values
            except Exception as e:
                logger.warning(f"Error applying anomaly detection for ARIMA: {str(e)}")

            # Check data stationarity
            adf_result = adfuller(cleaned_data)
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05 means stationary
            logger.info(f"ADF Stationarity test p-value: {adf_result[1]:.6f} - Stationary: {is_stationary}")

            # For time series forecasting, we'll use statsmodels ARIMA
            if len(cleaned_data) >= 30:  # Need sufficient data for reliable ARIMA
                try:
                    # Try different ARIMA orders and select best by AIC
                    best_aic = float('inf')
                    best_order = None
                    best_model = None

                    # Grid search for best parameters (p, d, q)
                    # p: AR order, d: differencing, q: MA order
                    p_values = range(0, 3)
                    d_values = range(0, 2)
                    q_values = range(0, 3)

                    # For non-stationary data, ensure at least d=1
                    if not is_stationary:
                        d_values = range(1, 3)

                    logger.info("Searching for optimal ARIMA parameters")

                    # Try different combinations to find best model
                    for p in p_values:
                        for d in d_values:
                            for q in q_values:
                                try:
                                    # Skip higher-order models if we already have too many parameters
                                    if p + q > 4:
                                        continue

                                    # Fit ARIMA model
                                    model = ARIMA(cleaned_data, order=(p, d, q))
                                    model_fit = model.fit()

                                    # Compare AIC
                                    current_aic = model_fit.aic
                                    if current_aic < best_aic:
                                        best_aic = current_aic
                                        best_order = (p, d, q)
                                        best_model = model_fit

                                except Exception:
                                    # Some parameter combinations may not converge
                                    continue

                    # If we found a valid model
                    if best_model is not None:
                        order = best_order
                        model_fit = best_model
                        logger.info(f"Selected ARIMA order: {order} with AIC: {best_aic:.2f}")
                    else:
                        # Fallback to simple model if grid search failed
                        if is_stationary:
                            order = (1, 0, 0)  # AR(1) for stationary data
                        else:
                            order = (1, 1, 0)  # ARIMA(1,1,0) for non-stationary

                        model = ARIMA(cleaned_data, order=order)
                        model_fit = model.fit()
                        logger.info(f"Using fallback ARIMA order: {order}")

                    # Get AIC for the model
                    aic = model_fit.aic

                    # Forecast future values
                    arima_forecast = model_fit.forecast(steps=horizon_days)

                    # Calculate prediction intervals (confidence intervals)
                    # Use historical volatility for more realistic bounds
                    # This approach will create tighter and more realistic bounds

                    # Calculate historical daily changes as percentage
                    close_pct_changes = np.diff(cleaned_data) / cleaned_data[:-1]

                    # Calculate the volatility as standard deviation of percentage changes
                    daily_volatility = np.std(close_pct_changes)

                    # Convert confidence level to z-score
                    # Commonly used z-scores: 90% -> 1.645, 95% -> 1.96, 99% -> 2.576
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

                    # Base boundaries on percentage changes rather than absolute values
                    forecast_bounds = []
                    for i in range(horizon_days):
                        # Volatility grows with square root of time horizon
                        # This is a standard approach in financial forecasting
                        time_factor = np.sqrt(i + 1)
                        pct_bound = daily_volatility * z_score * time_factor

                        # Cap the percentage bound at a reasonable level (10%)
                        pct_bound = min(pct_bound, 0.10)

                        forecast_bounds.append(pct_bound)

                    # Calculate actual bounds
                    lower_bounds = arima_forecast * (1 - np.array(forecast_bounds))
                    upper_bounds = arima_forecast * (1 + np.array(forecast_bounds))

                    # Ensure minimal forecast movement (avoid flat predictions)
                    if np.max(np.abs(np.diff(arima_forecast))) < daily_volatility * last_value:
                        # If forecast is too flat, add slight directional trend based on recent movement
                        recent_trend = np.mean(np.diff(cleaned_data[-5:]))
                        trend_direction = np.sign(recent_trend)
                        min_movement = daily_volatility * 0.5 * last_value

                        # Add progressive trend to each forecast point
                        for i in range(1, len(arima_forecast)):
                            arima_forecast[i] += trend_direction * min_movement * i/2

                        # Recalculate bounds with the adjusted forecast
                        lower_bounds = arima_forecast * (1 - np.array(forecast_bounds))
                        upper_bounds = arima_forecast * (1 + np.array(forecast_bounds))

                    # If correlation data is available, adjust predictions
                    used_correlation = False
                    used_correlation_factors = False
                    used_news = False
                    used_econ = False
                    correlation_factor_weights = {}
                    correlation_strength = 0.0
                    correlation_confidence = 0.0
                    used_correlation_details = {}

                    if correlation_result is not None:
                        try:
                            used_correlation = True
                            # Apply correlation factors as adjustments to the ARIMA predictions
                            # First, check for top influencing factors
                            if hasattr(correlation_result, 'top_influencing_factors'):
                                top_factors = correlation_result.top_influencing_factors
                                if top_factors and len(top_factors) > 0:
                                    used_correlation_factors = True
                                    # Calculate a weighted adjustment based on correlation factors
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

                                    # If we have valid factors, apply adjustments
                                    if total_weight > 0:
                                        drift_adjustment = 0

                                        # Apply each factor's influence
                                        for factor_name, details in factor_weights.items():
                                            factor_contribution = (details['correlation'] * details['weight'] / total_weight)
                                            drift_adjustment += factor_contribution
                                            correlation_factor_weights[factor_name] = details['weight'] / total_weight

                                        # Apply a nuanced adjustment that increases over time
                                        # For longer horizons, correlation factors have more influence
                                        adjustment_strength = min(0.15, 0.05 * len(factor_weights))
                                        for i in range(horizon_days):
                                            # Gradually increase effect of correlation factors (more impact further out)
                                            time_factor = 1.0 + (i / horizon_days)
                                            adjustment = drift_adjustment * adjustment_strength * time_factor
                                            arima_forecast[i] *= (1 + adjustment)
                                            lower_bounds[i] *= (1 + adjustment)
                                            upper_bounds[i] *= (1 + adjustment)

                                        logger.info("Applied correlation-based adjustments to ARIMA forecast")

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

                            # Apply correlation confidence from the correlation result if available
                            if hasattr(correlation_result, 'confidence_score'):
                                correlation_confidence = correlation_result.confidence_score

                            # Store factor weights for reporting
                            if correlation_factor_weights:
                                used_correlation_details['factors'] = correlation_factor_weights

                                # Calculate overall correlation strength
                                weighted_corr_strength = sum(abs(details['correlation']) * details['weight']
                                                             for details in factor_weights.values())
                                correlation_strength = min(1.0, weighted_corr_strength)
                        except Exception as e:
                            logger.warning(f"Error applying correlation adjustments to ARIMA: {str(e)}")

                    # Prepare dates for the predictions (starting from tomorrow)
                    last_date = exchange_df['date'].iloc[-1]
                    prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                                        for i in range(horizon_days)]

                    # Format the prediction results with proper dates
                    mean_predictions = {date: float(value) for date, value in zip(prediction_dates, arima_forecast)}
                    lower_bound = {date: float(value) for date, value in zip(prediction_dates, lower_bounds)}
                    upper_bound = {date: float(value) for date, value in zip(prediction_dates, upper_bounds)}

                    # Calculate backtest error metrics
                    if len(cleaned_data) >= 30:
                        # Split data for in-sample testing
                        train_size = len(cleaned_data) - min(14, len(cleaned_data) // 4)
                        train_data = cleaned_data[:train_size]
                        test_data = cleaned_data[train_size:]

                        # Fit model on training data
                        backtest_model = ARIMA(train_data, order=order)
                        backtest_fit = backtest_model.fit()

                        # Make predictions for test period
                        backtest_preds = backtest_fit.forecast(steps=len(test_data))

                        # Calculate error metrics
                        errors = backtest_preds - test_data
                        mean_square_error = np.mean(errors ** 2)
                        root_mean_square_error = np.sqrt(mean_square_error)
                        mean_absolute_error = np.mean(np.abs(errors))

                        logger.info(
                            f"ARIMA backtest metrics: MSE={mean_square_error:.6f}, "
                            f"RMSE={root_mean_square_error:.6f}, MAE={mean_absolute_error:.6f}"
                        )
                    else:
                        # Not enough data for proper backtest
                        logger.warning("Insufficient data for ARIMA backtest. Using model's native error metrics.")

                        # Use ARIMA's own error metrics
                        mean_square_error = model_fit.mse
                        root_mean_square_error = np.sqrt(mean_square_error)
                        mean_absolute_error = np.mean(np.abs(model_fit.resid))

                    # Calculate confidence score - ARIMA typically has higher baseline confidence
                    # when the model fits well

                    # 1. Data Quality Factor (30%): Based on amount of historical data
                    data_points = len(exchange_df)
                    data_quality_score = min(100, data_points / 2)  # Each 2 data points = 1% up to 100%

                    # 2. Model Fit Factor (25%): Based on AIC and ARIMA parameter selection
                    # Lower AIC is better
                    aic_score = 100 * np.exp(-abs(aic) / 1000)
                    model_fit_score = max(0, min(100, aic_score))

                    # 3. Correlation Factor (20%): Based on correlation strength and confidence
                    if correlation_confidence > 0:
                        correlation_score = (correlation_strength * 100 * 0.4) + (correlation_confidence * 0.6)
                    else:
                        correlation_score = correlation_strength * 100

                    # Apply a minimum correlation score if we used correlation data
                    if used_correlation_factors:
                        correlation_score = max(correlation_score, 30)
                    elif used_correlation:
                        correlation_score = max(correlation_score, 20)

                    # 4. Prediction Variance Factor (15%): Lower variance = higher confidence
                    # Use coefficient of variation to normalize
                    forecast_cv = daily_volatility / np.mean(arima_forecast) if np.mean(arima_forecast) != 0 else 1
                    variance_score = 100 * (1 - min(1, forecast_cv * 5))  # Scale to reasonable range

                    # 5. Anomaly Factor (10%): Credit for handling anomalies
                    anomaly_factor = 75  # Default score (good if no anomalies found)
                    if used_anomalies and 'anomaly_count' in locals():
                        # Higher score if anomalies were successfully handled
                        anomaly_ratio = min(1.0, anomaly_count / max(1, len(exchange_df)))
                        anomaly_factor = 100 * (1 - anomaly_ratio * 0.5)  # Less penalty for anomalies in ARIMA

                    # Weighted combination of all factors
                    confidence_score = (
                        (data_quality_score * 0.30) +
                        (model_fit_score * 0.25) +
                        (correlation_score * 0.20) +
                        (variance_score * 0.15) +
                        (anomaly_factor * 0.10)
                    )

                    # Ensure score is between 0 and 100
                    confidence_score = max(0, min(100, confidence_score))

                    # Cap confidence score at 85 since even the best models have significant uncertainty
                    confidence_score = min(85, confidence_score)

                    # Log confidence calculation
                    logger.info(
                        f"ARIMA confidence calculation: Data Quality: {data_quality_score:.1f}, "
                        f"Model Fit: {model_fit_score:.1f}, Correlation: {correlation_score:.1f}, "
                        f"Variance: {variance_score:.1f}, Anomaly: {anomaly_factor:.1f} => Final: {confidence_score:.1f}"
                    )

                    # Format dates for input range
                    first_date = exchange_df['date'].iloc[0]
                    last_date = exchange_df['date'].iloc[-1]
                    input_range = f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}"

                    return {
                        'mean_predictions': mean_predictions,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'model_version': f"ARIMA({order[0]},{order[1]},{order[2]})",
                        'confidence_score': round(confidence_score, 1),
                        'input_data_range': input_range,
                        'used_correlation_data': used_correlation,
                        'used_correlation_details': used_correlation_details,
                        'used_news_sentiment': used_news,
                        'used_economic_indicators': used_econ,
                        'used_anomaly_detection': used_anomalies,
                        'mean_square_error': float(mean_square_error),
                        'root_mean_square_error': float(root_mean_square_error),
                        'mean_absolute_error': float(mean_absolute_error),
                        'arima_params': {
                            'order': order,
                            'aic': float(aic),
                            'is_stationary': is_stationary
                        }
                    }
                except Exception as e:
                    logger.error(f"Error in ARIMA model fitting: {str(e)}")
                    logger.info("Falling back to statistical model due to ARIMA error")
            else:
                logger.warning(f"Insufficient data for ARIMA model (need at least 30 points, got {len(cleaned_data)})")

            # If we get here, ARIMA failed or had insufficient data
            # Fall back to statistical model
            return None

        except Exception as e:
            logger.error(f"Error making prediction with ARIMA model: {str(e)}")
            return None

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
