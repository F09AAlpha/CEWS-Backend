import numpy as np
from datetime import datetime, timedelta
from .alpha_vantage import AlphaVantageService


class VolatilityService:
    def __init__(self):
        self.alpha_vantage_service = AlphaVantageService()

    def calculate_volatility(self, base, target, days=30):
        """
        Calculate volatility metrics for a currency pair over a specified period
        """
        # Get historical data
        df, alpha_vantage_metadata = self.alpha_vantage_service.get_forex_daily(base, target)

        # Filter for the requested period
        start_date = datetime.now() - timedelta(days=days)
        period_df = df[df.index >= start_date]

        if period_df.empty:
            raise ValueError(f"No data available for {base}/{target} in the last {days} days")

        # Calculate daily returns
        period_df = period_df.copy()
        period_df.loc[:, 'return'] = period_df['close'].pct_change()
        period_df = period_df.dropna()  # Remove NaN values

        # Calculate annualized volatility (std dev of returns * sqrt(trading days) * 100 for percentage)
        current_volatility = period_df['return'].std() * np.sqrt(252) * 100

        # Calculate average volatility over different time windows
        windows = min(len(period_df) // 5, 5)  # Use at most 5 windows
        window_size = max(len(period_df) // windows, 1)

        volatilities = []
        for i in range(windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(period_df))
            if end_idx <= start_idx:
                continue
            window_data = period_df.iloc[start_idx:end_idx]
            if len(window_data) > 1:  # Need at least 2 points for std
                window_volatility = window_data['return'].std() * np.sqrt(252) * 100
                volatilities.append(window_volatility)

        average_volatility = np.mean(volatilities) if volatilities else current_volatility

        # Determine volatility level
        if current_volatility < 10:
            volatility_level = "NORMAL"
        elif current_volatility < 20:
            volatility_level = "HIGH"
        else:
            volatility_level = "EXTREME"

        # Determine trend
        if len(volatilities) < 2:
            trend = "STABLE"
        else:
            first_half = np.mean(volatilities[:len(volatilities)//2])
            second_half = np.mean(volatilities[len(volatilities)//2:])

            if second_half > first_half * 1.1:  # 10% increase
                trend = "INCREASING"
            elif second_half < first_half * 0.9:  # 10% decrease
                trend = "DECREASING"
            else:
                trend = "STABLE"

        # Calculate confidence score based on data amount and quality
        data_points = len(period_df)
        min_required = 20
        confidence_score = min(100, (data_points / min_required) * 100) if min_required > 0 else 0

        # Current time for event timestamps
        current_time = datetime.now()

        # Format response according to ADAGE 3.0 data model
        response = {
            "data_source": alpha_vantage_metadata['data_source'],
            "dataset_type": "Currency Volatility Analysis",
            "dataset_id": f"volatility_analysis_{base}_{target}_{current_time.strftime('%Y%m%d')}",
            "time_object": {
                "timestamp": current_time.isoformat(),
                "timezone": "GMT+0"
            },
            "events": [
                {
                    "time_object": {
                        "timestamp": current_time.isoformat(),
                        "duration": days,
                        "duration_unit": "day",
                        "timezone": "GMT+0"
                    },
                    "event_type": "volatility_analysis",
                    "attributes": {
                        "base_currency": base.upper(),
                        "target_currency": target.upper(),
                        "current_volatility": round(float(current_volatility), 2),
                        "average_volatility": round(float(average_volatility), 2),
                        "volatility_level": volatility_level,
                        "trend": trend,
                        "data_points": data_points,
                        "confidence_score": round(float(confidence_score), 2),
                        "analysis_period_days": days
                    }
                }
            ]
        }

        return response
