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
        df = self.alpha_vantage_service.get_forex_daily(base, target)

        # Filter for the requested period
        start_date = datetime.now() - timedelta(days=days)
        period_df = df[df.index >= start_date]

        if period_df.empty:
            raise ValueError(f"No data available for {base}/{target} in the last {days} days")

        # Calculate daily returns
        period_df['return'] = period_df['close'].pct_change()
        period_df = period_df.dropna()  # Remove NaN values

        # Standard deviation of returns * sqrt(trading days) * 100 = annualized volatility as percentage
        current_volatility = period_df['return'].std() * np.sqrt(252) * 100

        # Calculate average volatility over different time windows
        windows = min(len(period_df) // 5, 5)  # Use at most 5 windows
        window_size = len(period_df) // windows

        volatilities = []
        for i in range(windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            window_data = period_df.iloc[start_idx:end_idx]
            window_volatility = window_data['return'].std() * np.sqrt(252) * 100
            volatilities.append(window_volatility)

        average_volatility = np.mean(volatilities)

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

        return {
            "base": base,
            "target": target,
            "current_volatility": round(current_volatility, 2),
            "average_volatility": round(average_volatility, 2),
            "volatility_level": volatility_level,
            "analysis_period_days": days,
            "trend": trend
        }
