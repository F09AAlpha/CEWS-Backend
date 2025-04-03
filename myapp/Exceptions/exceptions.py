"""
Custom exceptions for the CEWS API.
"""


class CEWSException(Exception):
    """Base exception class for all CEWS-specific exceptions."""
    pass


class InvalidCurrencyCode(CEWSException):
    """Exception raised when an invalid currency code is provided."""
    pass


class InvalidCurrencyPair(CEWSException):
    """Exception raised when an invalid currency pair is provided (e.g., same currency for base and target)."""
    pass


class CorrelationDataUnavailable(CEWSException):
    """Exception raised when correlation data is not available for a given currency pair."""
    pass


class AlphaVantageError(CEWSException):
    """Exception raised when there's an error with Alpha Vantage API."""
    pass


class RateLimitError(AlphaVantageError):
    """Exception raised when Alpha Vantage API rate limit is exceeded."""
    pass


class InvalidRequestError(AlphaVantageError):
    """Exception raised when an invalid request is made to Alpha Vantage API."""
    pass


class TemporaryAPIError(AlphaVantageError):
    """Exception raised when Alpha Vantage API is temporarily unavailable."""
    pass


class InsufficientDataError(CEWSException):
    """Exception raised when there's not enough data for a meaningful analysis."""
    pass


class ProcessingError(CEWSException):
    """Exception raised when there's an error processing data."""
    pass
