from rest_framework.views import APIView
from rest_framework.response import Response


class HealthCheckView(APIView):
    """
    Service health check endpoint.
    """
    def get(self, request):
        """Handle GET request for health check."""
        return Response({"message": "Currency Collector API is running"})
