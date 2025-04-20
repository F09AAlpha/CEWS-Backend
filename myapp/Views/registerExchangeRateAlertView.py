from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.Serializers.exchangeRateAlertSerializer import ExchangeRateAlertSerializer
import uuid


class RegisterAlertView(APIView):
    def post(self, request):
        print("Received data:", request.data)  # Debugging

        data = request.data.copy()  # Make a mutable copy of request data

        # Generate a unique alert ID if not provided
        if not data.get("alert_id"):
            data["alert_id"] = f"ALERT-{uuid.uuid4().hex[:8]}"

        serializer = ExchangeRateAlertSerializer(data=data)
        if serializer.is_valid():
            serializer.save()  # Save the validated data
            return Response({
                "alert_id": serializer.data["alert_id"],
                "status": "registered",
                "message": "Alert successfully registered"
            }, status=status.HTTP_201_CREATED)

        print("Validation errors:", serializer.errors)  # Debugging
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
