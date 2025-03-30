from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert
from myapp.Serializers.exchangeRateAlertSerializer import ExchangeRateAlertSerializer
import uuid  

class RegisterAlertView(APIView):
    def post(self, request):
        # Debugging: Print the incoming data
        print("Received data:", request.data)

        data = request.data.copy()  # Make a mutable copy of request data

        # Generate a unique alert ID and add it to the request data
        alert_id = f"ALERT-{uuid.uuid4().hex[:8]}"
        data["alert_id"] = alert_id

        # Debugging: Print the modified data
        print("Modified data:", data)

        serializer = ExchangeRateAlertSerializer(data=data)
        if serializer.is_valid():
            serializer.save()  # Save the validated alert data to the database
            return Response({
                "alert_id": alert_id,
                "status": "registered",
                "message": "Alert successfully registered"
            }, status=status.HTTP_201_CREATED)  # Use 201 for resource creation

        # Debugging: Print validation errors if the data is not valid
        print("Validation errors:", serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

