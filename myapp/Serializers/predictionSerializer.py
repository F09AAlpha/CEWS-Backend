from rest_framework import serializers


class PredictionFactorSerializer(serializers.Serializer):
    """
    Serializer for factors that influenced the prediction
    """
    factor_name = serializers.CharField()
    impact_level = serializers.CharField()  # high, medium, low
    used_in_prediction = serializers.BooleanField()


class PredictionValueSerializer(serializers.Serializer):
    """
    Serializer for prediction values for a single time point
    """
    timestamp = serializers.DateTimeField()
    mean = serializers.FloatField()
    lower_bound = serializers.FloatField()
    upper_bound = serializers.FloatField()


class ModelAccuracySerializer(serializers.Serializer):
    """
    Serializer for model accuracy metrics
    """
    mean_square_error = serializers.FloatField(allow_null=True)
    root_mean_square_error = serializers.FloatField(allow_null=True)
    mean_absolute_error = serializers.FloatField(allow_null=True)
    description = serializers.CharField(
        default="Calculated from historical backtest",
        required=False
    )


class PredictionTimeObjectSerializer(serializers.Serializer):
    """
    Serializer for forecast time object in ADAGE 3.0 format
    """
    timestamp = serializers.DateTimeField()
    horizon_days = serializers.IntegerField()
    timezone = serializers.CharField()


class PredictionAttributesSerializer(serializers.Serializer):
    """
    Serializer for prediction attributes in ADAGE 3.0 format
    """
    base_currency = serializers.CharField(max_length=3)
    target_currency = serializers.CharField(max_length=3)
    current_rate = serializers.FloatField()
    change_percent = serializers.FloatField()
    confidence_score = serializers.FloatField()
    model_version = serializers.CharField()
    input_data_range = serializers.CharField()
    influencing_factors = PredictionFactorSerializer(many=True)
    prediction_values = PredictionValueSerializer(many=True)
    backtest_values = PredictionValueSerializer(many=True, required=False)

    # Error metrics fields (backward compatibility)
    mean_square_error = serializers.FloatField(allow_null=True, required=False)
    root_mean_square_error = serializers.FloatField(
        allow_null=True, required=False)
    mean_absolute_error = serializers.FloatField(
        allow_null=True, required=False)

    # New structured accuracy metrics
    model_accuracy = ModelAccuracySerializer(required=False)


class PredictionEventSerializer(serializers.Serializer):
    """
    Serializer for prediction event in ADAGE 3.0 format
    """
    time_object = PredictionTimeObjectSerializer()
    event_type = serializers.CharField()
    attributes = PredictionAttributesSerializer()


class DatasetTimeObjectSerializer(serializers.Serializer):
    """
    Serializer for dataset time object in ADAGE 3.0 format
    """
    timestamp = serializers.DateTimeField()
    timezone = serializers.CharField()


class CurrencyPredictionSerializer(serializers.Serializer):
    """
    ADAGE 3.0 compliant serializer for currency prediction
    """
    data_source = serializers.CharField()
    dataset_type = serializers.CharField()
    dataset_id = serializers.CharField()
    time_object = DatasetTimeObjectSerializer()
    events = PredictionEventSerializer(many=True)
