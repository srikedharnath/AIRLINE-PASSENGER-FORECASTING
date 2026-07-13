# Predictions
 
"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""
 
import joblib
import numpy as np
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
 
 
class Predictor:
 
    def __init__(self):
 
        self.data_path = "data/airline-passengers.csv"
 
        self.model_path = "models/lstm_model.keras"
 
        self.scaler_path = "models/scaler.pkl"
 
    def predict(self):
 
        # ----------------------------
        # Load Dataset
        # ----------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # ----------------------------
        # Scale Dataset
        # ----------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # ----------------------------
        # Generate Sequences
        # ----------------------------
 
        generator = SequenceGenerator(sequence_length=12)
 
        X, y = generator.create_sequences(scaled_df)
 
        # ----------------------------
        # Train Test Split
        # ----------------------------
 
        splitter = TimeSeriesSplit(train_size=0.80)
 
        X_train, X_test, y_train, y_test = splitter.split(X, y)
 
        # ----------------------------
        # Load Model
        # ----------------------------
 
        model = load_model(self.model_path)
 
        print("\nModel Loaded Successfully.")
 
        # ----------------------------
        # Load Scaler
        # ----------------------------
 
        scaler = joblib.load(self.scaler_path)
 
        print("Scaler Loaded Successfully.")
 
        # ----------------------------
        # Predict
        # ----------------------------
 
        predictions = model.predict(X_test)
 
        # ----------------------------
        # Convert back to original scale
        # ----------------------------
 
        predictions = scaler.inverse_transform(predictions)
 
        y_test = scaler.inverse_transform(y_test)
 
        print("\nPrediction Completed.")
 
        return y_test, predictions
if __name__ == "__main__":
 
    predictor = Predictor()
 
    actual, predicted = predictor.predict()
 
    print("\nFirst 10 Predictions\n")
 
    for i in range(10):
 
        print(
            f"Actual : {actual[i][0]:.2f}   "
            f"Predicted : {predicted[i][0]:.2f}"
        )
 
# Future forecasting
 
"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""
 
import numpy as np
import joblib
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
 
 
class Forecaster:
 
    def __init__(self):
 
        self.data_path = "data/airline-passengers.csv"
 
        self.model_path = "models/lstm_model.keras"
 
        self.scaler_path = "models/scaler.pkl"
 
        self.sequence_length = 12
 
    def forecast(self, future_months=12):
 
        # --------------------------
        # Load Dataset
        # --------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # --------------------------
        # Scale Dataset
        # --------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # --------------------------
        # Load Model
        # --------------------------
 
        model = load_model(self.model_path)
 
        # --------------------------
        # Load Scaler
        # --------------------------
 
        scaler = joblib.load(self.scaler_path)
 
        # --------------------------
        # Last 12 Months
        # --------------------------
 
        last_sequence = scaled_df.values[-self.sequence_length:]
 
        future_predictions = []
 
        # --------------------------
        # Forecast Loop
        # --------------------------
 
        for _ in range(future_months):
 
            input_data = last_sequence.reshape(
                1,
                self.sequence_length,
                1
            )
 
            prediction = model.predict(
                input_data,
                verbose=0
            )
 
            future_predictions.append(prediction[0,0])
 
            last_sequence = np.vstack(
                (
                    last_sequence[1:],
                    prediction
                )
            )
 
        # --------------------------
        # Convert Back
        # --------------------------
 
        future_predictions = np.array(
            future_predictions
        ).reshape(-1,1)
 
        future_predictions = scaler.inverse_transform(
            future_predictions
        )
 
        print("\nFuture Forecast Completed.")
 
        return future_predictions
if __name__ == "__main__":
 
    forecaster = Forecaster()
 
    future = forecaster.forecast(future_months=12)
 
    print("\nNext 12 Month Forecast\n")
 
    for i, value in enumerate(future, start=1):
 
        print(f"Month {i} : {value[0]:.2f}")
   
 