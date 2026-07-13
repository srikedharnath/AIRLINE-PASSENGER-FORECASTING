# Cleans the data, handles missing values, encoding, scaling, feature engineering.
# Ensures the model gets clean and usable data.

"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""

# Import required libraries
import joblib
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    """
    Preprocess the time series dataset.
    """

    def __init__(self):
        """
        Initialize the scaler.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def scale_data(self, df):
        """
        Scale the total_passengers column.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        scaled_df : pandas.DataFrame
        """

        print("\nOriginal Data")
        print(df.head())

        # Scale the total_passengers column
        scaled_values = self.scaler.fit_transform(df[["total_passengers"]])

        # Convert to DataFrame
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=["total_passengers"],
            index=df.index
        )

        # Save the scaler
        joblib.dump(self.scaler, "models/scaler.pkl")

        print("\nScaler saved successfully.")

        return scaled_df


# ===========================================
# Test the module
# ===========================================

if __name__ == "__main__":

    from src.data_loader import DataLoader

    DATA_PATH = "data/airline-passengers.csv"

    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()

    # Scale data
    preprocessor = Preprocessor()

    scaled_df = preprocessor.scale_data(df)

    print("\nScaled Dataset")
    print(scaled_df.head())