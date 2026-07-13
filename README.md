# ✈ Airline Passenger Forecasting using LSTM

## 📖 Project Overview

Airline Passenger Forecasting is a Deep Learning project that predicts future airline passenger traffic using a Long Short-Term Memory (LSTM) Neural Network. The model learns historical passenger trends and forecasts future passenger counts using time series analysis.

The project also includes an interactive Streamlit web application for visualization, forecasting, and downloading prediction results.

---

## 🚀 Features

- 📂 Load airline passenger dataset
- 🧹 Data preprocessing and normalization
- 📈 Sequence generation using Sliding Window Technique
- ✂️ Train-Test Split
- 🧠 LSTM Deep Learning Model
- 📊 Model Evaluation (MAE, MSE, RMSE)
- 🔮 Future Passenger Forecasting
- 📉 Interactive Historical Trend Visualization
- 📥 Download Forecast Results as CSV
- 🌐 Professional Streamlit Dashboard

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```text
AIRLINE-PASSENGER-FORECASTING
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets
│   └── airplane-mid-flight-stockcake.jpg
│
├── data
│   └── airline-passengers.csv
│
├── models
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── outputs
│
└── src
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── sequence_generator.py
    ├── train_test_split.py
    ├── model.py
    ├── train.py
    ├── predict.py
    ├── forecast.py
    ├── evaluate.py
    └── visualization.py
```

---

# 📊 Dataset

**Dataset Name**

Airline Passengers Dataset

**Features**

| Feature | Description |
|----------|-------------|
| Month | Month and Year |
| Total Passengers | Total Airline Passengers |

Dataset Size

- Total Records : **144**
- Time Period : **1949 - 1960**

---

# 🔄 Project Workflow

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Preprocessing
   │
   ▼
MinMax Scaling
   │
   ▼
Sequence Generation
(Sliding Window)
   │
   ▼
Train-Test Split
   │
   ▼
LSTM Model
   │
   ▼
Model Training
   │
   ▼
Prediction
   │
   ▼
Evaluation
   │
   ▼
Future Forecast
   │
   ▼
Streamlit Dashboard
```

---

# 🧹 Data Preprocessing

The preprocessing module performs the following operations:

- Loads the dataset
- Converts Month column to DateTime
- Sets Month as Index
- Scales passenger values using MinMaxScaler
- Saves scaler using Joblib

---

# 📈 Sequence Generation

The Sliding Window Technique is used to generate sequences.

Sequence Length = **12 Months**

Example

```text
Input :

Jan
Feb
Mar
Apr
May
Jun
Jul
Aug
Sep
Oct
Nov
Dec

↓

Output :

January of Next Year
```

---

# ✂️ Train Test Split

The generated sequences are divided into:

- **Training Data : 80%**
- **Testing Data : 20%**

This helps evaluate the model on unseen data.

---

# 🧠 Deep Learning Model

The forecasting model uses a Long Short-Term Memory (LSTM) Neural Network.

### Model Architecture

```text
Input Layer

↓

LSTM Layer
64 Units

↓

Dropout Layer
20%

↓

Dense Layer
32 Units

↓

Output Layer
1 Unit
```

---

# ⚙️ Model Configuration

| Parameter | Value |
|------------|-------|
| Model | LSTM |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Metric | Mean Absolute Error |
| Sequence Length | 12 Months |
| Epochs | 50 |
| Batch Size | 16 |

---

# 📊 Model Evaluation

The model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

# 🔮 Future Forecast

The trained model predicts future passenger traffic for:

- 3 Months
- 6 Months
- 9 Months
- 12 Months
- 18 Months
- 24 Months

The forecast is displayed in both graphical and tabular format.

---

# 🌐 Streamlit Dashboard

The dashboard provides:

- 📈 Historical Passenger Trend
- 📊 Model Performance Metrics
- 🔮 Future Passenger Forecast
- 📉 Historical vs Forecast Graph
- 📋 Forecast Table
- 📥 CSV Download Option

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/srikedharnath/AIRLINE-PASSENGER-FORECASTING.git
```

---

## Navigate to Project

```bash
cd AIRLINE-PASSENGER-FORECASTING
```

---

## Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📷 Dashboard Preview

You can add screenshots of:

- Home Dashboard
- Historical Trend
- Forecast Results
- Model Performance

inside the **assets** folder and reference them here.

Example:

```markdown
![Dashboard](assets/dashboard.png)
```

---

# 🎯 Future Enhancements

- GRU Model
- Simple RNN Model
- Multiple Airline Datasets
- Hyperparameter Tuning
- Cloud Deployment
- Real-Time Passenger Forecasting
- Dashboard Authentication
- Model Comparison

---

# 👨‍💻 Author

**Sri Kedharnath**

B.Tech – Computer Science and Engineering (Data Science)

Anurag University

GitHub: https://github.com/srikedharnath

---

# 📄 License

This project is developed for educational and learning purposes.

---

# ⭐ If you found this project useful

Please consider giving this repository a **Star ⭐** on GitHub.
