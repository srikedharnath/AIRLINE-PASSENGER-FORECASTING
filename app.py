import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.evaluate import Evaluator
from src.forecast import Forecaster

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Airline Forecast Dashboard",
    page_icon="🛫",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"]{
    background:#243B55 !important;
}

/* Make ALL sidebar text white */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *{
    color:white !important;
}

/* Button */
[data-testid="stSidebar"] .stButton>button{
    background:#2196F3 !important;
    color:white !important;
    border:none;
    border-radius:10px;
}

/* Selectbox */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div{
    background:#1E4D78 !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CACHE FUNCTIONS
# -------------------------------------------------

@st.cache_data
def load_dataset():
    loader = DataLoader("data/airline-passengers.csv")
    return loader.load_data()

@st.cache_data
def get_metrics():
    evaluator = Evaluator()
    return evaluator.evaluate()

@st.cache_data
def generate_forecast(months):
    forecaster = Forecaster()
    return forecaster.forecast(months)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = load_dataset()

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
<h1 style='text-align:center;
color:#0B4F9C;
margin-bottom:0px;'>
🛫 Airline Passenger Forecast Dashboard
</h1>

<p style='text-align:center;
font-size:22px;
margin-top:0px;
color:#555555;'>
Forecast Future Passenger Traffic Using LSTM Deep Learning
</p>
""", unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.markdown("## ✈️ Airline Forecast")

    st.markdown("---")

    future_months = st.selectbox(
        "📅 Forecast Months",
        [3, 6, 9, 12, 18, 24],
        index=3
    )

    run = st.button(
        "🚀 Generate Forecast",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.write("**LSTM Neural Network**")

    st.markdown("### 📂 Dataset")
    st.write("airline-passengers.csv")

    st.markdown("### 📊 Records")
    st.write("144 Monthly Records")

    st.markdown("### 📈 Training")
    st.write("Train: 80% | Test: 20%")

    st.markdown("---")

    st.caption("Developed by Tanvi Ganji")

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

st.subheader("📊 Model Performance")

mae, mse, rmse = get_metrics()

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")

st.divider()

# -------------------------------------------------
# DATASET + CHART
# -------------------------------------------------

st.subheader("📈 Historical Airline Data")

left, right = st.columns([1, 2])

with left:

    st.subheader("Dataset")
    st.dataframe(df, height=450)

with right:

    st.subheader("Historical Passenger Trend")

    fig = px.line(
        df,
        x=df.index,
        y="total_passengers",
        markers=True
    )

    fig.update_traces(
        line_color="#0B4F9C",
        line_width=3
    )

    fig.update_layout(
        template="plotly_white",
        title="Historical Passenger Trend",
        title_x=0.35,
        height=500,
        xaxis_title="Month",
        yaxis_title="Passengers"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# FORECAST
# -------------------------------------------------

st.subheader("🔮 Future Forecast")

if run:

    with st.spinner("Generating forecast..."):

        future = generate_forecast(future_months)

    last_date = df.index[-1]

    future_dates = pd.date_range(
        last_date + pd.DateOffset(months=1),
        periods=future_months,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecast": future.flatten()
    })

    st.divider()

    st.header("📈 Forecast Results")

    a, b = st.columns([2, 1])

    with a:

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=df.index,
                y=df["total_passengers"],
                mode="lines",
                name="Historical"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Forecast"],
                mode="lines+markers",
                line=dict(dash="dash"),
                name="Forecast"
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            title="Historical vs Forecast",
            title_x=0.35,
            height=500,
            legend=dict(
                orientation="h",
                y=1.05
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

    with b:

        st.dataframe(forecast_df)

        csv = forecast_df.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download Forecast",
            csv,
            "forecast.csv",
            "text/csv"
        )

    st.success("Forecast generated successfully!")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.markdown("""
<center>
Developed using ❤️ Streamlit | TensorFlow | LSTM | Plotly
</center>
""", unsafe_allow_html=True)