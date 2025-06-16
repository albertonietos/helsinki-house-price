import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Model Analysis - Helsinki House Prices",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Model Analysis")
st.write("Deep dive into model performance and feature importance")


@st.cache_data
def load_data():
    try:
        return pd.read_excel(
            "./data/cleaned/helsinki_house_price_cleaned.xls", engine="xlrd"
        )
    except FileNotFoundError:
        st.error("❌ Data file not found. Please check that the data file exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


@st.cache_resource
def train_and_evaluate_model():
    data = load_data()
    data = data[data.Total_rooms.notna() & data.Latitude.notna()]

    X = data[["Size", "Year", "Total_rooms", "Latitude", "Longitude"]]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    forest = RandomForestRegressor(random_state=42)
    forest.fit(X_train.values, y_train.values)

    # Make predictions
    y_pred_train = forest.predict(X_train.values)
    y_pred_test = forest.predict(X_test.values)

    return forest, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, data


with st.spinner("Training model and calculating metrics..."):
    (
        forest,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        data,
    ) = train_and_evaluate_model()

st.success("✅ Model analysis complete")

# Model Performance Section
st.header("📊 Model Performance")

col1, col2 = st.columns([2, 1])

with col1:
    # Calculate metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate MAPE
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    # Calculate percentage error relative to market average
    avg_price = y_test.mean()
    percentage_error = (mae_test / avg_price) * 100

    st.subheader("🎯 Model Accuracy Metrics")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            "R² Score (Test)",
            f"{r2_test:.3f}",
            help="Coefficient of determination - how well the model explains price variation",
        )
        st.metric(
            "MAE (Test)",
            f"{mae_test:,.0f} €",
            help="Mean Absolute Error - average prediction error",
        )

    with col_b:
        st.metric(
            "RMSE (Test)",
            f"{rmse_test:,.0f} €",
            help="Root Mean Square Error - penalizes larger errors more",
        )
        st.metric(
            "MAPE (Test)",
            f"{mape_test:.1f}%",
            help="Mean Absolute Percentage Error - error relative to each property's price",
        )

    st.info(
        f"💡 **Model Interpretation**: On average, predictions are off by {mae_test:,.0f} € ({percentage_error:.1f}% of market average)"
    )

with col2:
    st.subheader("📈 Training vs Test")
    st.metric("R² Training", f"{r2_train:.3f}")
    st.metric("R² Test", f"{r2_test:.3f}")

    overfitting = r2_train - r2_test
    if overfitting > 0.1:
        st.warning(f"⚠️ Possible overfitting: {overfitting:.3f}")
    else:
        st.success(f"✅ Good generalization: {overfitting:.3f}")

# Prediction vs Actual scatter plot
st.subheader("🎯 Predictions vs Actual Prices")

fig_scatter = go.Figure()

fig_scatter.add_trace(
    go.Scatter(
        x=y_test,
        y=y_pred_test,
        mode="markers",
        name="Test Predictions",
        marker=dict(
            color="rgba(55, 128, 191, 0.6)",
            size=6,
            line=dict(width=0.5, color="rgba(55, 128, 191, 0.8)"),
        ),
        hovertemplate="<b>Actual:</b> %{x:,.0f} €<br><b>Predicted:</b> %{y:,.0f} €<extra></extra>",
    )
)

# Perfect prediction line
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
fig_scatter.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="red", dash="dash", width=2),
    )
)

fig_scatter.update_layout(
    title="Model Predictions vs Actual Prices",
    xaxis_title="Actual Price (€)",
    yaxis_title="Predicted Price (€)",
    height=500,
    showlegend=True,
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.write(
    "We can see quite a nice correlation between the actual and predicted prices. At the same time, we can also see some heteroskedasticity in the residuals: the variance of the residuals (actual - predicted) increases with the predicted price."
)

# Feature Importance Section
st.header("🔍 Feature Importance Analysis")

st.write(
    """
Feature importance shows which property characteristics have the most impact on price predictions.
Error bars represent the standard deviation across different trees in the Random Forest.
"""
)

# Calculate feature importance with error bars
feature_names = ["Size", "Year", "Total_rooms", "Latitude", "Longitude"]
importances = forest.feature_importances_
std_importances = np.std(
    [tree.feature_importances_ for tree in forest.estimators_], axis=0
)

# Convert to percentages
importances_pct = importances * 100
std_importances_pct = std_importances * 100

# Sort features by importance (descending)
sorted_indices = np.argsort(importances_pct)[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importances = importances_pct[sorted_indices]
sorted_std = std_importances_pct[sorted_indices]

# Create feature importance plot (horizontal bar)
fig_importance = go.Figure()

fig_importance.add_trace(
    go.Bar(
        x=sorted_importances,
        y=sorted_features,
        orientation="h",
        error_x=dict(
            type="data",
            array=sorted_std,
            visible=True,
            color="rgba(0,0,0,0.3)",
            thickness=2,
            width=3,
        ),
        marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"][
            : len(sorted_features)
        ],
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.1f}%<br>Std Dev: %{error_x.array:.1f}%<extra></extra>",
    )
)

fig_importance.update_layout(
    title="Feature Importance in Price Prediction",
    xaxis_title="Importance (%)",
    yaxis_title="Features",
    height=400,
    showlegend=False,
)

st.plotly_chart(fig_importance, use_container_width=True)

# Feature importance interpretation
st.subheader("📝 Feature Importance Interpretation")

col1, col2 = st.columns(2)

with col1:
    for i, (feature, importance, std) in enumerate(
        zip(feature_names, importances_pct, std_importances_pct)
    ):
        if importance > 30:
            icon = "🔥"
            level = "Very High"
        elif importance > 20:
            icon = "⭐"
            level = "High"
        elif importance > 10:
            icon = "📊"
            level = "Medium"
        else:
            icon = "📉"
            level = "Low"

        st.write(f"{icon} **{feature}**: {importance:.1f}% ± {std:.1f}% ({level})")

with col2:
    st.write("**Feature Descriptions:**")
    st.write("• **Size**: Property area in square meters")
    st.write("• **Year**: Year the property was built")
    st.write("• **Total_rooms**: Number of rooms in the property")
    st.write("• **Latitude**: North-south geographic position")
    st.write("• **Longitude**: East-west geographic position")

st.write(
    """
The model relies heavily on the size of the property, which is expected since size is a strong predictor of price. Additionally, the feature is well-represented making it easy for the model to interpret and assign importance. This isn't the case for other features like latitude and longitude, which should be encoded with e.g. geohash.

While the number of rooms is important, most of the variability that they account for is likely explained by the size of the property. This can explain the low importance given to the number of rooms.
"""
)

# Partial Dependence Plots Section
st.header("📈 Partial Dependence Analysis")

st.write(
    """
Partial dependence plots show how each feature individually affects the predicted price,
while keeping all other features at their average values.
"""
)

# Calculate partial dependence for each feature
feature_indices = list(range(len(feature_names)))

# Create subplot figure
fig_pd = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=feature_names,
    specs=[
        [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}, {"type": "xy"}],
    ],
)

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

for i, (feature_idx, feature_name, color) in enumerate(
    zip(feature_indices, feature_names, colors)
):
    # Calculate partial dependence
    pd_result = partial_dependence(
        forest, X_train.values, [feature_idx], grid_resolution=50, method="recursion"
    )

    pd_values = pd_result["average"][0]
    feature_values = pd_result["grid_values"][0]

    row = (i // 3) + 1
    col = (i % 3) + 1

    fig_pd.add_trace(
        go.Scatter(
            x=feature_values,
            y=pd_values,
            mode="lines",
            name=feature_name,
            line=dict(color=color, width=3),
            hovertemplate=f"<b>{feature_name}</b><br>Value: %{{x}}<br>Price Effect: %{{y:,.0f}} €<extra></extra>",
        ),
        row=row,
        col=col,
    )

# Update layout
fig_pd.update_layout(
    title="Partial Dependence Plots - Individual Feature Effects on Price",
    height=800,
    showlegend=False,
)

# Update x-axis titles
fig_pd.update_xaxes(title_text="Size (m²)", row=1, col=1)
fig_pd.update_xaxes(title_text="Year Built", row=1, col=2)
fig_pd.update_xaxes(title_text="Number of Rooms", row=1, col=3)
fig_pd.update_xaxes(title_text="Latitude", row=2, col=1)
fig_pd.update_xaxes(title_text="Longitude", row=2, col=2)

# Update y-axis titles
fig_pd.update_yaxes(title_text="Price Effect (€)", row=1, col=1)
fig_pd.update_yaxes(title_text="Price Effect (€)", row=1, col=2)
fig_pd.update_yaxes(title_text="Price Effect (€)", row=1, col=3)
fig_pd.update_yaxes(title_text="Price Effect (€)", row=2, col=1)
fig_pd.update_yaxes(title_text="Price Effect (€)", row=2, col=2)

st.plotly_chart(fig_pd, use_container_width=True)

# Partial dependence insights
st.subheader("🔍 Partial Dependence Insights")

st.markdown(
    """
The partial dependence plots reveal several interesting patterns in Helsinki property pricing:

**🏠 Size**: Shows a strong linear relationship with price, as expected. Larger properties consistently command higher prices across the entire size range.

**📅 Year Built**: Displays a fascinating U-shaped pattern. Properties built in the mid-20th century (around 1960-1980) show the lowest prices, likely corresponding to periods of mass affordable housing construction. Both older properties (which survived likely represent higher-quality construction) and newer properties command premium prices. Furthermore, very old properties are likely located in central parts of Helsinki, contributing to the higher prices. On top of that, there is surivorship bias as the lower quality buildings from that time are no longer existing.

**🚪 Room Count**: Linear relationship up to approximately 4 rooms, after which additional rooms provide diminishing returns. This suggests that beyond 4 rooms, the number of rooms becomes less important than total size and layout efficiency.

**🌍 Latitude**: Lower latitudes (more southern locations) command higher prices, reflecting Helsinki's geography where the city center and premium coastal areas are located in the south.

**🗺️ Longitude**: Shows a clear peak in the central band, corresponding to Helsinki's city center, with prices declining as you move east or west from the urban core.
"""
)
# Model insights
st.header("💡 Key Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Strongest Price Drivers")
    # Get top 3 features
    top_features = list(
        sorted(
            list(zip(feature_names, importances_pct)),
            key=lambda x: float(x[1]),
            reverse=True,
        )
    )[:3]

    for i, (feature, importance) in enumerate(top_features, 1):
        st.write(f"{i}. **{feature}**: {importance:.1f}% importance")

with col2:
    st.subheader("📊 Model Reliability")
    st.write(f"• **Accuracy**: {r2_test:.1%} of price variation explained")
    st.write(f"• **Typical Error**: ±{mae_test:,.0f} € ({percentage_error:.1f}%)")
    st.write(f"• **MAPE**: {mape_test:.1f}% average relative error")

st.markdown("---")

st.info(
    """
💡 **Remember**: _All models are wrong, some models are useful._ This model provides useful insights into Helsinki property pricing patterns,
but real estate markets are complex with many factors not captured in this dataset. Use these predictions as a starting point for your analysis,
not as definitive valuations.
"""
)
