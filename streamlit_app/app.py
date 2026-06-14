import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])

# ---------------- CSS (HOVER EFFECTS) ----------------
st.markdown(
    """
    <style>
    .main {background-color: #0b1220;}

    h1, h2, h3 {
        color: #4ea1ff;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #111827;
        transition: 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(78,161,255,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.title("📊 Business Performance Dashboard")
st.markdown("Interactive dashboard with hover effects and business insights")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

# 🎮 WHAT-IF SLIDER
discount_effect = st.sidebar.slider(
    "Simulate Discount Impact (%)",
    0, 50, 10
)

df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category))
]

# ---------------- KPIs ----------------
sales = df["Sales"].sum()
profit = df["Profit"].sum()
margin = (profit / sales) * 100 if sales != 0 else 0

# ---------------- KPI DISPLAY ----------------
st.subheader("Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Sales", f"${sales:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Profit", f"${profit:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Profit Margin", f"{margin:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- WHAT-IF ANALYSIS ----------------
st.subheader("🎯 What-If Scenario")

simulated_profit = profit - (sales * (discount_effect / 100))

col1, col2 = st.columns(2)
col1.metric("Current Profit", f"${profit:,.0f}")
col2.metric("After Discount", f"${simulated_profit:,.0f}")

st.info(f"If discounts increase by {discount_effect}%, profit may decrease due to margin pressure.")

# ---------------- INTERPRETATION ----------------
st.subheader("What This Means")

if margin > 20:
    st.success("Business is strong and highly profitable.")
elif margin > 10:
    st.info("Business is stable but can be improved.")
else:
    st.warning("Profitability is low and needs attention.")

# ---------------- SALES TREND ----------------
st.subheader("Sales Over Time")

trend = df.groupby("Order Date")["Sales"].sum().reset_index()

fig1 = px.line(
    trend,
    x="Order Date",
    y="Sales",
    markers=True,
    hover_data={"Sales": ":,.2f"}
)

fig1.update_traces(
    hovertemplate="Date: %{x}<br>Sales: $%{y:,.2f}<extra></extra>"
)

fig1.update_layout(
    template="plotly_dark",
    hovermode="x unified"
)

st.plotly_chart(fig1, use_container_width=True)

st.caption("Hover over points to see exact sales values.")

# ---------------- PRODUCT EXPLORER ----------------
st.subheader("Product Explorer")

category_selected = st.selectbox(
    "Choose Category",
    df["Category"].unique()
)

filtered_products = df[df["Category"] == category_selected]

top_products = filtered_products.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()

fig2 = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    color_continuous_scale="Blues",
    hover_data={"Sales": ":,.2f"}
)

fig2.update_traces(
    hovertemplate="Product: %{y}<br>Sales: $%{x:,.2f}<extra></extra>"
)

fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

st.caption("Explore top products inside each category.")

# ---------------- LOW PRODUCTS ----------------
st.subheader("Low Performing Products")

low_products = df.groupby("Product Name")["Sales"].sum().nsmallest(10).reset_index()

fig3 = px.bar(
    low_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    color_continuous_scale="Reds"
)

fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)

st.caption("These products contribute least to revenue.")

# ---------------- REGION ANALYSIS ----------------
st.subheader("Sales by Region")

region_sales = df.groupby("Region")["Sales"].sum().reset_index()

fig4 = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    color="Region",
    text="Sales"
)

fig4.update_traces(
    hovertemplate="Region: %{x}<br>Sales: $%{y:,.2f}<extra></extra>"
)

fig4.update_layout(template="plotly_dark")

st.plotly_chart(fig4, use_container_width=True)

st.caption("Compare performance across regions.")

# ---------------- FORECAST ----------------
st.subheader("Future Sales Forecast (30 Days)")

forecast_df = df.groupby("Order Date")["Sales"].sum().reset_index()
forecast_df.columns = ["ds", "y"]

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig5 = px.line(
    forecast,
    x="ds",
    y="yhat"
)

fig5.update_layout(template="plotly_dark")

st.plotly_chart(fig5, use_container_width=True)

st.caption("Predicted future sales trend.")

# ---------------- HIDDEN INSIGHTS ----------------
st.subheader("Hidden Insights")

if margin < 15:
    st.warning("Low profitability detected.")

product_sales = df.groupby("Product Name")["Sales"].sum()

if product_sales.max() > product_sales.mean() * 5:
    st.info("Revenue is highly dependent on a few products.")

region_var = df.groupby("Region")["Sales"].sum().std()

if region_var > df["Sales"].mean() * 0.5:
    st.warning("Sales distribution across regions is uneven.")

# ---------------- FINAL SUMMARY ----------------
st.subheader("Overall Summary")

st.info(f"""
Total Sales: ${sales:,.0f}  
Total Profit: ${profit:,.0f}  
Profit Margin: {margin:.2f}%

Key insights:
- Revenue is concentrated in top products
- Regional performance is uneven
- Profit depends on pricing structure and product mix
""")