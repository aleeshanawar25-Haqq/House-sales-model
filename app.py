import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="House Sales ML Dashboard", layout="wide")

st.title("🏠 House Sales ML Dashboard")
st.markdown("Built by Haqq Nawaz 🚀")

# Load Data
df = pd.read_csv("house_sales.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Sidebar Filters
st.sidebar.header("Filter Data")

min_price = int(df["price"].min())
max_price = int(df["price"].max())

price_range = st.sidebar.slider(
    "Select Price Range",
    min_price,
    max_price,
    (min_price, max_price)
)

filtered_df = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
]

# ================= KPI =================
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Houses", len(filtered_df))
col2.metric("Average Price", f"${int(filtered_df['price'].mean())}")
col3.metric("Max Price", f"${int(filtered_df['price'].max())}")

# ================= VISUALIZATION =================

st.subheader("Price Distribution")

fig, ax = plt.subplots()
sns.histplot(filtered_df["price"], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Bedrooms vs Price")

fig2, ax2 = plt.subplots()
sns.scatterplot(x="bedrooms", y="price", data=filtered_df, ax=ax2)
st.pyplot(fig2)

# ================= MACHINE LEARNING =================

st.subheader("🤖 House Price Prediction Model")

features = ["bedrooms", "bathrooms", "sqft_living"]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

st.write(f"Model R² Score: **{round(score,2)}**")

# ================= PREDICTION =================

st.subheader("Predict House Price")

bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 1500)

if st.button("Predict Price"):
    prediction = model.predict(pd.DataFrame(
        [[bedrooms, bathrooms, sqft_living]],
        columns=features
    ))

    st.success(f"Estimated Price: ${int(prediction[0])}") 