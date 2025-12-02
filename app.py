import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
st.write("Current working directory:", os.getcwd())

df=pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\project\Car Sell Dataset.csv')

model = joblib.load(r"C:\Users\LENOVO\OneDrive\Desktop\project\carsell_model.pkl")

#print(df.head())
st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")
st.title(" 🚗 Car Price Prediction")
page=st.sidebar.radio("Choose View",["📊 Data Insights","about model","prediction"])

#st.markdown("Predict the resale price of your car by filling in the details below.")
if page == "📊 Data Insights":
    st.subheader("📊 Data Overview")
    st.write(df.head())

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        viz_df = df.copy()
        
        with np.errstate(over='raise'):
            try:
                if viz_df["Price"].max() < 50:  
                    safe_prices = np.clip(viz_df["Price"], -20, 20)  
                    viz_df["Price"] = np.exp(safe_prices)
            except (FloatingPointError, OverflowError):
                pass
        
        sns.histplot(viz_df["Price"], bins=40, ax=ax1, kde=True, color='green')
        ax1.set_title("Distribution of Selling Prices")
        ax1.set_xlabel("Selling Price (₹)")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        viz_df_brand = df.copy()
        
        with np.errstate(over='raise'):
            try:
                if viz_df_brand["Price"].max() < 50:
                    safe_prices = np.clip(viz_df_brand["Price"], -20, 20)
                    viz_df_brand["Price"] = np.exp(safe_prices)
            except (FloatingPointError, OverflowError):
                pass
        
        avg_price_per_brand = viz_df_brand.groupby("Brand")["Price"].mean().sort_values(ascending=False).head(10)
        avg_price_per_brand.plot(kind="bar", ax=ax2, color='orange')
        ax2.set_title("Average Price by Brand")
        ax2.set_ylabel("Avg Selling Price (₹)")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        
elif page=="about model":
    st.subheader("🧠 Model Insights")
    st.markdown("""
    - *Model Used*: Random Forest Regressor
    - *Target Variable*: Log of Selling Price
    - *Input Features*: Brand, Model Name, Model Variant, Car Type, Transmission, Fuel Type, Year, Kilometers, Owner, State, Accidental
    - *Performance Metrics*:
        - MAE: ₹2.9 Lakh
        - RMSE: ₹6.6 Lakh
    - *Note*: Random Forest handles non-linear relationships better than Linear Regression. Log-transformation was applied to reduce skewness.
    - *Model Benefits*: Better handling of categorical variables and reduced overfitting through ensemble learning.
    """)
else:
    st.sidebar.header("Enter Car Details:")
    col1, col2 = st.columns(2)
    with col1:
         brand = st.selectbox("Brand", df["Brand"].unique().tolist())
         #filtered dataframe
         filtered_df = df[df["Brand"] == brand]
         model_name = st.selectbox("Model Name", filtered_df["Model Name"].unique().tolist())
         model_variant = st.selectbox("Model Variant", filtered_df["Model Variant"].unique().tolist())
         car_type = st.selectbox("Car Type", filtered_df["Car Type"].unique().tolist())
         owner = st.selectbox("Owner", df["Owner"].unique().tolist())
         year = st.number_input("Year", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()))
    with col2:
        fuel_type = st.selectbox("Fuel Type", df["Fuel Type"].unique().tolist())
        transmission = st.selectbox("Transmission", df["Transmission"].unique().tolist())
        state = st.selectbox("State", df["State"].unique().tolist())
        accident = st.selectbox("Accidental", df["Accidental"].unique().tolist())
        kms_driven = st.number_input("Kilometers", min_value=0)

st.markdown("---")

if st.button("Predict Price"): 
    input_dict = {
        "Brand":brand,
        "Model Name":model_name,
        "Model Variant": model_variant,
        "Car Type":car_type,
        "Owner":owner,
        "Year": year,
        "Fuel Type": fuel_type,
        "Transmission": transmission,
        "State":state,
        "Accidental":accident,
        "Kilometers":kms_driven       
    }
    # Transform and predict
    input_df=pd.DataFrame([input_dict])
    price = model.predict(input_df)
    st.write(f" 💰 *Predicted Price*: ₹{price[0]:,.2f}")