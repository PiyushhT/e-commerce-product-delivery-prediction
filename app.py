import streamlit as st
import numpy as np
import joblib

# ========== Streamlit Page Config ==========
st.set_page_config(page_title="E-Commerce Delivery Prediction",
                   layout="wide",
                   page_icon="🚚")

# ========== Custom CSS for Cleaner Look ==========
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# ========== App Title ==========
st.title("🚚 E-Commerce Product Delivery Prediction")
st.markdown("Predict whether your product will be delivered **on time** based on shipment details. Ensure timely logistics decisions and enhance customer satisfaction.")

# ========== Load the Trained Model ==========
model = joblib.load("best_delivery_prediction_model.pkl")

# ========== Input Form ==========
st.markdown("### 📝 Enter Shipment Details")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        warehouse_block = st.selectbox("🏢 Warehouse Block", ('A', 'B', 'C', 'D', 'E'),
                                       help="Select warehouse block where product is stored",
                                       index=0)
        mode_of_shipment = st.selectbox("✈️ Mode of Shipment", ('Ship', 'Flight', 'Road'),
                                        help="Choose shipping method (Flight is fastest)",
                                        index=0)
        customer_care_calls = st.number_input("📞 Customer Care Calls", min_value=0, max_value=10, value=1,
                                              help="Number of enquiry calls made by customer")
        customer_rating = st.slider("⭐ Customer Rating", 1, 5, 3,
                                    help="Customer's rating of service (1 is low, 5 is high)")
        prior_purchases = st.number_input("🛒 Prior Purchases", min_value=0, max_value=20, value=1,
                                          help="Number of prior purchases by customer")

    with col2:
        cost_of_the_product = st.number_input("💲 Cost of Product (USD)", min_value=10, max_value=1000, value=100,
                                              help="Enter product price in USD")
        product_importance = st.selectbox("🎯 Product Importance", ('low', 'medium', 'high'),
                                          help="Importance level of product to customer",
                                          index=1)
        gender = st.selectbox("👤 Customer Gender", ('Male', 'Female'),
                              help="Gender of the customer",
                              index=0)
        discount_offered = st.number_input("🏷️ Discount Offered (%)", min_value=0, max_value=100, value=5,
                                           help="Discount offered on product in %")
        weight_in_gms = st.number_input("⚖️ Weight of Product (grams)", min_value=100, max_value=10000, value=500,
                                        help="Weight of product (100g - 10,000g)")

    submit = st.form_submit_button("🔍 Predict Delivery Time")

# ========== Encoding Function ==========
def encode_inputs(warehouse, shipment, importance, gender):
    warehouse_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    shipment_map = {'Flight': 0, 'Road': 1, 'Ship': 2}
    importance_map = {'high': 0, 'low': 1, 'medium': 2}
    gender_map = {'Female': 0, 'Male': 1}

    return (warehouse_map[warehouse], shipment_map[shipment],
            importance_map[importance], gender_map[gender])

# ========== Prediction ==========
if submit:
    w_enc, s_enc, p_enc, g_enc = encode_inputs(warehouse_block, mode_of_shipment, product_importance, gender)

    input_data = np.array([[w_enc, s_enc, customer_care_calls, customer_rating, cost_of_the_product,
                            prior_purchases, p_enc, g_enc, discount_offered, weight_in_gms]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    label_map = {0: "Delivered ON TIME ✅", 1: "NOT Delivered ON TIME ❌"}
    result_label = label_map[prediction]

    # Show result
    st.markdown("## 📊 **Prediction Result**")
    if prediction == 0:
        st.success(f"✅ **Delivered ON TIME** with **{prediction_proba[0]*100:.2f}%** confidence.")
    else:
        st.error(f"⚠️ **NOT Delivered ON TIME** with **{prediction_proba[1]*100:.2f}%** confidence.")

    st.info(f"📝 **Model Prediction:** {result_label}")

    # ======= Expandable Model Insights =======
    with st.expander("🔍 Model Insights"):
        st.write("""
        - **Model Used:** Random Forest Classifier
        - **Input Features:**
            - Warehouse Block
            - Mode of Shipment
            - Customer Care Calls
            - Customer Rating
            - Cost of the Product
            - Prior Purchases
            - Product Importance
            - Gender
            - Discount Offered
            - Weight in grams
        - **Target Variable:** Reached on Time (0 = On Time, 1 = Not On Time)
        """)

# ========== Footer ==========
st.markdown("---")
st.markdown("📌 *Developed as part of Capstone Project — E-Commerce Delivery Prediction*")
st.markdown("💡 *Created with Streamlit | Machine Learning Model powered by Random Forest Classifier*")

