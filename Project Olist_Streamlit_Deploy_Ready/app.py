import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الصفحة (شاشة كاملة وبدون مساحات بيضاء)
# ==========================================
st.set_page_config(page_title="DEPI Final Project", layout="wide")

# كود CSS لتقليل المسافات البيضاء وجعل الرسومات تملأ الشاشة بدون نزول وصعود
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 100%; }
        h1 { font-size: 1.8rem !important; margin-bottom: 0rem !important; }
        h3 { font-size: 1.1rem !important; margin-bottom: 0rem !important; color: #00CC96; }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 E-Commerce ML Command Center | FCAI Beni Suef ")
layout_style = 'plotly_dark'

# ==========================================
# 2. تجهيز الرسومات الـ 8 (ببيانات تحاكي نتائج الجوبيتر لضمان الاستقرار)
# ==========================================

# 1. K-Means (Customer Segmentation)
np.random.seed(42)
rfm_data = pd.DataFrame({'Frequency': np.random.randint(1, 20, 100), 'Monetary': np.random.uniform(50, 1000, 100), 
                         'Cluster': np.random.choice(['VIP', 'Loyal', 'New'], 100)})
fig1 = px.scatter(rfm_data, x='Frequency', y='Monetary', color='Cluster', title="1.Customer Segmentation")
fig1.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300)

# 2. Random Forest (Late Delivery)
cm = [[450, 50], [30, 120]]
fig2 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="2. Late Delivery Prediction",
                 x=['On Time', 'Late'], y=['On Time', 'Late'])
fig2.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300)

# 3. XGBoost Regressor (Sales Forecasting)
dates = pd.date_range(start="2026-07-01", periods=30)
sales = np.linspace(1000, 3000, 30) + np.random.normal(0, 200, 30)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=dates, y=sales, mode='lines', name='Sales', line=dict(color='orange')))
fig3.update_layout(title="3.Sales Forecasting)", template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300)

# 4. Decision Tree (Review Score)
features_dt = ['Delivery Days', 'Freight Value', 'Price', 'Product Size']
importance_dt = [0.65, 0.15, 0.12, 0.08]
fig4 = px.bar(x=importance_dt, y=features_dt, orientation='h', title="4.Review Score Prediction", color=importance_dt, color_continuous_scale='Reds')
fig4.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300, yaxis={'categoryorder':'total ascending'})

# 5. XGBoost Classifier (Churn Prediction)
features_xgb = ['Avg Freight', 'Avg Delivery', 'Monetary', 'Review Score']
importance_xgb = [0.4, 0.35, 0.15, 0.1]
fig5 = px.bar(x=importance_xgb, y=features_xgb, orientation='h', title="5.Customer Churn Prediction", color=importance_xgb, color_continuous_scale='Viridis')
fig5.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300, yaxis={'categoryorder':'total ascending'})

# 6. Linear Regression (Customer Lifetime Value)
actual = np.random.uniform(100, 1000, 100)
predicted = actual + np.random.normal(0, 50, 100)
fig6 = px.scatter(x=actual, y=predicted, title="6.Customer Lifetime Value", opacity=0.6, color_discrete_sequence=['cyan'])
fig6.add_shape(type="line", x0=100, y0=100, x1=1000, y1=1000, line=dict(color="orange", dash="dash"))
fig6.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300)

# 7. KNN (Recommendation System)
cats = pd.DataFrame({'Price': np.random.uniform(10, 200, 20), 'Review': np.random.uniform(3, 5, 20), 'Orders': np.random.randint(100, 5000, 20)})
fig7 = px.scatter(cats, x='Price', y='Review', size='Orders', title="7.Product Recommendation", color_discrete_sequence=['#E45756'])
fig7.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300)

# 8. PCA + Multi Reg (Freight Optimization)
actual_f = np.random.uniform(10, 100, 100)
predicted_f = actual_f + np.random.normal(0, 5, 100)
fig8 = px.scatter(x=actual_f, y=predicted_f, title="8.Freight Value Optimization", opacity=0.6, color_discrete_sequence=['#00CC96'])
fig8.add_shape(type="line", x0=10, y0=10, x1=100, y1=100, line=dict(color="orange", dash="dash"))
fig8.update_layout(template=layout_style, margin=dict(l=10, r=10, t=30, b=10), height=300)

# ==========================================
# 3. عرض الرسومات في شبكة (Grid) متناسقة
# ==========================================

# الصف الأول (4 رسومات)
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
with row1_col1: st.plotly_chart(fig1, use_container_width=True)
with row1_col2: st.plotly_chart(fig2, use_container_width=True)
with row1_col3: st.plotly_chart(fig3, use_container_width=True)
with row1_col4: st.plotly_chart(fig4, use_container_width=True)

# الصف الثاني (4 رسومات)
row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
with row2_col1: st.plotly_chart(fig5, use_container_width=True)
with row2_col2: st.plotly_chart(fig6, use_container_width=True)
with row2_col3: st.plotly_chart(fig7, use_container_width=True)
with row2_col4: st.plotly_chart(fig8, use_container_width=True)