import streamlit as st
import pandas as pd
import pickle
import sklearn

st.set_page_config(page_title="Cảnh Báo Học Vụ APP", layout="centered")
st.title("Dự Đoán Cảnh Báo Học Vụ")
# Load model
with open('academic_warning_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input form
col1, col2 = st.columns(2)
with col1:
    hoc_ky = st.number_input("Học kỳ", min_value=1, max_value=8, value=3)
    gpa = st.slider("GPA", 0.0, 4.0, 2.5, 0.01)
    tin_chi_dat = st.number_input("Tín chỉ đã đạt", min_value=0, max_value=30, value=15)
    tin_chi_dk = st.number_input("Tín chỉ đăng ký", min_value=12, max_value=30, value=20)

with col2:
    so_mon = st.number_input("Số môn học", min_value=4, max_value=10, value=6)
    so_mon_fail = st.number_input("Số môn rớt", min_value=0, max_value=5, value=1)
    ty_le_tham_gia = st.slider("Tỷ lệ tham gia (%)", 60, 100, 85)

if st.button("🚀 Dự đoán", type="primary"):
    input_df = pd.DataFrame([[hoc_ky, gpa, tin_chi_dat, tin_chi_dk, so_mon, so_mon_fail, ty_le_tham_gia]],
                            columns=['hoc_ky', 'gpa', 'tin_chi_dat', 'tin_chi_dk', 'so_mon', 'so_mon_fail', 'ty_le_tham_gia'])
    
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ **CÓ CẢNH BÁO HỌC VỤ** (Xác suất: {prob:.1%})")
        st.info("Gợi ý: Cần cải thiện GPA và hoàn thành tín chỉ sớm!")
    else:
        st.success(f"✅ **KHÔNG CẢNH BÁO** (Xác suất cảnh báo: {prob:.1%})")
    
    st.balloons()

st.caption("Dự đoán dựa trên mô hình Random Forest đã được huấn luyện. Vui lòng nhập thông tin chính xác để có kết quả tốt nhất.")