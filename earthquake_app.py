import streamlit as st
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium

# 1. 기본 설정 및 테마
st.set_page_config(page_title="Earthquake Risk Analyzer", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 5px; border: none; }
    .stTextInput>div>div>input { border-color: #ff4b4b; }
    h1 { color: #cc0000; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('earthquake.csv') 
    return df

df_new = load_data()

# 모델 파일을 불러오는 함수
@st.cache_resource
def load_clustering_model():
    # 학습된 모델 객체(예: KMeans)를 로드
    model = joblib.load('earthquake_model.pkl') 
    return model

model = load_clustering_model()

# 위험도 및 색상 설정 (원래 코드의 기준 유지)
risk_dict = {0: '조금 높음', 1: '낮음', 2: '높음', 3: '중간'}
colors = {0:'red', 1:'yellow', 2:'black', 3:'orange'}

# 2. 사이드바: 수치 입력
st.sidebar.header("📍 Location Input")
input_lat = st.sidebar.number_input("위도 (Latitude)", value=0.0, format="%.4f")
input_lon = st.sidebar.number_input("경도 (Longitude)", value=0.0, format="%.4f")

# 3. 메인 화면
st.title("Earthquake Risk Dashboard")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("지도에서 위치 선택")
    # 초기 지도 위치 설정
    m = folium.Map(location=[input_lat, input_lon], zoom_start=3)
    
    # 배경 데이터 표시 (수정: row['model'] -> row['cluster_4'])
    df_sample = df_new.sample(n=min(1000, len(df_new)), random_state=42)
    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['위도'], row['경도']],
            radius=2,
            color=colors.get(row['cluster_4'], 'gray'), # 여기를 수정했습니다
            fill=True,
            opacity=0.4
        ).add_to(m)

    # 지도 출력 및 클릭 이벤트 수신
    map_data = st_folium(m, width=800, height=500)

    clicked_lat, clicked_lon = None, None
    if map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']
        st.success(f"선택된 좌표: {clicked_lat:.4f}, {clicked_lon:.4f}")

# 대상 좌표 결정
target_lat = clicked_lat if clicked_lat else input_lat
target_lon = clicked_lon if clicked_lon else input_lon

with col2:
    st.subheader("Analysis Result")
    
    # 방법 1: 모델로 직접 예측 (추천)
    # input_features = [[target_lat, target_lon]]
    # predicted_cluster = model.predict(input_features)[0]
    
    # 방법 2: 기존 로직(주변 데이터 비율) 유지 (수정: near_df['model'] -> near_df['cluster_4'])
    near_df = df_new[
        (df_new['위도'] >= target_lat - 5) & (df_new['위도'] <= target_lat + 5) &
        (df_new['경도'] >= target_lon - 5) & (df_new['경도'] <= target_lon + 5)
    ]

    if not near_df.empty:
        # 여기도 cluster_4로 수정했습니다
        cluster_counts = near_df['cluster_4'].value_counts(normalize=True)
        main_cluster = cluster_counts.idxmax()
        risk_level = risk_dict[main_cluster]
        
        st.metric(label="예측 위험도", value=risk_level)
        st.write("**주변 군집 분포**")
        st.bar_chart(cluster_counts)
        
        if risk_level in ['높음', '조금 높음']:
            st.error(f"⚠️ 위험 단계: **{risk_level}**")
        else:
            st.info(f"✅ 안전 단계: **{risk_level}**")
    else:
        st.warning("주변에 참조할 데이터가 없습니다.")

    with st.expander("데이터 상세 보기"):
        st.write(f"좌표: {target_lat}, {target_lon}")
        st.dataframe(near_df.head())