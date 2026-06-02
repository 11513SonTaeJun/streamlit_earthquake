import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# 1. 페이지 기본 설정 및 모던 테마(붉은색/주황색/노란색 포인트) 적용
st.set_page_config(
    page_title="Global Earthquake Risk Analyzer",
    page_icon="🌋",
    layout="wide"
)

# 커스텀 CSS로 UI 스타일링 (다크 모드 기반 모던 스타일)
st.markdown("""
    <style>
    .main { background-color: #111111; color: #ffffff; }
    h1 { color: #FF4B4B; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    h3 { color: #FFA500; }
    .stButton>button {
        background-color: #FF4B4B; color: white; border-radius: 8px;
        border: none; padding: 10px 24px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #FF6B6B; color: white; }
    .card {
        background-color: #222222; padding: 20px; border-radius: 12px;
        border-left: 5px solid #FFD700; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 모델 및 데이터 로드 (캐싱 처리로 속도 최적화)
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('earthquake_model.pkl')
        scaler = joblib.load('earthquake_scaler.pkl')
        # 위경도 매핑 및 위험도 계산을 위해 원본 전처리 데이터 프레임도 필요합니다.
        # 여기서는 파일이 없을 경우를 대비해 예외처리를 하거나 빈 데이터프레임을 방지합니다.
        df_new = pd.read_csv('earthquake(1).csv') 
        # Colab 로직 복사: 필요한 컬럼만 필터링 된 데이터 구축 가정
        if 'cluster_4' not in df_new.columns and '규모' in df_new.columns:
            # 만약 원본만 저장했다면 앱 실행 시점에 빠르게 필터링
            df_new = df_new.rename(columns={
                'time': '발생시간', 'place': '발생지역', 'status': '검토상태', 'tsunami': '쓰나미여부',
                'significance': '영향도', 'data_type': '데이터유형', 'magnitudo': '규모', 'state': '지역',
                'longitude': '경도', 'latitude': '위도', 'depth': '진원깊이', 'date': '발생일시'
            })
            df_new = df_new[df_new['데이터유형'] == 'earthquake'].copy()
            df_new['발생일시'] = pd.to_datetime(df_new['발생일시'], format='mixed')
            df_new = df_new[df_new['발생일시'].dt.year >= 2020]
            df_new = df_new[(df_new['규모'] > 3) & (df_new['진원깊이'] < 70)]
            
            X = df_new[['영향도', '규모', '진원깊이']]
            X_scaled = scaler.transform(X)
            df_new['cluster_4'] = model.predict(X_scaled)
    except Exception as e:
        st.error(f"필수 파일(모델/스케일러/데이터)을 로드하는 데 실패했습니다: {e}")
        return None, None, None
    return model, scaler, df_new

model, scaler, df_new = load_artifacts()

# 위험도 매핑 및 요청하신 색상 설정
risk_dict = {0: '조금 높음', 1: '낮음', 2: '높음', 3: '중간'}
color_dict = {
    '높음': {'bg': '#000000', 'text': '#FFFFFF'},       # 검은색
    '조금 높음': {'bg': '#FF0000', 'text': '#FFFFFF'},  # 빨간색
    '중간': {'bg': '#FFA500', 'text': '#000000'},       # 주황색
    '낮음': {'bg': '#FFFF00', 'text': '#000000'}        # 노란색
}

# 3. UI 레이아웃 구성
st.title("🌋 세계 지진 데이터 기반 위험도 분석 시스템")
st.write("원하는 위치를 입력하거나 지도에서 클릭하여 해당 지역의 지진 위험도를 예측하세요.")
st.markdown("---")

if df_new is not None:
    # 세션 상태를 활용해 지도 클릭과 타이핑 입력 연동
    if 'lat' not in st.session_state:
        st.session_state.lat = 0.0
    if 'lon' not in st.session_state:
        st.session_state.lon = 0.0

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 위치 입력 방식 선택")
        input_method = st.radio("입력 방식을 선택하세요:", ["직접 타이핑 입력", "지도에서 마우스 선택"])
        
        if input_method == "직접 타이핑 입력":
            # 사용자가 직접 입력 가능
            lat_input = st.number_input("위도 (Latitude) 입력", value=st.session_state.lat, format="%.6f")
            lon_input = st.number_input("경도 (Longitude) 입력", value=st.session_state.lon, format="%.6f")
            st.session_state.lat = lat_input
            st.session_state.lon = lon_input
        else:
            st.info("💡 오른쪽 지도에서 원하는 위치를 클릭하면 위도와 경도가 자동으로 가져와집니다.")
            st.metric("선택된 위도", f"{st.session_state.lat:.4f}")
            st.metric("선택된 경도", f"{st.session_state.lon:.4f}")

        # 4. 분석 결과 계산 로직
        st.markdown("### 📊 실시간 위험도 분석 결과")
        
        lat, lon = st.session_state.lat, st.session_state.lon
        
        # 입력값 주변 ±5도 이내의 지진 데이터 추출 (Colab 알고리즘 그대로 활용)
        near_df = df_new[
            (df_new['위도'] >= lat - 5) & (df_new['위도'] <= lat + 5) &
            (df_new['경도'] >= lon - 5) & (df_new['경도'] <= lon + 5)
        ]

        if not near_df.empty:
            cluster_ratio = near_df['cluster_4'].value_counts(normalize=True)
            main_cluster = cluster_ratio.idxmax()
            risk_level = risk_dict[main_cluster]
            
            # 지정된 위험도 색상 가져오기
            bg_color = color_dict[risk_level]['bg']
            text_color = color_dict[risk_level]['text']
            
            # 모던한 스타일의 결과창 카드 시각화
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; text-align: center; border: 2px solid #555555;">
                    <h2 style="color: {text_color}; margin: 0;">위험도 판정: {risk_level}</h2>
                    <p style="color: {text_color}; opacity: 0.8; margin-top: 10px;">주변 데이터 매칭 점수 기반 분석 완료</p>
                </div>
            """, unsafe_allow_html=True)
            
            # 주변 데이터 통계 요약
            with st.expander("🔍 주변 지진 활동 통계 보기"):
                st.write(f"반경 내 탐지된 최근 지진 횟수: **{len(near_df)}건**")
                st.write(f"최대 규모: **{near_df['규모'].max():.2f}**")
                st.write(f"평균 진원 깊이: **{near_df['진원깊이'].mean():.2f} km**")
        else:
            st.warning("⚠️ 선택한 지역 반경 5도 이내에 최근 지진 데이터(2020년 이후 규모 3 초과)가 존재하지 않아 위험도를 측정할 수 없습니다.")

    with col2:
        st.subheader("🗺️ 지진 분포 및 분석 지도")
        
        # 기본 지도 생성 (중심점: 최근 선택 위치 또는 전세계)
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=3, tiles="CartoDB dark_matter")
        
        # 속도 최적화를 위한 쌤플 데이터 시각화 (Colab과 동일하게 1000개 추출)
        df_sample = df_new.sample(n=min(1000, len(df_new)), random_state=42)
        
        # 지도에 표시할 마커 색상 (요청사항 반영)
        # 0: 조금높음(Red), 1: 낮음(Yellow), 2: 높음(Black), 3: 중간(Orange)
        map_colors = {0: 'red', 1: 'yellow', 2: 'black', 3: 'orange'}
        
        for i, row in df_sample.iterrows():
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=2.5,
                color=map_colors.get(row['cluster_4'], 'gray'),
                fill=True,
                fill_opacity=0.6
            ).add_to(m)
            
        # 현재 선택한 사용자 입력 위치 표시
        folium.Marker(
            location=[st.session_state.lat, st.session_state.lon],
            icon=folium.Icon(color='lightgray', icon='star', icon_color='purple'),
            tooltip="선택된 분석 지점"
        ).add_to(m)

        # 5. 지도 클릭 이벤트 수집 (st_folium 활용)
        map_data = st_folium(m, width="100%", height=600, key="earthquake_map")
        
        # 지도 클릭 시 세션 상태 변화 및 리런(Rerun)
        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            
            if clicked_lat != st.session_state.lat or clicked_lon != st.session_state.lon:
                st.session_state.lat = clicked_lat
                st.session_state.lon = clicked_lon
                st.rerun()
