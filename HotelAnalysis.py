import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
# ⚠️ 시각화 개선을 위해 Plotly 라이브러리 추가
import plotly.express as px
from fpdf import FPDF

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="Hotel Similarity Analyzer Pro", layout="wide")

st.title("🏨 호텔 가치 유사도 및 시장 분석 시스템")
st.markdown("업데이트: **Plotly 기반 고정 정사각 평면 시장 지도(PCA) 반영**")

# --- 1. 데이터 전처리 함수 (동일) ---
def preprocess_data(df):
    df_proc = df.copy()
    df_proc = df_proc.apply(lambda x: x.str.strip() if x.dtype == "object" else x).fillna('없음')
    current_date = datetime.now() 
    binary_cols = ['브랜드', '헬스장', '수영장', '사우나', '욕장', '카페', '비즈니스 센터', '미팅룸', '연회장', '라운지', '루프탑바', '레스토랑', '조식 제공여부', '배기지 라커', '홀']
    for col in binary_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].apply(lambda x: 1.0 if str(x) == '있음' else 0.0)
    tier_cols = ['세탁실', '주차장']
    def tier_map(x):
        x = str(x)
        if '무료' in x: return 1.0
        if '유료' in x: return 0.5
        return 0.0
    for col in tier_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].apply(tier_map)
    if '성급' in df_proc.columns:
        df_proc['성급'] = pd.to_numeric(df_proc['성급'], errors='coerce').fillna(0) / 5
    if '객실크기' in df_proc.columns:
        df_proc['객실크기'] = pd.to_numeric(df_proc['객실크기'], errors='coerce').fillna(0)
        max_size = df_proc['객실크기'].max()
        if max_size > 0: df_proc['객실크기'] = df_proc['객실크기'] / max_size
    if '운영시기' in df_proc.columns:
        def calculate_months(date_str):
            try:
                d_str = str(date_str).replace(" ", "").replace(",", ".")
                past_date = datetime.strptime(d_str, "%Y.%m")
                return (current_date.year - past_date.year) * 12 + (current_date.month - past_date.month)
            except: return 9999 
        df_proc['운영개월수'] = df_proc['운영시기'].apply(calculate_months)
        max_months = df_proc['운영개월수'].max()
        df_proc['운영시기_점수'] = (max_months - df_proc['운영개월수']) / max_months if max_months > 0 else 0.0
        df_proc = df_proc.drop(columns=['운영시기', '운영개월수']).rename(columns={'운영시기_점수': '운영시기'})
    if '위치' in df_proc.columns:
        df_proc['위치'] = pd.to_numeric(df_proc['위치'], errors='coerce').fillna(0)
    return df_proc

# --- 2. PDF 생성 함수 (간소화 버전) ---
def create_pdf(hotel_a, hotel_b, score, gap_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Hotel Similarity Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Comparison: {hotel_a} vs {hotel_b}", ln=True)
    pdf.cell(200, 10, txt=f"Similarity Score: {score:.4f}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Top Differences (Gap):", ln=True)
    for index, row in gap_df.head(10).iterrows():
        # 한글 깨짐 방지를 위해 변수 처리 (실제 배포 시 한글 폰트 설정 필요)
        pdf.cell(200, 10, txt=f"- Feature {index}: {row['Gap']:.4f}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 3. 메인 로직 ---
uploaded_file = st.sidebar.file_uploader("호텔 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        first_col_name = df_raw.columns[0]
        df_trans = df_raw.set_index(first_col_name).T.reset_index()
        df_trans = df_trans.rename(columns={df_trans.columns[0]: '호텔명'})
        
        # 전처리 실행
        df_numeric = preprocess_data(df_trans)
        df_numeric = df_numeric.set_index('호텔명')
        features = df_numeric.select_dtypes(include=[np.number]).columns.tolist()

        # --- 사이드바 가중치 조절 ---
        st.sidebar.subheader("⚙️ 특성별 가중치 조절 (0~10)")
        weights_dict = {}
        for f in features:
            weights_dict[f] = st.sidebar.slider(f, 0, 10, 5)
        
        total_w = sum(weights_dict.values())
        norm_weights = {k: v / total_w if total_w > 0 else 1/len(features) for k, v in weights_dict.items()}
        
        weighted_df = df_numeric[features].copy()
        for f in features:
            weighted_df[f] = weighted_df[f] * norm_weights[f]

        matrix = weighted_df.values.astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        df_norm = pd.DataFrame(matrix / norms, index=weighted_df.index, columns=weighted_df.columns)

        # --- UI 레이아웃 ---
        tab1, tab2 = st.tabs(["📊 유사도 분석", "🗺️ 시장 지도 (PCA)"])

        with tab1:
            col1, col2 = st.columns(2)
            h_list = df_norm.index.tolist()
            with col1: target_a = st.selectbox("기준 호텔 A", h_list)
            with col2: target_b = st.selectbox("비교 호텔 B", h_list, index=1 if len(h_list)>1 else 0)

            if st.button("유사도 계산"):
                vec_a = df_norm.loc[target_a].values
                vec_b = df_norm.loc[target_b].values
                score = np.dot(vec_a, vec_b)
                
                st.metric("가치 유사도 점수", f"{score:.4f}")
                st.progress(float(np.clip(score, 0, 1)))
                
                gap_df = pd.DataFrame({target_a: df_norm.loc[target_a], target_b: df_norm.loc[target_b]})
                gap_df['Gap'] = (gap_df[target_a] - gap_df[target_b]).abs()
                
                st.write("#### 주요 특성별 점수 비교")
                st.bar_chart(gap_df[[target_a, target_b]])
                st.table(gap_df.sort_values(by='Gap', ascending=False))
                
                # PDF Report creation is unnecessary
                # pdf_data = create_pdf(target_a, target_b, score, gap_df.sort_values(by='Gap', ascending=False))
                # st.download_button("📄 PDF 리포트 다운로드", data=pdf_data, file_name="hotel_analysis.pdf")

        with tab2:
            st.subheader("📍 호텔 시장 포지셔닝 맵 (PCA)")
            st.markdown("전체 특성 벡터를 2차원으로 압축한 지도입니다. 점들이 가까울수록 가치 구성이 유사한 호텔입니다.")
            
            if len(h_list) >= 2:
                # 1. PCA 계산
                pca = PCA(n_components=2)
                pca_res = pca.fit_transform(df_norm)
                # 시각화를 위한 DataFrame 생성 (호텔명 포함)
                pca_df = pd.DataFrame(pca_res, columns=['x', 'y'], index=df_norm.index).reset_index()
                
                # ⚠️ 2. Plotly를 이용한 진짜 2D 산점도 그리기
                fig = px.scatter(pca_df, x='x', y='y', text='호텔명', # 점 위에 호텔명 표시
                                 labels={'x': 'First PC', 'y': 'Second PC'},
                                 title="Hotel Market Map")
                
                # ⚠️ 3. 차트 레이아웃 정밀 조정 (도메인 고정 및 정사각형 비율)
                fig.update_layout(
                    xaxis=dict(range=[-1.5, 1.5]), # x축 범위 고정
                    yaxis=dict(range=[-1.5, 1.5]), # y축 범위 고정
                    # scaleanchor="x" -> y축의 비율을 x축에 고정시켜 정사각 평면 만듦
                    yaxis_scaleanchor="x", 
                    width=700, height=700, # 차트 자체의 크기도 정사각형으로 세팅
                    showlegend=False # 호텔명이 점 위에 떠 있으므로 범례는 숨김
                )
                fig.update_traces(textposition='top center') # 텍스트 위치 조정
                
                # Streamlit에 Plotly 차트 표시
                st.plotly_chart(fig, use_container_width=False)
                
            else:
                st.info("지도를 그리려면 최소 2개 이상의 호텔 데이터가 필요합니다.")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")

else:
    st.info("사이드바에서 CSV 파일을 업로드해주세요.")
