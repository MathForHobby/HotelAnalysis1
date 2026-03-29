import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from fpdf import FPDF

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="Hotel Similarity Analyzer Pro", layout="wide")

st.title("🏨 호텔 가치 유사도 및 시장 분석 시스템")
st.markdown("수정 사항: **데이터 구조 인식 오류(KeyError) 해결 및 전처리 강화**")

# --- 1. 데이터 전처리 함수 ---
def preprocess_data(df):
    df_proc = df.copy()
    # 모든 텍스트 데이터의 공백 제거 및 결측치 처리
    df_proc = df_proc.apply(lambda x: x.str.strip() if x.dtype == "object" else x).fillna('없음')
    current_date = datetime.now() 

    # [1] 이진 변환
    binary_cols = ['브랜드', '헬스장', '수영장', '사우나', '욕장', '카페', '비즈니스 센터', '미팅룸', '연회장', '라운지', '루프탑바', '레스토랑', '조식 제공여부', '배기지 라커', '홀']
    for col in binary_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].apply(lambda x: 1.0 if str(x) == '있음' else 0.0)

    # [2] 3단계 변환 (세탁실, 주차장)
    tier_cols = ['세탁실', '주차장']
    def tier_map(x):
        x = str(x)
        if '무료' in x: return 1.0
        if '유료' in x: return 0.5
        return 0.0
    for col in tier_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].apply(tier_map)

    # [3] 성급, 객실크기
    if '성급' in df_proc.columns:
        df_proc['성급'] = pd.to_numeric(df_proc['성급'], errors='coerce').fillna(0) / 5
    if '객실크기' in df_proc.columns:
        df_proc['객실크기'] = pd.to_numeric(df_proc['객실크기'], errors='coerce').fillna(0)
        max_size = df_proc['객실크기'].max()
        if max_size > 0: df_proc['객실크기'] = df_proc['객실크기'] / max_size

    # [4] 운영시기 (YYYY.MM)
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

# --- 2. PDF 생성 함수 (한글 이슈 방지를 위해 간소화) ---
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
        pdf.cell(200, 10, txt=f"- {index}: {row['Gap']:.4f}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 3. 메인 로직 ---
uploaded_file = st.sidebar.file_uploader("호텔 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    try:
        # 1. 파일 읽기 및 강제 Transpose 로직
        df_raw = pd.read_csv(uploaded_file)
        
        # 첫 번째 열이 무엇이든 '특성명'으로 간주하고 뒤집기
        first_col_name = df_raw.columns[0]
        df_trans = df_raw.set_index(first_col_name).T.reset_index()
        
        # 뒤집힌 후 첫 번째 컬럼(호텔명이 들어간 곳)을 강제로 '호텔명'으로 명명
        df_trans = df_trans.rename(columns={df_trans.columns[0]: '호텔명'})
        
        st.subheader("📋 데이터 로드 완료")
        st.dataframe(df_trans, use_container_width=True)

        # 2. 전처리 실행
        df_numeric = preprocess_data(df_trans)
        
        # '호텔명' 컬럼이 확실히 있는지 확인 후 인덱스 설정
        if '호텔명' in df_numeric.columns:
            df_numeric = df_numeric.set_index('호텔명')
        else:
            st.error("데이터 변환 중 '호텔명'을 찾지 못했습니다. 파일 형식을 확인해주세요.")
            st.stop()

        features = df_numeric.select_dtypes(include=[np.number]).columns.tolist()

        # --- 사이드바 가중치 조절 ---
        st.sidebar.subheader("⚙️ 특성별 가중치 조절 (0~10)")
        weights_dict = {}
        for f in features:
            weights_dict[f] = st.sidebar.slider(f, 0, 10, 5)
        
        total_w = sum(weights_dict.values())
        norm_weights = {k: v / total_w if total_w > 0 else 1/len(features) for k, v in weights_dict.items()}
        
        # 가중치 및 정규화 적용
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

                # pdf_data = create_pdf(target_a, target_b, score, gap_df.sort_values(by='Gap', ascending=False))
                # st.download_button("📄 PDF 리포트 다운로드", data=pdf_data, file_name="hotel_analysis.pdf")

        with tab2:
            if len(h_list) >= 2:
                pca = PCA(n_components=2)
                pca_res = pca.fit_transform(df_norm)
                pca_df = pd.DataFrame(pca_res, columns=['x', 'y'], index=df_norm.index)
                st.scatter_chart(pca_df)
            else:
                st.info("비교할 호텔이 부족합니다.")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.info("팁: CSV 파일의 첫 번째 칸(A1)에 '호텔명' 혹은 '특성'이 있는지 확인해주세요.")

else:
    st.info("사이드바에서 CSV 파일을 업로드해주세요.")
