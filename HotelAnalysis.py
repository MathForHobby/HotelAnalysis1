import streamlit as st
import pandas as pd
import numpy as np

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="Hotel Similarity Analyzer", layout="wide")

st.title("🏨 호텔 가치 유사도 분석 시스템 (Custom Logic)")
st.markdown("질문자님의 **수치화 가이드라인**이 적용된 전문 분석 툴입니다.")

# --- 1. 데이터 전처리 함수 (질문자님 가이드 반영) ---
def preprocess_data(df):
    df_proc = df.copy()
    
    # [1] 단순 이진 변환 (있음: 1, 없음: 0)
    binary_cols = [
        '브랜드', '헬스장', '수영장', '사우나', '욕장', '카페', 
        '비즈니스 센터', '미팅룸', '연회장', '라운지', 
        '루프탑바', '레스토랑', '조식 제공여부', '배기지 라커', '홀'
    ]
    for col in binary_cols:
        if col in df_proc.columns:
            # 텍스트 데이터를 1과 0으로 매핑
            df_proc[col] = df_proc[col].map({'있음': 1, '없음': 0}).fillna(0)

    # [2] 3단계 변환 (없음: 0, 유료: 0.5, 무료: 1)
    tier_cols = ['세탁실', '주차장']
    for col in tier_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map({'없음': 0, '유료': 0.5, '무료': 1}).fillna(0)

    # [3] 성급 (별 갯수 / 5)
    if '성급' in df_proc.columns:
        df_proc['성급'] = pd.to_numeric(df_proc['성급'], errors='coerce').fillna(0) / 5

    # [4] 객실크기 (가장 큰 방 대비 비율)
    if '객실크기' in df_proc.columns:
        df_proc['객실크기'] = pd.to_numeric(df_proc['객실크기'], errors='coerce').fillna(0)
        max_size = df_proc['객실크기'].max()
        if max_size > 0:
            df_proc['객실크기'] = df_proc['객실크기'] / max_size

    # [5] 운영연수 (신규 호텔 가점 공식)
    # 공식: (가장 오래된 곳 연수 - 내 연수) / 가장 오래된 곳 연수
    if '운영연수' in df_proc.columns:
        df_proc['운영연수'] = pd.to_numeric(df_proc['운영연수'], errors='coerce').fillna(0)
        max_years = df_proc['운영연수'].max()
        if max_years > 0:
            df_proc['운영연수'] = (max_years - df_proc['운영연수']) / max_years

    # [6] 위치 (사용자 입력값 그대로 사용)
    if '위치' in df_proc.columns:
        df_proc['위치'] = pd.to_numeric(df_proc['위치'], errors='coerce').fillna(0)

    return df_proc

# --- 2. 사이드바: 파일 업로드 ---
uploaded_file = st.sidebar.file_uploader("호텔 특성 파일(.csv)을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("📋 입력된 원본 데이터")
    st.dataframe(df_raw, use_container_width=True)

    # 수치화 실행
    if '호텔명' not in df_raw.columns:
        st.error("CSV 파일에 '호텔명' 컬럼이 반드시 포함되어야 합니다.")
    else:
        df_numeric = preprocess_data(df_raw)
        df_numeric.set_index('호텔명', inplace=True)

        # --- 3. 벡터 정규화 및 내적 계산 준비 ---
        # 수치 데이터만 추출하여 L2 Normalization 수행
        matrix = df_numeric.values.astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = matrix / norms
        
        df_norm = pd.DataFrame(normalized_matrix, index=df_numeric.index, columns=df_numeric.columns)

        # --- 4. 메인 분석 UI ---
        st.divider()
        col1, col2 = st.columns(2)
        
        hotel_list = df_raw['호텔명'].unique().tolist()
        
        with col1:
            target_a = st.selectbox("기준 호텔 A 선택", hotel_list)
        with col2:
            target_b = st.selectbox("비교 호텔 B 선택", hotel_list, index=1 if len(hotel_list)>1 else 0)

        if st.button("유사도 분석 실행"):
            vec_a = df_norm.loc[target_a].values
            vec_b = df_norm.loc[target_b].values
            
            # 내적(Inner Product) 계산
            similarity = np.dot(vec_a, vec_b)
            
            # 결과 리포트
            st.success(f"### 두 호텔의 가치 유사도: **{similarity:.4f}**")
            st.progress(float(similarity))
            
            # 특성별 수치 비교 차트
            st.write("#### 📊 특성별 벡터 값 비교 (정규화된 점수)")
            comparison_df = pd.DataFrame({
                target_a: df_norm.loc[target_a],
                target_b: df_norm.loc[target_b]
            })
            comparison_df['Gap'] = (comparison_df[target_a] - comparison_df[target_b]).abs()
            
            # 바 차트로 시각화
            st.bar_chart(comparison_df[[target_a, target_b]])
            
            # 상세 표 제공
            st.write("#### 상세 데이터 비교")
            st.table(comparison_df.sort_values(by='Gap', ascending=False))

else:
    st.info("왼쪽 사이드바에서 분석할 호텔 데이터(.csv)를 업로드해주세요.")
    
    # 동생분을 위한 샘플 데이터 안내
    with st.expander("CSV 파일 양식 가이드 (컬럼명 주의)"):
        st.write("""
        파일에는 다음 컬럼들이 포함되어야 하며, 데이터는 '있음', '없음', '무료', '유료' 등의 텍스트로 구성되어도 됩니다.
        
        **컬럼 리스트:**
        호텔명, 위치, 브랜드, 성급, 헬스장, 수영장, 사우나, 욕장, 세탁실, 주차장(유료/무료/없음), 카페, 비즈니스 센터, 미팅룸, 연회장, 라운지, 루프탑바, 레스토랑, 조식 제공여부, 배기지 라커, 홀, 객실크기, 운영연수
        """)
