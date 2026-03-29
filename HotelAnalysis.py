import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="Hotel Similarity Analyzer", layout="wide")

st.title("🏨 호텔 가치 유사도 분석 시스템 (v2.1)")
st.markdown("수정 반영: **주차장** 컬럼명 간소화 및 **빈 데이터(결측치) 0점 처리** 로직 적용")

# --- 1. 데이터 전처리 함수 ---
def preprocess_data(df):
    # 전처리 전 모든 결측치를 '없음' 문자열로 채움 (빈 칸 에러 방지)
    df_proc = df.copy().fillna('없음')
    current_date = datetime.now() 

    # [1] 단순 이진 변환 (있음: 1, 없음/빈칸: 0)
    binary_cols = [
        '브랜드', '헬스장', '수영장', '사우나', '욕장', '카페', 
        '비즈니스 센터', '미팅룸', '연회장', '라운지', 
        '루프탑바', '레스토랑', '조식 제공여부', '배기지 라커', '홀'
    ]
    for col in binary_cols:
        if col in df_proc.columns:
            # 매핑되지 않는 값이나 빈 값은 모두 0으로 처리
            df_proc[col] = df_proc[col].map({'있음': 1, '없음': 0}).fillna(0)

    # [2] 3단계 변환 (없음: 0, 유료: 0.5, 무료: 1)
    # 컬럼명을 '주차장'으로 변경하여 반영
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

    # [5] 운영시기(YYYY.MM) -> 운영 개월 수 -> 신규성 점수(0~1) 변환
    if '운영시기' in df_proc.columns:
        def calculate_months(date_str):
            try:
                # '없음'이거나 날짜 형식이 아니면 아주 오래된 것으로 간주하거나 0 처리
                if date_str == '없음': return 9999 
                past_date = datetime.strptime(str(date_str), "%Y.%m")
                return (current_date.year - past_date.year) * 12 + (current_date.month - past_date.month)
            except:
                return 9999 # 에러 발생 시 최하점 처리를 위해 큰 값 반환

        df_proc['운영개월수'] = df_proc['운영시기'].apply(calculate_months)
        max_months = df_proc['운영개월수'].max()
        
        if max_months > 0:
            # (최대 운영개월 - 내 운영개월) / 최대 운영개월
            df_proc['운영시기_점수'] = (max_months - df_proc['운영개월수']) / max_months
        else:
            df_proc['운영시기_점수'] = 0.0
        
        df_proc = df_proc.drop(columns=['운영시기', '운영개월수'])
        df_proc = df_proc.rename(columns={'운영시기_점수': '운영시기'})

    # [6] 위치 (숫자 데이터, 빈 칸은 0)
    if '위치' in df_proc.columns:
        df_proc['위치'] = pd.to_numeric(df_proc['위치'], errors='coerce').fillna(0)

    return df_proc

# --- 2. 메인 UI ---
uploaded_file = st.sidebar.file_uploader("호텔 특성 파일(.csv)을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # CSV 로드 시 빈 값을 처리하기 쉽게 로드
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("📋 입력된 원본 데이터")
    st.dataframe(df_raw, use_container_width=True)

    if '호텔명' not in df_raw.columns:
        st.error("CSV 파일에 '호텔명' 컬럼이 포함되어야 합니다.")
    else:
        # 전처리 수행
        df_numeric = preprocess_data(df_raw)
        df_numeric.set_index('호텔명', inplace=True)

        # 벡터 정규화(L2 Norm)
        matrix = df_numeric.values.astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = matrix / norms
        df_norm = pd.DataFrame(normalized_matrix, index=df_numeric.index, columns=df_numeric.columns)

        # --- 3. 분석 UI ---
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
            
            similarity = np.dot(vec_a, vec_b)
            
            st.success(f"### 두 호텔의 가치 유사도: **{similarity:.4f}**")
            st.progress(float(similarity))
            
            # 비교 테이블 및 차트
            comparison_df = pd.DataFrame({
                target_a: df_norm.loc[target_a],
                target_b: df_norm.loc[target_b]
            })
            comparison_df['Gap'] = (comparison_df[target_a] - comparison_df[target_b]).abs()
            
            st.write("#### 📊 특성별 벡터 점수 비교 (0~1)")
            st.bar_chart(comparison_df[[target_a, target_b]])
            
            st.write("#### 상세 데이터 분석 테이블")
            st.table(comparison_df.sort_values(by='Gap', ascending=False))

else:
    st.info("왼쪽 사이드바에서 분석할 호텔 데이터를 업로드해 주세요.")
    with st.expander("CSV 파일 작성 가이드"):
        st.write("""
        - **주차장**: '무료', '유료', '없음' 혹은 빈칸으로 입력하세요.
        - **빈칸 처리**: 데이터를 입력하지 않은 칸은 자동으로 '없음' 혹은 '0점' 처리됩니다.
        - **운영시기**: '2024.03' 형식으로 입력하세요.
        """)
