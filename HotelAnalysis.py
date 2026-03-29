import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="Hotel Similarity Analyzer", layout="wide")

st.title("🏨 호텔 가치 유사도 분석 시스템 (v2.3)")
st.markdown("수정 사항: **데이터 전처리 안정성 강화 및 Transpose 로직 최적화**")

# --- 1. 데이터 전처리 함수 ---
def preprocess_data(df):
    # 전처리 전 모든 결측치를 '없음'으로 채우고, 텍스트의 앞뒤 공백 제거
    df_proc = df.copy().apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df_proc = df_proc.fillna('없음')
    current_date = datetime.now() 

    # [1] 단순 이진 변환
    binary_cols = [
        '브랜드', '헬스장', '수영장', '사우나', '욕장', '카페', 
        '비즈니스 센터', '미팅룸', '연회장', '라운지', 
        '루프탑바', '레스토랑', '조식 제공여부', '배기지 라커', '홀'
    ]
    for col in binary_cols:
        if col in df_proc.columns:
            # 매핑 시 'None', 'nan', 빈칸 등 예외 케이스 방어
            df_proc[col] = df_proc[col].apply(lambda x: 1 if str(x) == '있음' else 0)

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

    # [3] 성급 (별 갯수 / 5)
    if '성급' in df_proc.columns:
        df_proc['성급'] = pd.to_numeric(df_proc['성급'], errors='coerce').fillna(0) / 5

    # [4] 객실크기 (가장 큰 방 대비 비율)
    if '객실크기' in df_proc.columns:
        df_proc['객실크기'] = pd.to_numeric(df_proc['객실크기'], errors='coerce').fillna(0)
        max_size = df_proc['객실크기'].max()
        if max_size > 0:
            df_proc['객실크기'] = df_proc['객실크기'] / max_size

    # [5] 운영시기(YYYY.MM) -> 신규성 점수 변환
    if '운영시기' in df_proc.columns:
        def calculate_months(date_str):
            try:
                # 2024.03 또는 2024.3 등 다양한 형식 대응
                d_str = str(date_str).replace(" ", "").replace(",", ".")
                past_date = datetime.strptime(d_str, "%Y.%m")
                return (current_date.year - past_date.year) * 12 + (current_date.month - past_date.month)
            except:
                return 9999 

        df_proc['운영개월수'] = df_proc['운영시기'].apply(calculate_months)
        max_months = df_proc['운영개월수'].max()
        
        if max_months > 0:
            df_proc['운영시기_점수'] = (max_months - df_proc['운영개월수']) / max_months
        else:
            df_proc['운영시기_점수'] = 0.0
        
        df_proc = df_proc.drop(columns=['운영시기', '운영개월수'])
        df_proc = df_proc.rename(columns={'운영시기_점수': '운영시기'})

    # [6] 위치
    if '위치' in df_proc.columns:
        df_proc['위치'] = pd.to_numeric(df_proc['위치'], errors='coerce').fillna(0)

    return df_proc

# --- 2. 메인 UI 및 파일 로드 ---
uploaded_file = st.sidebar.file_uploader("호텔 특성 파일(.csv)을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    try:
        # CSV 로드
        df_input = pd.read_csv(uploaded_file)
        
        # ⚠️ Transpose 로직 보강
        # 첫 번째 열의 이름을 '특성명'으로 강제 지정하여 인덱스 설정
        df_input = df_input.rename(columns={df_input.columns[0]: '특성명'})
        df_transposed = df_input.set_index('특성명').T.reset_index()
        
        # 뒤집힌 후 첫 번째 컬럼(호텔명) 이름 정리
        df_transposed = df_transposed.rename(columns={df_transposed.columns[0]: '호텔명'})
        
        st.subheader("📋 분석용 데이터 변환 완료")
        st.dataframe(df_transposed, use_container_width=True)

        # 전처리 및 수치화
        df_numeric = preprocess_data(df_transposed)
        df_numeric.set_index('호텔명', inplace=True)

        # 수치형 데이터만 사용하여 벡터 정규화
        matrix = df_numeric.select_dtypes(include=[np.number]).values.astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = matrix / norms
        
        df_norm = pd.DataFrame(normalized_matrix, index=df_numeric.index, columns=df_numeric.columns)

        # --- 3. 분석 UI ---
        st.divider()
        hotel_list = df_transposed['호텔명'].unique().tolist()
        
        if len(hotel_list) < 2:
            st.warning("비교를 위해 최소 2개 이상의 호텔 데이터가 필요합니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_a = st.selectbox("기준 호텔 A 선택", hotel_list)
            with col2:
                target_b = st.selectbox("비교 호텔 B 선택", hotel_list, index=1)

            if st.button("유사도 분석 실행"):
                vec_a = df_norm.loc[target_a].values
                vec_b = df_norm.loc[target_b].values
                
                similarity = np.dot(vec_a, vec_b)
                
                st.success(f"### 두 호텔의 가치 유사도: **{similarity:.4f}**")
                st.progress(float(np.clip(similarity, 0.0, 1.0)))
                
                comparison_df = pd.DataFrame({
                    target_a: df_norm.loc[target_a],
                    target_b: df_norm.loc[target_b]
                })
                comparison_df['Gap'] = (comparison_df[target_a] - comparison_df[target_b]).abs()
                
                st.write("#### 📊 특성별 벡터 점수 비교 (0~1)")
                st.bar_chart(comparison_df[[target_a, target_b]])
                
                st.write("#### 상세 데이터 분석 테이블")
                st.table(comparison_df.sort_values(by='Gap', ascending=False))
                
    except Exception as e:
        st.error(f"데이터 처리 중 에러가 발생했습니다: {e}")
        st.info("CSV 파일의 형식이 '특성명'이 첫 번째 열에 오고 호텔들이 옆으로 나열된 형태인지 확인해주세요.")

else:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
