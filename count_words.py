import pandas as pd

# 1. 데이터 읽기
df = pd.read_csv('test.csv')
df.columns = ['Date', 'User', 'Message']

# 2. 단어/멤버별 집계
rows = []
for _, row in df.iterrows():
    user = row['User']
    message = row['Message']
    if pd.isna(message):
        continue
    for word in message.split():
        rows.append({'User': user, 'Word': word})

df_words = pd.DataFrame(rows)

# 3. 피벗 테이블 (멤버별 단어별 빈도)
pivot = pd.pivot_table(df_words, index='Word', columns='User', aggfunc='size', fill_value=0)

# 4. 전체 단어 빈도(모든 멤버 합계) 컬럼 생성
pivot['전체빈도'] = pivot.sum(axis=1)

# 5. 전체빈도 기준 내림차순 정렬
pivot_sorted = pivot.sort_values('전체빈도', ascending=False)

# 6. 컬럼 순서 조정: [단어, 전체빈도, 멤버들...]
pivot_sorted = pivot_sorted[['전체빈도'] + [col for col in pivot_sorted.columns if col != '전체빈도']]

# 7. 인덱스(단어)를 컬럼으로
pivot_sorted = pivot_sorted.reset_index()

# 8. 엑셀로 저장
pivot_sorted.to_excel('kakao_word_freq_by_user.xlsx', index=False, sheet_name='단어_멤버별_빈도')