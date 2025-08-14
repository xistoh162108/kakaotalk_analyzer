import pandas as pd

# 1. CSV 파일 읽기
df = pd.read_csv('test.csv')  # 파일명 맞게 수정
df.columns = ['Date', 'User', 'Message']  # 컬럼명 확인 후 수정

# 2. 날짜/시간 컬럼 변환 및 파생 컬럼 추가
df['Date'] = pd.to_datetime(df['Date'])
df['Date_only'] = df['Date'].dt.date
df['Year'] = df['Date'].dt.isocalendar().year
df['Week'] = df['Date'].dt.isocalendar().week
weekday_map = {0:'월요일', 1:'화요일', 2:'수요일', 3:'목요일', 4:'금요일', 5:'토요일', 6:'일요일'}
df['Weekday'] = df['Date'].dt.weekday.map(weekday_map)
df['Hour'] = df['Date'].dt.hour

# 3. 전체 통계
## 날짜별
daily = df.groupby('Date_only').size().rename('Count').reset_index()
daily_sorted = daily.sort_values('Count', ascending=False)

## 주별
weekly = df.groupby(['Year','Week']).size().rename('Count').reset_index()
weekly_sorted = weekly.sort_values('Count', ascending=False)

## 요일별
weekday = df.groupby('Weekday').size().rename('Count').reset_index()
weekday_sorted = weekday.sort_values('Count', ascending=False)

## 시간별
hourly = df.groupby('Hour').size().rename('Count').reset_index()
hourly_sorted = hourly.sort_values('Count', ascending=False)

# 4. 사용자별 통계 저장 (각각의 데이터프레임을 dict에 저장)
users = df['User'].unique()
user_stats = {}

for user in users:
    sub = df[df['User'] == user]
    # 날짜별
    d = sub.groupby('Date_only').size().rename('Count').reset_index()
    d_sorted = d.sort_values('Count', ascending=False)
    # 주별
    w = sub.groupby(['Year','Week']).size().rename('Count').reset_index()
    w_sorted = w.sort_values('Count', ascending=False)
    # 요일별
    wd = sub.groupby('Weekday').size().rename('Count').reset_index()
    wd_sorted = wd.sort_values('Count', ascending=False)
    # 시간별
    h = sub.groupby('Hour').size().rename('Count').reset_index()
    h_sorted = h.sort_values('Count', ascending=False)
    # dict에 저장
    user_stats[user] = {
        'daily': d, 'daily_sorted': d_sorted,
        'weekly': w, 'weekly_sorted': w_sorted,
        'weekday': wd, 'weekday_sorted': wd_sorted,
        'hourly': h, 'hourly_sorted': h_sorted,
    }

# 5. 통합 엑셀로 저장
with pd.ExcelWriter('kakao_stats.xlsx') as writer:
    # 전체
    daily.to_excel(writer, sheet_name='전체_날짜별', index=False)
    daily_sorted.to_excel(writer, sheet_name='전체_날짜별_정렬', index=False)
    weekly.to_excel(writer, sheet_name='전체_주별', index=False)
    weekly_sorted.to_excel(writer, sheet_name='전체_주별_정렬', index=False)
    weekday.to_excel(writer, sheet_name='전체_요일별', index=False)
    weekday_sorted.to_excel(writer, sheet_name='전체_요일별_정렬', index=False)
    hourly.to_excel(writer, sheet_name='전체_시간별', index=False)
    hourly_sorted.to_excel(writer, sheet_name='전체_시간별_정렬', index=False)
    # 사용자별
    for user in users:
        for k, v in user_stats[user].items():
            sheet_name = f'{user}_{k}'
            v.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # 시트명 31자 제한

# 6. CSV로도 저장하고 싶다면, 예시
daily.to_csv('전체_날짜별.csv', index=False)
daily_sorted.to_csv('전체_날짜별_정렬.csv', index=False)
# 사용자별 예시
for user in users:
    user_stats[user]['daily'].to_csv(f'{user}_날짜별.csv', index=False)
    user_stats[user]['daily_sorted'].to_csv(f'{user}_날짜별_정렬.csv', index=False)