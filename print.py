import pandas as pd

df = pd.read_csv('test.csv')
df.columns = ['Date', 'User', 'Message']

# 전체 대화(메시지) 수
total_messages = len(df)
print(f'전체 대화(메시지) 수: {total_messages}')

# 전체 단어 수 (공백 기준)
all_words = []
for msg in df['Message'].dropna():
    words = msg.split()
    all_words.extend(words)
total_words = len(all_words)
print(f'전체 단어 수: {total_words}')

# 사용자별 대화(메시지) 수
user_counts = df['User'].value_counts()
print('\n사용자별 대화(메시지) 수:')
for user, cnt in user_counts.items():
    print(f' - {user}: {cnt}')