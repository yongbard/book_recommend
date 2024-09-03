import pandas as pd
import numpy as np

def load_data(sample_size=5000):
    print("데이터를 로드하고 있습니다...")
    books_df = pd.read_csv('./dataset/books.csv', sep=';', on_bad_lines='skip', low_memory=False)
    ratings_df = pd.read_csv('./dataset/ratings.csv', sep=';')
    users_df = pd.read_csv('./dataset/users.csv', sep=';')
    
    # ISBN 형식 통일 및 공백 제거
    books_df['ISBN'] = books_df['ISBN'].str.strip().str.upper()
    ratings_df['ISBN'] = ratings_df['ISBN'].str.strip().str.upper()
    
    # 평점 범위 확인 및 이상치 제거 (0-10 범위 외의 값 제거)
    ratings_df = ratings_df[(ratings_df['Rating'] >= 0) & (ratings_df['Rating'] <= 10)]
    
    # 평점이 있는 책만 선택
    valid_isbns = set(ratings_df['ISBN']) & set(books_df['ISBN'])
    books_df = books_df[books_df['ISBN'].isin(valid_isbns)]
    ratings_df = ratings_df[ratings_df['ISBN'].isin(valid_isbns)]
    
    # 평가한 책이 1개 이상인 사용자만 선택
    user_rating_counts = ratings_df['User-ID'].value_counts()
    valid_users = user_rating_counts[user_rating_counts > 0].index
    ratings_df = ratings_df[ratings_df['User-ID'].isin(valid_users)]
    
    if len(ratings_df) > sample_size:
        print(f"평점 데이터를 {sample_size}개로 샘플링합니다...")
        sampled_users = np.random.choice(ratings_df['User-ID'].unique(), size=min(len(valid_users), sample_size // 2), replace=False)
        ratings_df = ratings_df[ratings_df['User-ID'].isin(sampled_users)]
        ratings_df = ratings_df.groupby('User-ID').apply(lambda x: x.sample(min(len(x), 5))).reset_index(drop=True)
        
        if len(ratings_df) > sample_size:
            ratings_df = ratings_df.sample(n=sample_size, random_state=42)
    
    # 최종 데이터셋 정리
    books_df = books_df[books_df['ISBN'].isin(ratings_df['ISBN'])]
    users_df = users_df[users_df['User-ID'].isin(ratings_df['User-ID'])]
    
    print(f"최종 데이터 크기:")
    print(f"Books: {len(books_df)}, Ratings: {len(ratings_df)}, Users: {len(users_df)}")
    print(f"유니크한 ISBN 수: {ratings_df['ISBN'].nunique()}")
    print(f"유니크한 사용자 수: {ratings_df['User-ID'].nunique()}")
    print(f"평균 사용자당 평점 수: {len(ratings_df) / ratings_df['User-ID'].nunique():.2f}")
    print(f"평점 범위: {ratings_df['Rating'].min()} - {ratings_df['Rating'].max()}")
    
    return books_df, ratings_df, users_df