from data_loader import load_data
from model_trainer import train_and_evaluate_model
from recommend import get_top_n_recommendations
import numpy as np
import time

def main():
    """
    메인 함수: 데이터 로드, 모델 훈련 및 평가, 추천 생성을 실행합니다.
    """
    start_time = time.time()
    
    books_df, ratings_df, users_df = load_data(sample_size=5000)  # 샘플 크기를 1000으로 유지

    user_item_matrix, user_id_map, item_id_map, item_means = train_and_evaluate_model(ratings_df)

    # 유효한 사용자 ID 선택
    valid_user_ids = list(user_id_map.values())
    if not valid_user_ids:
        print("경고: 유효한 사용자 ID가 없습니다.")
        return

    sample_user_id = np.random.choice(valid_user_ids)
    user_id_index = list(user_id_map.keys())[list(user_id_map.values()).index(sample_user_id)]
    
    print(f"선택된 사용자 ID: {sample_user_id}")
    print("추천을 생성하고 있습니다...")
    recommendations, scores = get_top_n_recommendations(user_item_matrix, user_id_index, item_means)
    
    print(f"\n사용자 {sample_user_id}에 대한 상위 10개 추천 도서:")
    for item_id, score in zip(recommendations, scores):
        isbn = item_id_map[item_id]
        if isbn in books_df['ISBN'].values:
            book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
            print(f"ISBN: {isbn}, 제목: {book_info['Title']}, 저자: {book_info['Author']}, 예측 평점: {score:.2f}")
        else:
            print(f"ISBN: {isbn}, 책 정보 없음, 예측 평점: {score:.2f}")
    
    end_time = time.time()
    print(f"\n총 실행 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    main()