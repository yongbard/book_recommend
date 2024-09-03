import numpy as np
from scipy.sparse import csr_matrix

def get_top_n_recommendations(user_item_matrix, user_id, item_means, n=10):
    if user_id >= user_item_matrix.shape[0]:
        print(f"경고: 사용자 ID {user_id}가 행렬의 범위를 벗어났습니다.")
        return [], []
    
    user_ratings = user_item_matrix[user_id].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    unrated_items = np.where(user_ratings == 0)[0]
    
    print(f"사용자가 평가한 아이템 수: {len(rated_items)}")
    print(f"미평가 아이템 수: {len(unrated_items)}")
    
    if len(rated_items) == 0:
        print("경고: 사용자가 평가한 아이템이 없습니다. 전체 아이템 중 평균 평점이 높은 아이템을 추천합니다.")
        top_items = np.argsort(item_means)[-n:][::-1]
        scores = item_means[top_items]
    else:
        user_mean = np.mean(user_ratings[rated_items])
        print(f"사용자의 평균 평점: {user_mean:.2f}")
        
        # 협업 필터링 기반의 간단한 추천 시스템
        similar_users = np.dot(user_item_matrix, user_item_matrix[user_id].T).toarray().flatten()
        similar_users[user_id] = 0  # 자기 자신 제외
        top_similar_users = np.argsort(similar_users)[-10:]  # 상위 10명의 유사 사용자 선택
        
        pred_ratings = np.zeros(user_item_matrix.shape[1])
        for similar_user in top_similar_users:
            sim_user_ratings = user_item_matrix[similar_user].toarray().flatten()
            pred_ratings += sim_user_ratings * similar_users[similar_user]
        
        pred_ratings /= np.sum(np.abs(similar_users[top_similar_users]))
        
        # 예측 평점과 아이템 평균의 가중 평균
        scores = 0.7 * pred_ratings + 0.3 * item_means
        top_items = unrated_items[np.argsort(scores[unrated_items])[-n:][::-1]]
        scores = scores[top_items]
    
    return top_items, scores