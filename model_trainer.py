from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from matrix_builder import create_user_item_matrix
from model_utils import save_model, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from matrix_builder import create_user_item_matrix
from model_utils import save_model, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from matrix_builder import create_user_item_matrix
from model_utils import save_model, load_model
from tqdm import tqdm

def calculate_item_means(user_item_matrix):
    """
    각 아이템의 평균 평점을 계산합니다.
    
    :param user_item_matrix: 사용자-아이템 행렬 (CSR 형식)
    :return: 아이템 평균 평점 배열
    """
    print("아이템 평균 평점을 계산하고 있습니다...")
    item_means = np.zeros(user_item_matrix.shape[1])
    for i in tqdm(range(user_item_matrix.shape[1]), desc="아이템 평균 계산"):
        ratings = user_item_matrix[:, i].data
        if len(ratings) > 0:
            item_means[i] = np.mean(ratings)
    return item_means

def train_and_evaluate_model(ratings_df, model_filename='book_recommendation_model.pkl'):
    """
    모델을 훈련하고 평가합니다.
    
    :param ratings_df: 평점 데이터 DataFrame
    :param model_filename: 모델 파일 이름
    :return: user_item_matrix, user_id_map, item_id_map, item_means
    """
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    if os.path.exists(model_filename):
        print("기존 모델을 로드합니다.")
        try:
            model_data = load_model(model_filename)
            if isinstance(model_data, tuple) and len(model_data) == 4:
                user_item_matrix, user_id_map, item_id_map, item_means = model_data
            else:
                print("저장된 모델 데이터 형식이 올바르지 않습니다. 새 모델을 생성합니다.")
                raise ValueError
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}. 새 모델을 생성합니다.")
            user_item_matrix, user_id_map, item_id_map = create_user_item_matrix(train_df)
            item_means = calculate_item_means(user_item_matrix)
            save_model((user_item_matrix, user_id_map, item_id_map, item_means), model_filename)
    else:
        print("새로운 모델을 학습합니다.")
        user_item_matrix, user_id_map, item_id_map = create_user_item_matrix(train_df)
        item_means = calculate_item_means(user_item_matrix)
        save_model((user_item_matrix, user_id_map, item_id_map, item_means), model_filename)
    
    print("모델을 평가하고 있습니다...")
    y_true, y_pred = evaluate_model(test_df, user_item_matrix, user_id_map, item_id_map)
    
    if len(y_true) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"\n모델 성능:")
        print(f"RMSE: {rmse:.4f}")
    else:
        print("\n경고: 테스트 데이터에서 예측을 수행할 수 없었습니다.")
    
    return user_item_matrix, user_id_map, item_id_map, item_means

def evaluate_model(test_df, user_item_matrix, user_id_map, item_id_map):
    """
    모델을 평가합니다.
    
    :param test_df: 테스트 데이터 DataFrame
    :param user_item_matrix: 사용자-아이템 행렬
    :param user_id_map: 사용자 ID 매핑 딕셔너리
    :param item_id_map: 아이템 ID 매핑 딕셔너리
    :return: y_true, y_pred (실제 값과 예측 값 리스트)
    """
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        if row['User-ID'] in user_id_map.values() and row['ISBN'] in item_id_map.values():
            user_id = list(user_id_map.keys())[list(user_id_map.values()).index(row['User-ID'])]
            item_id = list(item_id_map.keys())[list(item_id_map.values()).index(row['ISBN'])]
            
            if user_id < user_item_matrix.shape[0] and item_id < user_item_matrix.shape[1]:
                user_ratings = user_item_matrix[user_id].data
                item_ratings = user_item_matrix[:, item_id].data
                
                if len(user_ratings) > 0 and len(item_ratings) > 0:
                    user_mean = np.mean(user_ratings)
                    item_mean = np.mean(item_ratings)
                    pred = (user_mean + item_mean) / 2
                    
                    y_true.append(row['Rating'])
                    y_pred.append(pred)
    
    return y_true, y_pred