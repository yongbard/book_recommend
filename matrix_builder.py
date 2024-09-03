from scipy.sparse import csr_matrix
from tqdm import tqdm

def create_user_item_matrix(ratings_df):
    """
    사용자-아이템 행렬을 생성합니다.
    
    :param ratings_df: 평점 데이터 DataFrame
    :return: 희소 행렬 (CSR 형식), user_id_map, item_id_map
    """
    print("사용자-아이템 행렬을 생성하고 있습니다...")
    user_ids = ratings_df['User-ID'].astype('category')
    item_ids = ratings_df['ISBN'].astype('category')
    
    user_id_map = dict(enumerate(user_ids.cat.categories))
    item_id_map = dict(enumerate(item_ids.cat.categories))
    
    row = user_ids.cat.codes
    col = item_ids.cat.codes
    data = ratings_df['Rating'].values
    
    shape = (len(user_id_map), len(item_id_map))
    return csr_matrix((data, (row, col)), shape=shape), user_id_map, item_id_map