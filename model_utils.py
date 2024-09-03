import pickle

def save_model(model_data, filename):
    """
    모델 데이터를 파일로 저장합니다.
    
    :param model_data: 저장할 모델 데이터 (튜플 형태)
    :param filename: 저장할 파일 이름
    """
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(filename):
    """
    파일에서 모델 데이터를 로드합니다.
    
    :param filename: 로드할 파일 이름
    :return: 로드된 모델 데이터 (튜플 형태)
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)