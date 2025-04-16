import pandas as pd 

def end_str(text):
    if not isinstance(text, str):  # NaN 등 비문자열 방지
        return text
    if text.endswith('.'):  # 이미 마침표가 있으면 그대로 반환
        return text
    return text + '.'

def prepro_data(path) : 
    
    train = pd.read_csv(path , index_col = 0)

    # 부위 나누기
    train['부위'] = train['부위'].str.split('/').str[0].str.strip()

    train = train.drop_duplicates().reset_index(drop = True)

    train['재발방지대책 및 향후조치계획'] = train['재발방지대책 및 향후조치계획'].apply(end_str)

    return train 
