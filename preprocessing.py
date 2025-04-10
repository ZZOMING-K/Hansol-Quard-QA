import pandas as pd

def read_tabular_data(df_path):
    df = pd.read_csv(df_path)
    return df

def preprocessing_data(df, **kwargs):
    drop_list = kwargs['drop_list']
    df = df.drop(drop_list, axis=1)

    # nan_deal_option : 결측치 어떻게 처리할지에 대한 방법
    # 0번 : 결측치 제거
    # 1번 : 결측치 특정값으로 채우기
    if kwargs['nan_option'] == 0:
        df.dropna(inplace=True)
    if kwargs['nan_option'] == 1:
        # 컬럼명(key), 채울값(value)로 구성된 dictionary load
        fill_dict = kwargs['fill_dict']

        # 채우기
        # 예시 :
        # train['인적사고'] = train['인적사고'].fillna('알 수 없음')
        # train['물적사고'] = train['물적사고'].fillna('알 수 없음')
        # train['공종'] = train['공종'].fillna('알 수 없음 > 알 수 없음')
        # train['사고객체'] = train['사고객체'].fillna('알 수 없음 > 알 수 없음')
        # train['작업프로세스'] = train['작업프로세스'].fillna('알 수 없음')
        # train['사고원인'] = train['사고원인'].fillna('알 수 없음')
        for k, v in fill_dict.itmes():
            df[k] = df[k].fillna(v)


    return df




