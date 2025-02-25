import pandas as pd

def load_datasets():
    """Carrega dados de treino e teste"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def preprocess_data(df):
    """Pré-processa um DataFrame (funciona para treino e teste)"""
    # Processar PassengerId
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['Number_in_Group'] = df['PassengerId'].str.split('_').str[1].astype(int)

    # Processar Cabin
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Cabin_num'] = cabin_split[1].astype(float) if not cabin_split[1].isnull().all() else 0.0
    df['Side'] = cabin_split[2]

    # Remover colunas não utilizadas
    df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
    return df