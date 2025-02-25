from ler_dados import load_datasets, preprocess_data
from knn_holdout import train_and_evaluate

def main():
    # Carregar dados
    print("Carregando dados...")
    train_df, test_df = load_datasets()

    # Pré-processar
    print("Pré-processando dados...")
    processed_train = preprocess_data(train_df)
    processed_test = preprocess_data(test_df)  # Processado mas não usado

    # Separar features e target
    X = processed_train.drop('Transported', axis=1)
    y = processed_train['Transported']

    # Treinar e avaliar
    print("Treinando modelo...")
    accuracy, model = train_and_evaluate(X, y)
    print(f"\nAcurácia na validação: {accuracy:.2f}")
    #print(f"Modelo treinado: {model}")

if __name__ == "__main__":
    main()