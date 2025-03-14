# Importar bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar e explorar os dados
def carregar_dados():
    iris = load_iris()
    X = iris.data  # Características (features)
    y = iris.target  # Rótulos (labels)
    print("Classes:", iris.target_names)
    print("Formato de X:", X.shape)
    print("Formato de y:", y.shape)
    return X, y, iris.target_names

# 2. Dividir os dados em treino e teste
def dividir_dados(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# 3. Pré-processar os dados
def preprocessar_dados(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# 4. Treinar o modelo
def treinar_modelo(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model

# 5. Avaliar o modelo
def avaliar_modelo(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Melhorar o modelo com validação cruzada
def validacao_cruzada(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print("Acurácia com validação cruzada:", scores.mean())

# 7. Ajustar hiperparâmetros com GridSearchCV
def ajustar_hiperparametros(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    return grid_search.best_estimator_

# Função principal para executar o pipeline
def main():
    # 1. Carregar dados
    X, y, target_names = carregar_dados()

    # 2. Dividir dados
    X_train, X_test, y_train, y_test = dividir_dados(X, y)

    # 3. Pré-processar dados
    X_train, X_test = preprocessar_dados(X_train, X_test)

    # 4. Treinar modelo
    model = treinar_modelo(X_train, y_train)

    # 5. Avaliar modelo
    avaliar_modelo(model, X_test, y_test, target_names)

    # 6. Validação cruzada
    validacao_cruzada(model, X, y)

    # 7. Ajustar hiperparâmetros
    best_model = ajustar_hiperparametros(X_train, y_train)

    # Avaliar o modelo ajustado
    print("\nAvaliando o modelo com os melhores hiperparâmetros:")
    avaliar_modelo(best_model, X_test, y_test, target_names)

# Executar o script
if __name__ == "__main__":
    main()
    