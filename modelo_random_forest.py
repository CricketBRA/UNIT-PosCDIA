
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "jogos_integrados.csv"
)

# Separação temporal igual à experiência anterior
treino = df[df["ano_metacritic"] <= 2015].copy()
teste = df[df["ano_metacritic"] >= 2016].copy()

variaveis = [
    "metascore",
    "user_score",
    "plataforma_padrao",
    "genre_metacritic",
    "ano_metacritic"
]

alvo = "total_sales"

X_treino = treino[variaveis]
y_treino = treino[alvo]

X_teste = teste[variaveis]
y_teste = teste[alvo]

numericas = [
    "metascore",
    "user_score",
    "ano_metacritic"
]

categoricas = [
    "plataforma_padrao",
    "genre_metacritic"
]

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas
        ),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            categoricas
        )
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocessamento", preprocessamento),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

referencia = DummyRegressor(strategy="median")

referencia.fit(X_treino, y_treino)
modelo.fit(X_treino, y_treino)

previsao_referencia = referencia.predict(X_teste)
previsao_modelo = modelo.predict(X_teste)


def avaliar(nome, real, previsto):
    mae = mean_absolute_error(real, previsto)
    rmse = np.sqrt(mean_squared_error(real, previsto))
    r2 = r2_score(real, previsto)

    print(f"\n{nome}")
    print(f"MAE: {mae:.4f} milhões")
    print(f"RMSE: {rmse:.4f} milhões")
    print(f"R²: {r2:.4f}")

    return {
        "modelo": nome,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


print("=" * 60)
print("EXPERIMENTO RANDOM FOREST")
print("=" * 60)

print("\nRegistros de treinamento:", len(treino))
print("Registros de teste:", len(teste))

resultados = []

resultados.append(
    avaliar(
        "Referência (mediana)",
        y_teste,
        previsao_referencia
    )
)

resultados.append(
    avaliar(
        "Random Forest",
        y_teste,
        previsao_modelo
    )
)

pd.DataFrame(resultados).to_csv(
    RESULTADOS / "metricas_random_forest.csv",
    index=False,
    encoding="utf-8-sig"
)

previsoes = teste[
    [
        "name",
        "platform",
        "ano_metacritic",
        "metascore",
        "user_score",
        "total_sales"
    ]
].copy()

previsoes["vendas_previstas"] = previsao_modelo
previsoes["erro_absoluto"] = abs(
    previsoes["total_sales"]
    - previsoes["vendas_previstas"]
)

previsoes.to_csv(
    RESULTADOS / "previsoes_random_forest.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos gerados:")
print("- metricas_random_forest.csv")
print("- previsoes_random_forest.csv")

print("\nEXPERIMENTO CONCLUÍDO!")
