
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
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

# Manter a mesma separação temporal
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

floresta = Pipeline(
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

# Treinar usando log(1 + vendas)
modelo_log = TransformedTargetRegressor(
    regressor=floresta,
    func=np.log1p,
    inverse_func=np.expm1
)

modelo_log.fit(X_treino, y_treino)

previsoes = modelo_log.predict(X_teste)

# Garantir previsões não negativas
previsoes = np.maximum(previsoes, 0)

mae = mean_absolute_error(y_teste, previsoes)
rmse = np.sqrt(
    mean_squared_error(y_teste, previsoes)
)
r2 = r2_score(y_teste, previsoes)

print("=" * 60)
print("RANDOM FOREST COM TRANSFORMAÇÃO LOGARÍTMICA")
print("=" * 60)

print("\nRegistros de treinamento:", len(treino))
print("Registros de teste:", len(teste))

print("\nRESULTADOS:")
print(f"MAE: {mae:.4f} milhões")
print(f"RMSE: {rmse:.4f} milhões")
print(f"R²: {r2:.4f}")

resultado = pd.DataFrame([
    {
        "modelo": "Random Forest Log",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
])

resultado.to_csv(
    RESULTADOS / "metricas_random_forest_log.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_previsoes = teste[
    [
        "name",
        "platform",
        "ano_metacritic",
        "metascore",
        "user_score",
        "total_sales"
    ]
].copy()

tabela_previsoes["vendas_previstas"] = previsoes

tabela_previsoes["erro_absoluto"] = abs(
    tabela_previsoes["total_sales"]
    - tabela_previsoes["vendas_previstas"]
)

tabela_previsoes.to_csv(
    RESULTADOS / "previsoes_random_forest_log.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos gerados:")
print("- metricas_random_forest_log.csv")
print("- previsoes_random_forest_log.csv")

print("\nEXPERIMENTO CONCLUÍDO!")
