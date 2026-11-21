
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

# Mesma divisão temporal dos experimentos anteriores
treino = df[df["ano_metacritic"] <= 2015].copy()
teste = df[df["ano_metacritic"] >= 2016].copy()

# Retiramos apenas o ano das variáveis de entrada
variaveis = [
    "metascore",
    "user_score",
    "plataforma_padrao",
    "genre_metacritic"
]

X_treino = treino[variaveis]
y_treino = treino["total_sales"]

X_teste = teste[variaveis]
y_teste = teste["total_sales"]

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            ["metascore", "user_score"]
        ),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            ["plataforma_padrao", "genre_metacritic"]
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

modelo = TransformedTargetRegressor(
    regressor=floresta,
    func=np.log1p,
    inverse_func=np.expm1
)

modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
previsoes = np.maximum(previsoes, 0)

mae = mean_absolute_error(y_teste, previsoes)
rmse = np.sqrt(mean_squared_error(y_teste, previsoes))
r2 = r2_score(y_teste, previsoes)

print("=" * 60)
print("RANDOM FOREST LOG - SEM ANO DE LANÇAMENTO")
print("=" * 60)

print("\nTreinamento:", len(treino))
print("Teste:", len(teste))

print("\nRESULTADOS:")
print(f"MAE: {mae:.4f} milhões")
print(f"RMSE: {rmse:.4f} milhões")
print(f"R²: {r2:.4f}")

pd.DataFrame([
    {
        "modelo": "Random Forest Log sem ano",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
]).to_csv(
    RESULTADOS / "metricas_sem_ano.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- metricas_sem_ano.csv")

print("\nEXPERIMENTO CONCLUÍDO!")
