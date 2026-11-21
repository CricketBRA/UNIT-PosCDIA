
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

variaveis = [
    "metascore",
    "user_score",
    "plataforma_padrao",
    "genre_metacritic",
    "ano_metacritic"
]

bases = {
    "Base completa": df,
    "Anos coincidentes": df[
        df["diferenca_anos"] == 0
    ].copy()
}

resultados = []

for nome_base, dados in bases.items():

    treino = dados[
        dados["ano_metacritic"] <= 2015
    ].copy()

    teste = dados[
        dados["ano_metacritic"] >= 2016
    ].copy()

    X_treino = treino[variaveis]
    X_teste = teste[variaveis]

    y_treino = treino["total_sales"]
    y_teste = teste["total_sales"]

    preprocessamento = ColumnTransformer(
        transformers=[
            (
                "numericas",
                SimpleImputer(strategy="median"),
                [
                    "metascore",
                    "user_score",
                    "ano_metacritic"
                ]
            ),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                [
                    "plataforma_padrao",
                    "genre_metacritic"
                ]
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

    previsoes = np.maximum(
        modelo.predict(X_teste),
        0
    )

    mae = mean_absolute_error(
        y_teste,
        previsoes
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_teste,
            previsoes
        )
    )

    r2 = r2_score(
        y_teste,
        previsoes
    )

    resultados.append({
        "base": nome_base,
        "registros": len(dados),
        "treinamento": len(treino),
        "teste": len(teste),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

comparacao = pd.DataFrame(resultados)

print("=" * 75)
print("TESTE DE SENSIBILIDADE DA INTEGRAÇÃO")
print("=" * 75)

print(
    comparacao.round(4).to_string(index=False)
)

comparacao.to_csv(
    RESULTADOS / "teste_sensibilidade_integracao.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- teste_sensibilidade_integracao.csv")

print("\nTESTE CONCLUÍDO!")
