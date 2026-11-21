
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

treino = df[df["ano_metacritic"] <= 2015].copy()
teste = df[df["ano_metacritic"] >= 2016].copy()

configuracoes = {
    "Sem notas": [
        "plataforma_padrao",
        "genre_metacritic",
        "ano_metacritic"
    ],
    "Apenas Metascore": [
        "metascore",
        "plataforma_padrao",
        "genre_metacritic",
        "ano_metacritic"
    ],
    "Todas as notas": [
        "metascore",
        "user_score",
        "plataforma_padrao",
        "genre_metacritic",
        "ano_metacritic"
    ]
}

resultados = []

for nome, variaveis in configuracoes.items():

    print(f"\nTreinando: {nome}")

    X_treino = treino[variaveis]
    X_teste = teste[variaveis]

    y_treino = treino["total_sales"]
    y_teste = teste["total_sales"]

    numericas = [
        coluna for coluna in variaveis
        if coluna in [
            "metascore",
            "user_score",
            "ano_metacritic"
        ]
    ]

    categoricas = [
        coluna for coluna in variaveis
        if coluna in [
            "plataforma_padrao",
            "genre_metacritic"
        ]
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

    modelo = TransformedTargetRegressor(
        regressor=floresta,
        func=np.log1p,
        inverse_func=np.expm1
    )

    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    previsoes = np.maximum(previsoes, 0)

    mae = mean_absolute_error(y_teste, previsoes)
    rmse = np.sqrt(
        mean_squared_error(y_teste, previsoes)
    )
    r2 = r2_score(y_teste, previsoes)

    resultados.append({
        "modelo": nome,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

comparacao = pd.DataFrame(resultados)

print("\n" + "=" * 65)
print("COMPARAÇÃO DO IMPACTO DAS NOTAS")
print("=" * 65)

print(
    comparacao.round(4).to_string(index=False)
)

comparacao.to_csv(
    RESULTADOS / "comparacao_impacto_notas.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- comparacao_impacto_notas.csv")

print("\nEXPERIMENTO CONCLUÍDO!")
