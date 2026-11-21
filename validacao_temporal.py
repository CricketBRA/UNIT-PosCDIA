
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

periodos = [
    (2012, 2013, 2014),
    (2014, 2015, 2016),
    (2016, 2017, 2018)
]

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

for fim_treino, inicio_teste, fim_teste in periodos:

    treino = df[
        df["ano_metacritic"] <= fim_treino
    ].copy()

    teste = df[
        (df["ano_metacritic"] >= inicio_teste)
        & (df["ano_metacritic"] <= fim_teste)
    ].copy()

    print(
        f"\nPERÍODO: treino até {fim_treino}, "
        f"teste {inicio_teste}-{fim_teste}"
    )

    print("Treinamento:", len(treino))
    print("Teste:", len(teste))

    for nome, variaveis in configuracoes.items():

        X_treino = treino[variaveis]
        X_teste = teste[variaveis]

        y_treino = treino["total_sales"]
        y_teste = teste["total_sales"]

        numericas = [
            c for c in variaveis
            if c in [
                "metascore",
                "user_score",
                "ano_metacritic"
            ]
        ]

        categoricas = [
            c for c in variaveis
            if c in [
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

        mae = mean_absolute_error(
            y_teste,
            previsoes
        )

        rmse = np.sqrt(
            mean_squared_error(y_teste, previsoes)
        )

        r2 = r2_score(y_teste, previsoes)

        resultados.append({
            "treino_ate": fim_treino,
            "teste": f"{inicio_teste}-{fim_teste}",
            "modelo": nome,
            "quantidade_teste": len(teste),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

comparacao = pd.DataFrame(resultados)

print("\n" + "=" * 75)
print("RESULTADOS DA VALIDAÇÃO TEMPORAL")
print("=" * 75)

print(
    comparacao.round(4).to_string(index=False)
)

comparacao.to_csv(
    RESULTADOS / "validacao_temporal.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- validacao_temporal.csv")

print("\nVALIDAÇÃO CONCLUÍDA!")
