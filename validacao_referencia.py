
from pathlib import Path
import pandas as pd
import numpy as np

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

periodos = [
    (2012, 2013, 2014),
    (2014, 2015, 2016),
    (2016, 2017, 2018)
]

resultados = []

for fim_treino, inicio_teste, fim_teste in periodos:

    treino = df[
        df["ano_metacritic"] <= fim_treino
    ].copy()

    teste = df[
        (df["ano_metacritic"] >= inicio_teste)
        & (df["ano_metacritic"] <= fim_teste)
    ].copy()

    # O DummyRegressor não utiliza efetivamente
    # as características dos jogos.
    X_treino = treino[["metascore"]]
    X_teste = teste[["metascore"]]

    y_treino = treino["total_sales"]
    y_teste = teste["total_sales"]

    modelo = DummyRegressor(strategy="median")

    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)

    mae = mean_absolute_error(y_teste, previsoes)
    rmse = np.sqrt(
        mean_squared_error(y_teste, previsoes)
    )
    r2 = r2_score(y_teste, previsoes)

    resultados.append({
        "treino_ate": fim_treino,
        "teste": f"{inicio_teste}-{fim_teste}",
        "modelo": "Referência (mediana)",
        "quantidade_teste": len(teste),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

referencia = pd.DataFrame(resultados)

modelos_ml = pd.read_csv(
    RESULTADOS / "validacao_temporal.csv"
)

comparacao = pd.concat(
    [referencia, modelos_ml],
    ignore_index=True
)

comparacao = comparacao.sort_values(
    ["teste", "MAE"]
)

print("=" * 75)
print("VALIDAÇÃO TEMPORAL COM MODELO DE REFERÊNCIA")
print("=" * 75)

print(
    comparacao.round(4).to_string(index=False)
)

comparacao.to_csv(
    RESULTADOS / "validacao_temporal_com_referencia.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- validacao_temporal_com_referencia.csv")

print("\nCOMPARAÇÃO CONCLUÍDA!")
