
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

arquivos = {
    "Random Forest": "previsoes_random_forest.csv",
    "Random Forest Log": "previsoes_random_forest_log.csv"
}

tabelas = []

for nome_modelo, arquivo in arquivos.items():
    df = pd.read_csv(RESULTADOS / arquivo)

    df["modelo"] = nome_modelo

    df["erro"] = (
        df["vendas_previstas"] - df["total_sales"]
    )

    df["erro_absoluto"] = df["erro"].abs()

    df["erro_quadrado"] = df["erro"] ** 2

    df["faixa_vendas"] = pd.cut(
        df["total_sales"],
        bins=[-0.001, 0.1, 0.5, 1, 5, float("inf")],
        labels=[
            "Até 100 mil",
            "100 a 500 mil",
            "500 mil a 1 milhão",
            "1 a 5 milhões",
            "Acima de 5 milhões"
        ]
    )

    resumo = df.groupby(
        "faixa_vendas",
        observed=True
    ).agg(
        quantidade=("total_sales", "size"),
        previsao_media=("vendas_previstas", "mean"),
        erro_medio=("erro", "mean"),
        MAE=("erro_absoluto", "mean"),
        MSE=("erro_quadrado", "mean")
    )

    resumo["RMSE"] = np.sqrt(resumo["MSE"])
    resumo = resumo.drop(columns=["MSE"])

    resumo["modelo"] = nome_modelo

    tabelas.append(resumo.reset_index())

comparacao = pd.concat(
    tabelas,
    ignore_index=True
)

print("=" * 75)
print("COMPARAÇÃO DOS MODELOS POR FAIXA DE VENDAS")
print("=" * 75)

print(
    comparacao[
        [
            "modelo",
            "faixa_vendas",
            "quantidade",
            "previsao_media",
            "erro_medio",
            "MAE",
            "RMSE"
        ]
    ].round(4).to_string(index=False)
)

comparacao.to_csv(
    RESULTADOS / "comparacao_modelos_faixas.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- comparacao_modelos_faixas.csv")

print("\nCOMPARAÇÃO CONCLUÍDA!")
