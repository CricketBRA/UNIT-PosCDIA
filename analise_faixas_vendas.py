
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "previsoes_random_forest.csv"
)

# Calcular erros
df["erro"] = (
    df["vendas_previstas"] - df["total_sales"]
)

df["erro_absoluto"] = df["erro"].abs()

df["erro_quadrado"] = df["erro"] ** 2

# Criar faixas de vendas reais
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

# Estatísticas por faixa
resumo = df.groupby(
    "faixa_vendas",
    observed=True
).agg(
    quantidade=("total_sales", "size"),
    vendas_medias=("total_sales", "mean"),
    previsao_media=("vendas_previstas", "mean"),
    erro_medio=("erro", "mean"),
    MAE=("erro_absoluto", "mean"),
    MSE=("erro_quadrado", "mean")
)

resumo["RMSE"] = resumo["MSE"] ** 0.5

resumo = resumo.drop(columns=["MSE"])

print("=" * 70)
print("DESEMPENHO DO RANDOM FOREST POR FAIXA DE VENDAS")
print("=" * 70)

print("\nResultados por faixa:")
print(resumo.round(4).to_string())

print("\nTotal de registros:", len(df))

# Salvar tabela
resumo.to_csv(
    RESULTADOS / "desempenho_por_faixa.csv",
    encoding="utf-8-sig"
)

print("\nArquivo gerado:")
print("- desempenho_por_faixa.csv")

print("\nANÁLISE CONCLUÍDA!")
