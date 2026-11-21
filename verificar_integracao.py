
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

arquivo = RESULTADOS / "jogos_integrados.csv"

df = pd.read_csv(arquivo)

print("=" * 65)
print("VERIFICAÇÃO DA BASE INTEGRADA")
print("=" * 65)

print("\nQuantidade de registros:", len(df))
print("Quantidade de colunas:", len(df.columns))

# Verificar valores ausentes
print("\nVALORES AUSENTES:")
print(
    df.isna()
    .sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
    .to_string()
)

# Verificar duplicações
print("\nLinhas exatamente duplicadas:")
print(df.duplicated().sum())

print("\nJogos repetidos por nome, plataforma e ano:")
print(
    df.duplicated(
        subset=[
            "nome_normalizado",
            "plataforma_padrao",
            "ano_metacritic"
        ]
    ).sum()
)

# Verificar valores numéricos
for coluna in ["metascore", "user_score", "total_sales"]:
    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )

print("\nNOTAS E VENDAS:")
print(
    df[
        ["metascore", "user_score", "total_sales"]
    ].describe().to_string()
)

print("\nVALORES FORA DOS LIMITES ESPERADOS:")

print(
    "Metascore fora de 0 a 100:",
    (~df["metascore"].between(0, 100)).sum()
)

print(
    "User Score fora de 0 a 10:",
    (~df["user_score"].between(0, 10)).sum()
)

print(
    "Vendas negativas ou iguais a zero:",
    (df["total_sales"] <= 0).sum()
)

# Comparar anos das duas bases
print("\nDIFERENÇA ENTRE OS ANOS DE LANÇAMENTO:")
print(
    df["diferenca_anos"]
    .value_counts()
    .sort_index()
    .to_string()
)

# Distribuição temporal
print("\nREGISTROS POR ANO:")
print(
    df["ano_metacritic"]
    .value_counts()
    .sort_index()
    .to_string()
)

# Distribuição das vendas
print("\nFAIXAS DE VENDAS:")

faixas = pd.cut(
    df["total_sales"],
    bins=[0, 0.1, 0.5, 1, 5, float("inf")],
    labels=[
        "Até 100 mil",
        "100 mil a 500 mil",
        "500 mil a 1 milhão",
        "1 a 5 milhões",
        "Acima de 5 milhões"
    ],
    include_lowest=True
)

print(faixas.value_counts().to_string())

print("\nVERIFICAÇÃO CONCLUÍDA!")
