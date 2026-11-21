
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "previsoes_random_forest.csv"
)

print("=" * 65)
print("ANÁLISE DOS ERROS DO RANDOM FOREST")
print("=" * 65)

# Calcular a diferença entre previsão e valor real
df["erro"] = (
    df["vendas_previstas"] - df["total_sales"]
)

df["erro_absoluto"] = df["erro"].abs()

print("\nQUANTIDADE DE PREVISÕES:", len(df))

print("\n10 MAIORES ERROS:")
print(
    df.nlargest(10, "erro_absoluto")[
        [
            "name",
            "platform",
            "total_sales",
            "vendas_previstas",
            "erro_absoluto"
        ]
    ].to_string(index=False)
)

print("\n10 MENORES ERROS:")
print(
    df.nsmallest(10, "erro_absoluto")[
        [
            "name",
            "platform",
            "total_sales",
            "vendas_previstas",
            "erro_absoluto"
        ]
    ].to_string(index=False)
)

print("\nTENDÊNCIA DAS PREVISÕES:")

superestimadas = (df["erro"] > 0).sum()
subestimadas = (df["erro"] < 0).sum()

print("Superestimadas:", superestimadas)
print("Subestimadas:", subestimadas)

print("\nPREVISÕES POR FAIXA DE ERRO:")

print(
    "Erro até 100 mil unidades:",
    (df["erro_absoluto"] <= 0.1).sum()
)

print(
    "Erro até 500 mil unidades:",
    (df["erro_absoluto"] <= 0.5).sum()
)

print(
    "Erro até 1 milhão de unidades:",
    (df["erro_absoluto"] <= 1).sum()
)

# Gráfico: vendas reais x previstas
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=df,
    x="total_sales",
    y="vendas_previstas",
    alpha=0.5
)

limite = max(
    df["total_sales"].max(),
    df["vendas_previstas"].max()
)

plt.plot(
    [0, limite],
    [0, limite],
    linestyle="--",
    color="red",
    label="Previsão perfeita"
)

plt.title("Random Forest: Vendas Reais x Previstas")
plt.xlabel("Vendas reais (milhões)")
plt.ylabel("Vendas previstas (milhões)")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTADOS / "random_forest_reais_vs_previstas.png",
    dpi=300
)

plt.close()

print("\nGráfico gerado:")
print("- random_forest_reais_vs_previstas.png")

print("\nANÁLISE CONCLUÍDA!")
