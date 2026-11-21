
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "jogos_integrados.csv"
)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("ANÁLISE DA RELAÇÃO ENTRE NOTAS E VENDAS")
print("=" * 60)

# Correlações
correlacao_criticos = df[
    ["metascore", "total_sales"]
].corr().iloc[0, 1]

correlacao_usuarios = df[
    ["user_score", "total_sales"]
].corr().iloc[0, 1]

print("\nCorrelação entre Metascore e vendas:")
print(round(correlacao_criticos, 4))

print("\nCorrelação entre nota dos usuários e vendas:")
print(round(correlacao_usuarios, 4))

# Gráfico 1: Metascore e vendas
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="metascore",
    y="total_sales",
    alpha=0.35
)

plt.title("Metascore x Vendas Globais")
plt.xlabel("Metascore")
plt.ylabel("Vendas (milhões de unidades)")
plt.tight_layout()

plt.savefig(
    RESULTADOS / "metascore_vs_vendas.png",
    dpi=300
)

plt.close()

# Gráfico 2: Nota dos usuários e vendas
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="user_score",
    y="total_sales",
    alpha=0.35
)

plt.title("Nota dos Usuários x Vendas Globais")
plt.xlabel("Nota dos usuários")
plt.ylabel("Vendas (milhões de unidades)")
plt.tight_layout()

plt.savefig(
    RESULTADOS / "usuarios_vs_vendas.png",
    dpi=300
)

plt.close()

print("\nGráficos gerados:")
print("- metascore_vs_vendas.png")
print("- usuarios_vs_vendas.png")

print("\nANÁLISE CONCLUÍDA!")
