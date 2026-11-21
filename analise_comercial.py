
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. CONFIGURACAO
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)

CAMINHO_DADOS = RESULTADOS / "vendas_preparadas.csv"

if not CAMINHO_DADOS.exists():
    raise FileNotFoundError(
        "Arquivo vendas_preparadas.csv nao encontrado. "
        "Execute limpeza_dados.py primeiro."
    )

df = pd.read_csv(CAMINHO_DADOS)

COLUNAS = [
    "genre",
    "console",
    "publisher",
    "release_year",
    "total_sales"
]

faltando = [col for col in COLUNAS if col not in df.columns]

if faltando:
    raise ValueError(f"Colunas ausentes: {faltando}")

# As vendas estao expressas em milhoes de unidades.
df["total_sales"] = pd.to_numeric(
    df["total_sales"], errors="coerce"
)

df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("ANALISE COMERCIAL DOS VIDEOGAMES")
print("=" * 65)

print(f"\nRegistros analisados: {len(df):,}")
print(f"Vendas zeradas: {(df['total_sales'] == 0).sum():,}")
print(f"Vendas ausentes: {df['total_sales'].isna().sum():,}")

print(
    f"Vendas totais registradas: "
    f"{df['total_sales'].sum():,.2f} milhoes"
)


# --------------------------------------------------
# 2. FUNCAO PARA RANKINGS
# --------------------------------------------------

def criar_ranking(coluna, nome_arquivo, titulo, top_n=10):
    dados = df.dropna(subset=[coluna, "total_sales"])

    ranking = (
        dados.groupby(coluna)["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    total = df["total_sales"].sum()

    ranking["participacao_percentual"] = (
        ranking["total_sales"] / total * 100
        if total > 0 else 0
    )

    ranking.to_csv(
        RESULTADOS / f"{nome_arquivo}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{titulo.upper()}")
    print("-" * 65)
    print(ranking.head(top_n).round(2).to_string(index=False))

    top = ranking.head(top_n).sort_values("total_sales")

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=top,
        x="total_sales",
        y=coluna,
        color="#4472C4"
    )

    plt.title(titulo)
    plt.xlabel("Vendas acumuladas (milhoes de unidades)")
    plt.ylabel("")
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"{nome_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return ranking


# --------------------------------------------------
# 3. GENEROS
# --------------------------------------------------

ranking_generos = criar_ranking(
    coluna="genre",
    nome_arquivo="ranking_generos",
    titulo="Top 10 generos por vendas registradas"
)


# --------------------------------------------------
# 4. PLATAFORMAS
# --------------------------------------------------

ranking_plataformas = criar_ranking(
    coluna="console",
    nome_arquivo="ranking_plataformas",
    titulo="Top 10 plataformas por vendas registradas"
)


# --------------------------------------------------
# 5. EDITORAS
# --------------------------------------------------

ranking_editoras = criar_ranking(
    coluna="publisher",
    nome_arquivo="ranking_editoras",
    titulo="Top 10 editoras por vendas registradas"
)


# --------------------------------------------------
# 6. EVOLUCAO POR ANO DE LANCAMENTO
# --------------------------------------------------

dados_anuais = df.dropna(
    subset=["release_year", "total_sales"]
).copy()

dados_anuais["release_year"] = (
    dados_anuais["release_year"].astype(int)
)

evolucao = (
    dados_anuais.groupby("release_year")
    .agg(
        vendas_milhoes=("total_sales", "sum"),
        quantidade_registros=("total_sales", "size")
    )
    .reset_index()
    .sort_values("release_year")
)

evolucao.to_csv(
    RESULTADOS / "evolucao_vendas_anuais.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nEVOLUCAO DAS VENDAS POR ANO")
print("-" * 65)
print(evolucao.round(2).to_string(index=False))

plt.figure(figsize=(13, 6))

sns.lineplot(
    data=evolucao,
    x="release_year",
    y="vendas_milhoes",
    marker="o",
    color="#4472C4"
)

plt.title("Vendas registradas por ano de lancamento")
plt.xlabel("Ano de lancamento")
plt.ylabel("Vendas acumuladas (milhoes de unidades)")
plt.tight_layout()

plt.savefig(
    RESULTADOS / "evolucao_vendas_anuais.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 7. FINALIZACAO
# --------------------------------------------------

print("\n" + "=" * 65)
print("ANALISE COMERCIAL CONCLUIDA!")
print("=" * 65)

print("\nArquivos gerados na pasta resultados:")
print("- ranking_generos.csv")
print("- ranking_generos.png")
print("- ranking_plataformas.csv")
print("- ranking_plataformas.png")
print("- ranking_editoras.csv")
print("- ranking_editoras.png")
print("- evolucao_vendas_anuais.csv")
print("- evolucao_vendas_anuais.png")
