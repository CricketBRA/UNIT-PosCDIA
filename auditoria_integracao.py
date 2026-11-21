
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "jogos_integrados.csv"
)

print("=" * 65)
print("AUDITORIA DA INTEGRAÇÃO DOS DATASETS")
print("=" * 65)

print("\nTotal de registros:", len(df))

print("\nCOLUNAS DISPONÍVEIS:")
print(df.columns.tolist())

# Conferir anos de lançamento
colunas_anos = [
    c for c in df.columns
    if "ano" in c.lower()
    or "year" in c.lower()
]

print("\nCOLUNAS RELACIONADAS A ANOS:")
print(colunas_anos)

if (
    "ano_metacritic" in df.columns
    and "release_year" in df.columns
):
    df["diferenca_anos"] = (
        df["ano_metacritic"]
        - df["release_year"]
    ).abs()

    print("\nDIFERENÇAS ENTRE ANOS:")
    print(
        df["diferenca_anos"]
        .value_counts()
        .sort_index()
    )

    divergentes = df[
        df["diferenca_anos"] > 0
    ].copy()

    print("\nEXEMPLOS DE ANOS DIVERGENTES:")

    colunas_exibir = [
        c for c in [
            "name",
            "platform",
            "ano_metacritic",
            "release_year",
            "metascore",
            "total_sales"
        ]
        if c in divergentes.columns
    ]

    print(
        divergentes[colunas_exibir]
        .head(20)
        .to_string(index=False)
    )

    divergentes.to_csv(
        RESULTADOS / "auditoria_anos_divergentes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArquivo gerado:")
    print("- auditoria_anos_divergentes.csv")

else:
    print(
        "\nNão foi possível identificar "
        "automaticamente as duas colunas de ano."
    )

print("\nAUDITORIA CONCLUÍDA!")
