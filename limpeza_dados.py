
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import unicodedata
import re

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(exist_ok=True)


def ler_csv_do_zip(parte_nome_zip, nome_csv):
    arquivos = [
        caminho for caminho in DADOS.glob("*.zip")
        if parte_nome_zip.lower() in caminho.name.lower()
    ]

    if len(arquivos) != 1:
        raise FileNotFoundError(
            f"Esperado um ZIP contendo '{parte_nome_zip}'. "
            f"Encontrados: {len(arquivos)}"
        )

    with ZipFile(arquivos[0], "r") as arquivo_zip:
        conteudo = arquivo_zip.read(nome_csv)

    return pd.read_csv(BytesIO(conteudo), low_memory=False)


def normalizar_nome(nome):
    if pd.isna(nome):
        return ""

    nome = str(nome).strip().lower()

    nome = unicodedata.normalize("NFKD", nome)
    nome = "".join(
        letra for letra in nome
        if not unicodedata.combining(letra)
    )

    nome = re.sub(r"[^a-z0-9]+", " ", nome)
    nome = re.sub(r"\s+", " ", nome).strip()

    return nome


# 1. Carregar Metacritic
metacritic = ler_csv_do_zip(
    "Metacritic Video Games Reviews",
    "metacritic_dataset_clean.csv"
)

# 2. Carregar vendas
vendas = ler_csv_do_zip(
    "Video Game Sales & Industry Data",
    "Video_Games_Sales_Cleaned.csv"
)

print("=" * 60)
print("LIMPEZA DOS DATASETS")
print("=" * 60)

print("\nMETACRITIC")
print("Antes:", len(metacritic))

metacritic = metacritic.drop_duplicates().copy()

metacritic["nome_normalizado"] = (
    metacritic["name"].apply(normalizar_nome)
)

metacritic["release_date"] = pd.to_datetime(
    metacritic["release_date"],
    errors="coerce"
)

metacritic["ano"] = metacritic["release_date"].dt.year

print("Depois:", len(metacritic))
print("Datas inválidas:", metacritic["release_date"].isna().sum())

print("\nVENDAS")
print("Antes:", len(vendas))

vendas = vendas.drop_duplicates().copy()

vendas["nome_normalizado"] = (
    vendas["title"].apply(normalizar_nome)
)

vendas["ano"] = pd.to_numeric(
    vendas["release_year"],
    errors="coerce"
).astype("Int64")

print("Depois:", len(vendas))
print("Anos inválidos:", vendas["ano"].isna().sum())

# Salvar arquivos preparados
metacritic.to_csv(
    RESULTADOS / "metacritic_preparado.csv",
    index=False,
    encoding="utf-8-sig"
)

vendas.to_csv(
    RESULTADOS / "vendas_preparadas.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos:")
print("- metacritic_preparado.csv")
print("- vendas_preparadas.csv")

print("\nLIMPEZA CONCLUÍDA!")
