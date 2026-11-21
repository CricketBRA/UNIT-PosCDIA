
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

# Localizar a pasta do projeto
BASE = Path(__file__).resolve().parent
PASTA_DADOS = BASE / "dados"

# Encontrar os arquivos ZIP
arquivos_zip = sorted(PASTA_DADOS.glob("*.zip"))

print("=" * 65)
print("ANÁLISE INICIAL DOS DATASETS")
print("=" * 65)

print(f"\nQuantidade de arquivos ZIP: {len(arquivos_zip)}")

for caminho_zip in arquivos_zip:

    print("\n" + "=" * 65)
    print("ARQUIVO ZIP:", caminho_zip.name)
    print("=" * 65)

    with ZipFile(caminho_zip, "r") as zip_file:

        # Localizar todos os CSVs dentro do ZIP
        arquivos_csv = [
            nome for nome in zip_file.namelist()
            if nome.lower().endswith(".csv")
        ]

        print("\nArquivos CSV encontrados:")
        for nome in arquivos_csv:
            print("-", nome)

        for nome_csv in arquivos_csv:

            print("\n" + "-" * 65)
            print("ANALISANDO:", nome_csv)
            print("-" * 65)

            # Ler o CSV diretamente do ZIP
            conteudo = zip_file.read(nome_csv)

            try:
                df = pd.read_csv(BytesIO(conteudo), low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(
                    BytesIO(conteudo),
                    encoding="latin-1",
                    low_memory=False
                )

            # Dimensões do dataset
            print("\nLinhas e colunas:", df.shape)

            # Nomes das colunas
            print("\nColunas:")
            print(df.columns.tolist())

            # Primeiros registros
            print("\nPrimeiros 3 registros:")
            print(df.head(3).to_string(index=False))

            # Valores ausentes
            print("\nValores ausentes:")
            print(
                df.isna()
                .sum()
                .sort_values(ascending=False)
                .to_string()
            )

            # Linhas duplicadas
            print("\nLinhas totalmente duplicadas:",
                  df.duplicated().sum())

print("\n" + "=" * 65)
print("DIAGNÓSTICO INICIAL CONCLUÍDO!")
print("=" * 65)
