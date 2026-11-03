"""
Descarga los datos reales que usa la clase de modelos epidémicos y los reduce
a series de tiempo pequeñas (unos cientos de KB) que se guardan en `_data/`.

Uso
---
    python _scripts/datos_epidemias_mexico.py            # todo
    python _scripts/datos_epidemias_mexico.py --solo flunet vacunas

Fuentes
-------
1. Secretaría de Salud, Dirección General de Epidemiología (DGE), datos abiertos
   de enfermedad respiratoria viral (COVID-19 e influenza). Un registro por
   persona notificada al SISVER. Cierres anuales 2020-2025 y el archivo vigente
   (2026):
       https://www.gob.mx/salud/documentos/datos-abiertos-152127
       https://www.gob.mx/salud/documentos/datos-abiertos-bases-historicas-direccion-general-de-epidemiologia
2. OMS, FluNet: casos de influenza confirmados por laboratorio en México, por
   semana epidemiológica.
       https://www.who.int/tools/flunet
3. Our World in Data: dosis de vacuna contra COVID-19 aplicadas en México
   (compiladas de los comunicados de la Secretaría de Salud y de la OMS/OPS).
       https://github.com/owid/covid-19-data

Qué se guarda
-------------
* `_data/covid_mx_diario.csv`: conteos diarios, para el país y para la Ciudad
  de México (entidad de residencia 09):
    - registros:            personas notificadas (casos sospechosos estudiados)
    - muestras:             personas a las que se tomó PCR o prueba de antígeno
    - covid_confirmados:    casos confirmados de COVID-19 (clasificación 1, 2 o 3)
    - covid_conf_60mas:     de ellos, los de 60 años o más
    - covid_hospitalizados: confirmados que fueron hospitalizados
    - flu_confirmados:      casos confirmados de influenza (solo desde que la
                            base incluye la variable, a mediados de 2024)
  todos por FECHA DE INICIO DE SÍNTOMAS, y
    - covid_defunciones, covid_def_60mas: defunciones de casos confirmados,
                            por FECHA DE DEFUNCIÓN.
* `_data/flunet_mx_semanal.csv`: FluNet, México.
* `_data/vacunacion_covid_mx.csv`: OWID, México.

Los archivos de la DGE pesan cientos de MB comprimidos. Se leen por bloques
directamente del zip y nunca se guardan descomprimidos.
"""

import argparse
import io
import pathlib
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DATOS = RAIZ / "_data"

URL_DGE = "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos"
ARCHIVOS_DGE = {
    2020: f"{URL_DGE}/historicos/2020/COVID19MEXICO2020.zip",
    2021: f"{URL_DGE}/historicos/2021/COVID19MEXICO2021.zip",
    2022: f"{URL_DGE}/historicos/2022/COVID19MEXICO2022.zip",
    2023: f"{URL_DGE}/historicos/2023/COVID19MEXICO2023.zip",
    2024: f"{URL_DGE}/historicos/2024/COVID19MEXICO2024.zip",
    2025: f"{URL_DGE}/historicos/2025/COVID19MEXICO2025.zip",
    2026: f"{URL_DGE}/datos_abiertos_influenza_covid19.zip",  # archivo vigente
}

URL_FLUNET = (
    "https://xmart-api-public.who.int/FLUMART/VIW_FNT"
    "?$format=csv&$filter=COUNTRY_CODE%20eq%20%27MEX%27"
)
URL_OWID = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/vaccinations/country_data/Mexico.csv"
)

# Columnas que nos interesan. Los nombres cambiaron con el tiempo: hasta 2024 la
# clasificación se llamaba CLASIFICACION_FINAL; después, CLASIFICACION_FINAL_COVID.
COLUMNAS = [
    "FECHA_INGRESO", "FECHA_SINTOMAS", "FECHA_DEF", "ENTIDAD_RES", "TIPO_PACIENTE",
    "EDAD", "TOMA_MUESTRA_LAB", "TOMA_MUESTRA_ANTIGENO",
    "CLASIFICACION_FINAL", "CLASIFICACION_FINAL_COVID", "CLASIFICACION_FINAL_FLU",
]


def descargar(url, destino):
    """Descarga `url` a `destino` por bloques (los archivos pueden pesar >100 MB)."""
    peticion = urllib.request.Request(url, headers={"User-Agent": "biologia-matematica"})
    with urllib.request.urlopen(peticion, timeout=600) as r, open(destino, "wb") as f:
        while True:
            bloque = r.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)


def resumir_bloque(df):
    """Convierte un bloque de registros individuales en conteos diarios."""
    clasif = "CLASIFICACION_FINAL_COVID" if "CLASIFICACION_FINAL_COVID" in df else "CLASIFICACION_FINAL"
    conf = df[clasif].isin([1, 2, 3])
    mayor = df["EDAD"] >= 60
    muestra = (df["TOMA_MUESTRA_LAB"] == 1)
    if "TOMA_MUESTRA_ANTIGENO" in df:
        muestra |= (df["TOMA_MUESTRA_ANTIGENO"] == 1)
    murio = df["FECHA_DEF"] != "9999-99-99"
    flu = df["CLASIFICACION_FINAL_FLU"] == 3 if "CLASIFICACION_FINAL_FLU" in df else None

    partes = []
    for ambito, filtro in [("Nacional", np.ones(len(df), bool)),
                           ("CDMX", (df["ENTIDAD_RES"].astype(str).str.zfill(2) == "09").to_numpy())]:
        d = df[filtro]
        # Por fecha de inicio de síntomas.
        indicadores = pd.DataFrame({
            "fecha": d["FECHA_SINTOMAS"],
            "registros": 1,
            "muestras": muestra[filtro].astype(int),
            "covid_confirmados": conf[filtro].astype(int),
            "covid_conf_60mas": (conf & mayor)[filtro].astype(int),
            "covid_hospitalizados": (conf & (df["TIPO_PACIENTE"] == 2))[filtro].astype(int),
            "flu_confirmados": flu[filtro].astype(int) if flu is not None else 0,
        })
        s = indicadores.groupby("fecha").sum()
        # Por fecha de defunción (solo casos confirmados que murieron).
        m = (conf & murio)[filtro]
        defs = pd.DataFrame({
            "fecha": d.loc[m, "FECHA_DEF"],
            "covid_defunciones": 1,
            "covid_def_60mas": mayor[filtro][m].astype(int),
        }).groupby("fecha").sum()
        s = s.join(defs, how="outer").fillna(0)
        s["ambito"] = ambito
        partes.append(s.reset_index())
    return pd.concat(partes)


def procesar_dge(anio, url, tmp):
    zip_local = tmp / f"dge_{anio}.zip"
    print(f"[DGE {anio}] descargando {url}", flush=True)
    descargar(url, zip_local)
    print(f"[DGE {anio}] {zip_local.stat().st_size / 1e6:.0f} MB; leyendo", flush=True)
    resumenes, ingreso_min, ingreso_max, n = [], "9999", "0000", 0
    with zipfile.ZipFile(zip_local) as z:
        nombre = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
        with z.open(nombre) as f:
            texto = io.TextIOWrapper(f, encoding="latin-1")
            encabezado = pd.read_csv(texto, nrows=0).columns
            usar = [c for c in COLUMNAS if c in encabezado]
        with z.open(nombre) as f:
            texto = io.TextIOWrapper(f, encoding="latin-1")
            for bloque in pd.read_csv(texto, usecols=usar, dtype={"ENTIDAD_RES": str},
                                      chunksize=1_000_000, low_memory=False):
                n += len(bloque)
                ingreso_min = min(ingreso_min, bloque["FECHA_INGRESO"].min())
                ingreso_max = max(ingreso_max, bloque["FECHA_INGRESO"].max())
                resumenes.append(resumir_bloque(bloque))
    zip_local.unlink()
    print(f"[DGE {anio}] {n:,} registros; FECHA_INGRESO de {ingreso_min} a {ingreso_max}", flush=True)
    r = pd.concat(resumenes).groupby(["ambito", "fecha"]).sum().reset_index()
    r["archivo"] = anio
    return r


def dge():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        todos = [procesar_dge(a, u, tmp) for a, u in ARCHIVOS_DGE.items()]
    df = pd.concat(todos)
    # Cada persona aparece en un solo archivo anual (el del año de su ingreso),
    # pero sus fechas de síntomas o de defunción pueden caer en el año siguiente:
    # por eso sumamos sobre archivos.
    df = df.drop(columns="archivo").groupby(["ambito", "fecha"]).sum().reset_index()
    df = df[(df["fecha"] >= "2020-01-01") & (df["fecha"] <= "2026-12-31")]
    columnas = ["registros", "muestras", "covid_confirmados", "covid_conf_60mas",
                "covid_hospitalizados", "flu_confirmados", "covid_defunciones", "covid_def_60mas"]
    df[columnas] = df[columnas].astype(int)
    df = df[["fecha", "ambito"] + columnas].sort_values(["ambito", "fecha"])
    df.to_csv(DATOS / "covid_mx_diario.csv", index=False)
    print(f"Guardado {DATOS / 'covid_mx_diario.csv'} ({len(df)} filas)")


def flunet():
    print("[FluNet] descargando", flush=True)
    df = pd.read_csv(URL_FLUNET)
    columnas = ["ISO_YEAR", "ISO_WEEK", "ISO_WEEKSTARTDATE", "ORIGIN_SOURCE",
                "SPEC_PROCESSED_NB", "INF_A", "INF_B", "INF_ALL", "INF_NEGATIVE"]
    df = df[[c for c in columnas if c in df.columns]].sort_values("ISO_WEEKSTARTDATE")
    df.to_csv(DATOS / "flunet_mx_semanal.csv", index=False)
    print(f"Guardado {DATOS / 'flunet_mx_semanal.csv'} ({len(df)} filas)")


def vacunas():
    print("[OWID] descargando", flush=True)
    df = pd.read_csv(URL_OWID)
    df = df[["date", "total_vaccinations", "people_vaccinated",
             "people_fully_vaccinated", "total_boosters"]]
    df.to_csv(DATOS / "vacunacion_covid_mx.csv", index=False)
    print(f"Guardado {DATOS / 'vacunacion_covid_mx.csv'} ({len(df)} filas)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solo", nargs="*", choices=["dge", "flunet", "vacunas"],
                   default=["dge", "flunet", "vacunas"])
    for tarea in p.parse_args().solo:
        {"dge": dge, "flunet": flunet, "vacunas": vacunas}[tarea]()
