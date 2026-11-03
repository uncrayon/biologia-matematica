#!/usr/bin/env bash
# Exporta los cuadernos de marimo de la clase de modelos epidémicos como páginas
# web interactivas (WebAssembly) dentro de assets/epidemias/, y copia ahí los datos
# que necesitan. Las páginas se publican tal cual con el resto del sitio.
#
# Uso (desde la raíz del repositorio):
#     bash _scripts/exportar_epidemias.sh
#
# Requiere: pip install marimo
set -euo pipefail
cd "$(dirname "$0")/.."

destino=assets/epidemias
mkdir -p "$destino/datos"

for n in 1 2; do
  marimo export html-wasm "_notebooks/epidemias-sesion$n.py" \
    -o "$destino/sesion$n.html" --single-file --mode run --show-code -f
done

# Los datos que lee la sesión 2 (ver `leer_datos` en el cuaderno).
cp _data/covid_mx_diario.csv _data/vacunacion_covid_mx.csv _data/flunet_mx_semanal.csv "$destino/datos/"

echo "Listo: $destino"
ls -la "$destino" "$destino/datos"
