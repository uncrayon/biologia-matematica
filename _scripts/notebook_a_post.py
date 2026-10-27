"""
Convierte un notebook (ya ejecutado) de `_notebooks/` en una entrada del blog.

Uso
---
    python _scripts/notebook_a_post.py _notebooks/mp1-p1-memorizacion.ipynb

Qué hace
--------
1. Lee los metadatos del blog guardados en el propio notebook, en
   `metadata["blog"]` (título, fecha, categorías y un resumen corto).
2. Exporta el notebook a Markdown con nbconvert (código, texto y salidas).
3. Guarda las figuras en `assets/images/<nombre-del-notebook>/` para que
   Jekyll las publique (las carpetas que empiezan con "_" no se publican).
4. Escribe `_posts/<fecha>-<nombre-del-notebook>.md` con su front matter.

Detalles importantes
--------------------
* Todo el cuerpo va dentro de `{% raw %} ... {% endraw %}` para que Liquid
  (el motor de plantillas de Jekyll) no interprete llaves `{{ }}` que pueden
  aparecer en el código o en las fórmulas.
* Las imágenes son la excepción: se cierra el bloque raw solo alrededor de
  cada imagen para poder usar el filtro `relative_url` (respeta `baseurl`).
* El resumen va en el front matter como `excerpt`; así la página principal
  no intenta cortar un extracto a la mitad de un bloque raw.
"""

import argparse
import pathlib
import re

import nbformat
from nbconvert import MarkdownExporter

# Carpeta raíz del repositorio (este archivo vive en <raíz>/_scripts/).
RAIZ = pathlib.Path(__file__).resolve().parent.parent

# Plantilla del front matter de Jekyll. Las comillas de los textos se escapan
# al llenarla (ver `_texto_yaml`).
FRONT_MATTER = """---
layout: post
title: {titulo}
date: {fecha}
categories: {categorias}
excerpt: {resumen}
---
"""

# Enlaces al notebook original (GitHub para verlo y Colab para ejecutarlo).
ENLACES = (
    "**Notebook:** "
    "[ver en GitHub](https://github.com/uncrayon/biologia-matematica/blob/main/_notebooks/{nombre}.ipynb)"
    " · [abrir en Colab](https://colab.research.google.com/github/uncrayon/biologia-matematica/blob/main/_notebooks/{nombre}.ipynb)\n"
)


def _texto_yaml(texto):
    """Devuelve `texto` entre comillas dobles, escapando lo necesario para YAML."""
    return '"' + texto.replace("\\", "\\\\").replace('"', '\\"') + '"'


def convertir(ruta_notebook):
    """Convierte un notebook en una entrada del blog y regresa la ruta del post."""
    ruta_notebook = pathlib.Path(ruta_notebook).resolve()
    nombre = ruta_notebook.stem  # p. ej. "mp1-p1-memorizacion"
    nb = nbformat.read(ruta_notebook, as_version=4)
    meta = nb.metadata["blog"]

    # La primera celda empieza con el título (# ...). En el blog el título ya
    # lo pone el layout, así que quitamos SOLO esa línea (el resto de la celda,
    # normalmente la introducción, se conserva).
    primera = nb.cells[0]
    if primera.cell_type == "markdown" and primera.source.lstrip().startswith("# "):
        primera.source = primera.source.lstrip().partition("\n")[2].lstrip("\n")
        if not primera.source.strip():
            nb.cells = nb.cells[1:]

    # Exportamos a Markdown. Con `output_files_dir` controlamos el prefijo
    # con el que nbconvert nombra las figuras dentro del Markdown.
    exportador = MarkdownExporter()
    cuerpo, recursos = exportador.from_notebook_node(
        nb, resources={"output_files_dir": nombre, "unique_key": nombre}
    )

    # Guardamos cada figura en assets/images/<nombre>/. Primero borramos las
    # figuras viejas para no dejar archivos huérfanos al regenerar.
    carpeta_img = RAIZ / "assets" / "images" / nombre
    carpeta_img.mkdir(parents=True, exist_ok=True)
    for vieja in carpeta_img.glob("*.png"):
        vieja.unlink()
    for archivo, contenido in recursos.get("outputs", {}).items():
        (carpeta_img / pathlib.Path(archivo).name).write_bytes(contenido)

    # Reescribimos los enlaces de las figuras: ![png](nombre/output_3_0.png)
    # -> salir de raw, usar relative_url y volver a entrar en raw.
    patron = re.compile(r"!\[[^\]]*\]\(" + re.escape(nombre) + r"/([^)]+)\)")
    cuerpo = patron.sub(
        lambda m: (
            "{% endraw %}![Figura]({{ '/assets/images/"
            + nombre + "/" + m.group(1)
            + "' | relative_url }}){% raw %}"
        ),
        cuerpo,
    )

    front = FRONT_MATTER.format(
        titulo=_texto_yaml(meta["title"]),
        fecha=meta["date"],
        categorias=meta.get("categories", "miniproyecto-1"),
        resumen=_texto_yaml(meta["resumen"]),
    )
    fecha_archivo = meta["date"][:10]  # AAAA-MM-DD
    ruta_post = RAIZ / "_posts" / f"{fecha_archivo}-{nombre}.md"
    ruta_post.write_text(
        front + "\n" + ENLACES.format(nombre=nombre) + "\n{% raw %}\n" + cuerpo.strip() + "\n{% endraw %}\n",
        encoding="utf-8",
    )
    return ruta_post


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="+", help="notebooks ejecutados a convertir")
    args = parser.parse_args()
    for ruta in args.notebooks:
        print("Escrito:", convertir(ruta))
