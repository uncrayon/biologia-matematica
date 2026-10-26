---
layout: post
title: "Miniproyecto 1: índice de soluciones"
date: 2026-10-26 00:08:00 -0600
categories: miniproyecto-1
excerpt: "Soluciones comentadas de los siete problemas del Miniproyecto 1 de Biología Matemática I, cada una con su entrada en el blog y su notebook de Python."
---

Estas son las soluciones comentadas de los siete problemas del **Miniproyecto 1** de Biología Matemática I. Cada problema tiene dos versiones con el mismo contenido:

* una **entrada del blog** (los enlaces de abajo), y
* un **notebook de Jupyter** en la carpeta [`_notebooks`](https://github.com/uncrayon/biologia-matematica/tree/main/_notebooks) del repositorio, que se puede ejecutar en Colab desde el enlace al inicio de cada entrada.

Todo el código está comentado en español y usa semillas fijas, así que al ejecutar los notebooks se obtienen exactamente los mismos resultados.

{% assign entradas = site.categories['miniproyecto-1'] | sort: 'date' | reverse %}
{% for entrada in entradas %}{% if entrada.url != page.url %}
* [{{ entrada.title | remove: "Miniproyecto 1 · " }}]({{ entrada.url | relative_url }}). {{ entrada.excerpt }}
{% endif %}{% endfor %}

## Notas importantes

* **Problema 1:** en lugar de datos reales del experimento de memorización usamos **datos simulados** de un participante virtual cuyo valor de κ conocemos. Así se puede comprobar si los métodos de estimación recuperan el valor correcto.
* **Problema 2:** la comparación con Tetschke et al. (2018) está escrita **de memoria**, sin acceso al artículo, y está marcada como tal: hay que verificarla con el original.
* **Problema 4:** no se pudo descargar el conjunto de datos de Zenodo (Benzekry et al., 2014); la comparación usa **datos simulados** con la misma estructura, claramente identificados como tales.
* **Problemas 6 y 7:** además de los notebooks (que reimplementan los modelos de NetLogo en Python para poder hacer estadística con cientos de simulaciones), están los modelos de NetLogo modificados: [`Ants-Levy.nlogo`]({{ '/assets/netlogo/Ants-Levy.nlogo' | relative_url }}) y [`Wolf-Sheep-Rebano.nlogo`]({{ '/assets/netlogo/Wolf-Sheep-Rebano.nlogo' | relative_url }}). Se escribieron sin poder ejecutar NetLogo, así que **no están probados**.

Para regenerar una entrada después de modificar su notebook:

```bash
python _scripts/notebook_a_post.py _notebooks/mp1-p1-memorizacion.ipynb
```
