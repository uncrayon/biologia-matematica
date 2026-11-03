---
layout: post
title: "Modelos epidémicos: de las ecuaciones a los datos de México"
date: 2026-11-02 20:00:00 -0600
categories: epidemias
excerpt: "Dos sesiones interactivas sobre modelos compartimentales (SI, SIS, SIR, SIRS, SEIR, vacunación): equilibrios, estabilidad, bifurcaciones y R0; ajuste a datos reales de influenza y COVID-19 en México; y simulaciones estocásticas para ver cuándo dejan de valer las ecuaciones."
---

Material para dos sesiones de una hora sobre **modelos epidémicos compartimentales**. Las dos partes son cuadernos de [marimo](https://marimo.io) que corren **directamente en el navegador** (Python compilado a WebAssembly): no hay que instalar nada, y todas las figuras tienen controles para mover los parámetros.

<p>
<a href="{{ '/assets/epidemias/sesion1.html' | relative_url }}"><strong>▶ Sesión 1 · Teoría</strong></a>
&nbsp;·&nbsp;
<a href="{{ '/assets/epidemias/sesion2.html' | relative_url }}"><strong>▶ Sesión 2 · Datos de México y límites del modelo</strong></a>
</p>

*La primera vez que se abre una página tarda entre 30 segundos y un minuto: el navegador descarga Python y las bibliotecas científicas. Conviene abrirlas antes de la clase. Funcionan mejor en una computadora que en un teléfono.*

## Sesión 1 · Teoría

1. **¿Qué estamos modelando?** Las variables del modelo no son personas: son cantidades continuas (densidades, como la biomasa en ecología). La ecuación diferencial es el límite de población infinita de un proceso aleatorio con personas enteras (Kurtz); una simulación compara ambos para distintos tamaños de población.
2. **La incidencia**: cómo se construye $\beta SI/N$ (incidencia estándar) frente a $\beta SI$ (acción de masas), y por qué la elección cambia el umbral.
3. **Adimensionalización**: el SIR pasa de tres parámetros a uno solo, $R_0$; y la población $N$ desaparece del modelo.
4. **SI, SIS, SIR, SIR con demografía, SIRS y SEIR**, cada uno con sus equilibrios, jacobianos, estabilidad y diagramas de bifurcación. Se muestra por qué el SIR **no** tiene una bifurcación (tiene una recta de equilibrios no hiperbólicos y un umbral), mientras que el SIS, el SIR con demografía y el SIRS tienen una bifurcación transcrítica en $R_0 = 1$.
5. **Cómo calcular $R_0$**: directamente, como umbral de estabilidad, con la matriz de siguiente generación (con `sympy`, incluido un modelo de dengue con vector) y a partir de la tasa de crecimiento.
6. **Vacunación**: cobertura crítica $p_c = (1 - 1/R_0)/e$ como bifurcación en el parámetro de cobertura, y una campaña de vacunación durante una epidemia.

## Sesión 2 · Datos de México y límites del modelo

1. **Los datos**: qué mide realmente el sistema de vigilancia (pruebas, retrasos, el cambio a vigilancia centinela de octubre de 2023, las vacaciones en los reportes de influenza).
2. **Identificabilidad**: adimensionalizando con la población susceptible efectiva se demuestra que una curva de casos solo determina $R_{\text{ef}} = R_0 s_0$ y $K = \rho s_0 N$, no $R_0$, $s_0$ ni la fracción de reporte por separado.
3. **Tasa de crecimiento y ajuste completo** (SIR y SEIR) para la temporada de influenza 2025-2026 y la segunda ola de COVID-19 (nacional y Ciudad de México).
4. **Vacunación**: por qué la curva de casos no sirve para evaluar la vacuna, qué sí se ve en los datos (la proporción de defunciones de personas de 60 años o más) y un escenario SEIR con la cobertura real de México.
5. **Poblaciones pequeñas**: simulaciones exactas de Gillespie con $N = 4, 10, 100, 1000$ comparadas con la EDO y con la distribución exacta; extinción segura en el SIS finito; hogares y el umbral $R_* = R_G \mu_h$.
6. **Resumen**: cuándo sirve el modelo y cuándo no.

## Datos

Los datos son reales y se descargan de las fuentes oficiales con [`_scripts/datos_epidemias_mexico.py`](https://github.com/uncrayon/biologia-matematica/blob/main/_scripts/datos_epidemias_mexico.py), que reduce unos 21 millones de registros a tablas pequeñas en [`_data/`](https://github.com/uncrayon/biologia-matematica/tree/main/_data):

* **Secretaría de Salud, Dirección General de Epidemiología**: [datos abiertos](https://www.gob.mx/salud/documentos/datos-abiertos-152127) de enfermedad respiratoria viral (COVID-19 e influenza), cierres anuales 2020-2025 y el archivo vigente de 2026. Conteos diarios por fecha de inicio de síntomas, para el país y la Ciudad de México (`covid_mx_diario.csv`).
* **OMS, FluNet**: influenza confirmada por laboratorio en México por semana (`flunet_mx_semanal.csv`).
* **Our World in Data**: vacunas contra COVID-19 aplicadas en México (`vacunacion_covid_mx.csv`).

## Para trabajar con los cuadernos en tu computadora

Los cuadernos fuente son [`_notebooks/epidemias-sesion1.py`](https://github.com/uncrayon/biologia-matematica/blob/main/_notebooks/epidemias-sesion1.py) y [`_notebooks/epidemias-sesion2.py`](https://github.com/uncrayon/biologia-matematica/blob/main/_notebooks/epidemias-sesion2.py). Son archivos de Python normales; para abrirlos y editarlos:

```bash
pip install marimo numpy scipy matplotlib pandas sympy
marimo edit _notebooks/epidemias-sesion1.py
```

Para actualizar los datos y volver a generar las páginas web:

```bash
python _scripts/datos_epidemias_mexico.py
bash _scripts/exportar_epidemias.sh
```
