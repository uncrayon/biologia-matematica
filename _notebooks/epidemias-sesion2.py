import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Modelos epidémicos · Sesión 2: datos y límites",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelos epidémicos compartimentales
    ## Sesión 2 · Datos reales de México, estimación de parámetros y límites del modelo

    Cuaderno **interactivo**: todo se calcula en tu navegador. La primera carga tarda unos segundos (descarga Python y los datos).

    **Contenido**

    1. Los datos: qué se mide (y qué no).
    2. El modelo de observación y por qué no todos los parámetros se pueden estimar.
    3. Estimar el número reproductivo con la tasa de crecimiento.
    4. Ajustar el modelo completo (SIR y SEIR) a una ola: influenza 2025-2026 y COVID-19 2020-2021.
    5. Vacunación: qué muestran los datos y qué muestra el modelo.
    6. ¿Cuándo deja de valer el modelo? Simulaciones exactas (Gillespie) con $N = 4, 10, 100, 1000$.
    7. SIS en poblaciones finitas: la extinción es segura.
    8. Hogares: poblaciones pequeñas conectadas entre sí.
    9. Resumen: cuándo sí y cuándo no.

    **Fuentes de datos** (se descargan y resumen con `_scripts/datos_epidemias_mexico.py` del repositorio):

    * Secretaría de Salud, Dirección General de Epidemiología (DGE): datos abiertos de enfermedad respiratoria viral, un registro por persona notificada, de 2020 a octubre de 2026 (unos 21 millones de registros). Aquí usamos conteos diarios por **fecha de inicio de síntomas**.
    * OMS, FluNet: influenza confirmada por laboratorio en México, por semana.
    * Our World in Data: dosis de vacuna contra COVID-19 aplicadas en México.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import random
    import sys
    import pathlib
    from scipy.integrate import odeint
    from scipy.optimize import brentq, least_squares

    return brentq, least_squares, mo, np, odeint, pathlib, pd, plt, random, sys


@app.cell(hide_code=True)
def _(plt):
    COLOR = {
        "S": "#2a78d6",  # azul
        "I": "#eb6834",  # naranja
        "R": "#1baf7a",  # aqua
        "E": "#eda100",  # amarillo
        "V": "#e87ba4",  # magenta
    }
    SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # orden categórico fijo
    GRIS = "#9a9993"
    TINTA = "#52514e"
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#bdbcb6",
        "axes.labelcolor": TINTA,
        "axes.titlecolor": "#0b0b0b",
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "axes.grid": True,
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })
    return COLOR, GRIS, SERIE


@app.cell
def _(mo, pathlib, pd, sys):
    # En la versión web los datos se piden al mismo sitio donde vive esta página.
    RUTA_WEB = "/biologia-matematica/assets/epidemias/datos/"

    def leer_datos(nombre, **opciones):
        """Lee un CSV de `_data/` localmente o, en el navegador (WebAssembly), desde el sitio."""
        ubicacion = str(mo.notebook_location())
        if sys.platform == "emscripten":
            from urllib.parse import urlsplit
            from pyodide.http import open_url
            u = urlsplit(ubicacion.removeprefix("blob:"))
            return pd.read_csv(open_url(f"{u.scheme}://{u.netloc}{RUTA_WEB}{nombre}"), **opciones)
        return pd.read_csv(pathlib.Path(ubicacion).parent / "_data" / nombre, **opciones)

    # Conteos diarios de la DGE (por fecha de inicio de síntomas; defunciones por fecha de defunción).
    diario = leer_datos("covid_mx_diario.csv", parse_dates=["fecha"])
    vacunas = leer_datos("vacunacion_covid_mx.csv", parse_dates=["date"])
    flunet = leer_datos("flunet_mx_semanal.csv", parse_dates=["ISO_WEEKSTARTDATE"])

    POBLACION = {"Nacional": 126_014_024, "CDMX": 9_209_944}   # INEGI, Censo 2020

    def serie_diaria(ambito, columna):
        x = diario[diario["ambito"] == ambito].set_index("fecha")[columna]
        return x.asfreq("D", fill_value=0)

    def serie_semanal(ambito, columna, inicio=None, fin=None):
        """Suma semanal (semanas que empiezan en lunes)."""
        x = serie_diaria(ambito, columna)[inicio:fin]
        return x.resample("W-MON", label="left", closed="left").sum()

    return POBLACION, diario, flunet, serie_diaria, serie_semanal, vacunas


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Los datos: qué se mide (y qué no)

    En la sesión 1 insistimos en que $S$, $I$, $R$ son cantidades continuas que representan una población muy grande. **Los datos son otra cosa todavía**: son personas que (1) tuvieron síntomas, (2) fueron a una unidad de salud que notifica al sistema de vigilancia, (3) les tomaron una muestra y (4) salieron positivas. La figura muestra, para todo el país, cuántas personas se notificaron, a cuántas se les tomó muestra y cuántas se confirmaron.
    """)
    return


@app.cell(hide_code=True)
def _(GRIS, SERIE, pd, plt, serie_semanal):
    _cols = [("registros", "personas notificadas"), ("muestras", "con muestra"),
             ("covid_confirmados", "COVID-19 confirmado"), ("flu_confirmados", "influenza confirmada")]
    _fig, _ax = plt.subplots(figsize=(10.5, 4))
    for (_c, _nombre), _color in zip(_cols, SERIE):
        _s = serie_semanal("Nacional", _c, "2020-03-02", "2026-10-11")
        if _c == "flu_confirmados":
            _s = _s["2024-07-01":]
        _ax.plot(_s.index, _s.clip(lower=1), color=_color, lw=1.6, label=_nombre)
    _ax.axvline(pd.Timestamp("2023-10-01"), color=GRIS, ls=":", lw=1.5)
    _ax.text(pd.Timestamp("2023-10-20"), 3e5, "oct 2023: solo unidades\ncentinela (USMER)", fontsize=8, va="top")
    _ax.set(yscale="log", title="México: personas por semana (por fecha de inicio de síntomas)", ylabel="personas por semana", ylim=(10, 2e6))
    _ax.legend(loc="lower left", fontsize=8, ncol=2)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lo que hay que ver en esta figura antes de ajustar cualquier modelo:

    1. **Los confirmados siguen a las pruebas.** Cuando hubo más pruebas hubo más confirmados. El número de casos confirmados depende de la epidemia **y** de la política de pruebas.
    2. **En octubre de 2023 cambió el sistema, no la epidemia.** A partir de ese mes la base solo contiene notificaciones de las **unidades monitoras de enfermedad respiratoria (USMER)**, la red de vigilancia centinela; las demás unidades dejaron de aparecer (lo comprobamos en los registros: las de origen "no USMER" pasan de unas 50 000 al mes a cero). Las notificaciones caen unas 10 veces de golpe. Un modelo que tomara eso como una caída de los contagios estaría completamente equivocado.
    3. **La influenza solo aparece desde mediados de 2024**, cuando la base empezó a incluir su clasificación. Para años anteriores usamos FluNet.
    4. **Las últimas semanas siempre están incompletas** (retraso en la notificación y en los resultados de laboratorio): la caída al final de la serie es un artefacto.

    El siguiente ejemplo de artefacto es más sutil. Comparamos dos fuentes de la misma influenza: FluNet (OMS), que cuenta por **semana de notificación**, y la DGE contada por **fecha de inicio de síntomas**.
    """)
    return


@app.cell(hide_code=True)
def _(SERIE, flunet, mo, plt, serie_semanal):
    _fn = flunet.groupby("ISO_WEEKSTARTDATE")[["INF_ALL", "SPEC_PROCESSED_NB"]].sum()["2024-07-01":"2026-10-04"]
    _dge = serie_semanal("Nacional", "flu_confirmados", "2024-07-01", "2026-10-04")
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax[0].plot(_fn.index, _fn["INF_ALL"], color=SERIE[0], lw=1.6, label="FluNet (semana de reporte)")
    _ax[0].plot(_dge.index, _dge.values, color=SERIE[1], lw=1.6, label="DGE (inicio de síntomas)")
    _ax[0].set(title="Influenza confirmada por semana", ylabel="casos por semana")
    _ax[0].legend(fontsize=8)
    _ax[0].tick_params(axis="x", rotation=30)
    _fl = flunet.groupby("ISO_WEEKSTARTDATE")["INF_ALL"].sum()["2014-01-01":"2026-10-04"]
    _ax[1].plot(_fl.index, _fl.values, color=SERIE[0], lw=1.2)
    _ax[1].set(title="FluNet, México, 2014-2026", ylabel="casos por semana")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "Izquierda: FluNet muestra caídas bruscas en la última semana de cada año (diciembre de 2024 y de 2025): son las "
        "**vacaciones de fin de año** (se procesan menos muestras), no una pausa en los contagios. Por fecha de síntomas "
        "la curva es suave. Derecha: la influenza es **estacional** (cada invierno) y casi desaparece en 2020-2021, "
        "cuando las medidas contra la COVID-19 redujeron los contactos. Ninguna de esas dos cosas está en un SIR."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. El modelo de observación y la identificabilidad

    ### 2.1 Qué relaciona el modelo con los datos

    Supongamos una ola que sigue un SIR (o SEIR) en una población de $N$ personas, de las cuales una fracción $s_0$ es susceptible al inicio de la ola (las demás son inmunes por infecciones o vacunas previas). Lo que el modelo predice es la **incidencia**: el número de infecciones nuevas por unidad de tiempo, $\beta S I/N$. Los datos son una fracción $\rho$ de ellas (la **fracción de reporte**: las que tienen síntomas, se notifican y se confirman):

    $$
    C_k \approx \rho \times \big(\text{infecciones nuevas en la semana } k\big) = \rho \big[S(t_k) - S(t_{k+1})\big].
    $$

    ### 2.2 Adimensionalizar dice qué se puede estimar

    Escalemos con la **población susceptible efectiva** $s_0 N$: $x = \dfrac{S}{s_0 N}$, $y = \dfrac{I}{s_0 N}$. Como $\dfrac{dS}{dt} = -\beta \dfrac{SI}{N}$,
    $$
    s_0 N \frac{dx}{dt} = -\beta\,\frac{(s_0 N x)(s_0 N y)}{N} \quad\Longrightarrow\quad \frac{dx}{dt} = -(\beta s_0)\, x\, y, \qquad \frac{dy}{dt} = (\beta s_0)\, x\, y - \gamma y ,
    $$
    con $x(0) \approx 1$. Y los datos:
    $$
    C_k \approx \underbrace{\rho\, s_0\, N}_{K}\;\big[x(t_k) - x(t_{k+1})\big].
    $$

    La conclusión es exacta, no una aproximación numérica:

    * La **forma** de la curva depende solo de $\beta s_0$ y $\gamma$, es decir, de $R_{\text{ef}} = R_0\, s_0$ y de la escala de tiempo.
    * La **altura** depende solo del producto $K = \rho\, s_0\, N$.

    Por lo tanto, **con una curva de casos no se pueden separar** $R_0$ de $s_0$, ni $\rho$ de $s_0$ (ni de $N$). Infinitas combinaciones dan exactamente la misma curva. Lo que sí se puede estimar es $R_{\text{ef}}$, $K$ y, con menos precisión, $\gamma$. Para separar el resto se necesita **otra información**: encuestas serológicas (para $s_0$), estudios de la proporción de asintomáticos y de la búsqueda de atención médica (para $\rho$), estudios clínicos (para $\gamma$ y $\sigma$).

    **Esta es la versión cuantitativa de "el modelo no son personas"**: el "tamaño de población" que ve un ajuste a datos de casos es $K$, que puede ser mil veces menor que la población real.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    CONJUNTOS = {
        "Influenza, temporada 2025-2026 (nacional)": dict(
            ambito="Nacional", columna="flu_confirmados", inicio="2025-10-06", fin="2026-05-03",
            crecimiento=("2025-11-10", "2025-12-21"), latencia=1.5, infeccioso=2.0),
        "COVID-19, segunda ola 2020-2021 (nacional)": dict(
            ambito="Nacional", columna="covid_confirmados", inicio="2020-10-05", fin="2021-03-28",
            crecimiento=("2020-10-26", "2020-12-27"), latencia=3.0, infeccioso=2.5),
        "COVID-19, segunda ola 2020-2021 (Ciudad de México)": dict(
            ambito="CDMX", columna="covid_confirmados", inicio="2020-10-05", fin="2021-03-28",
            crecimiento=("2020-10-26", "2020-12-27"), latencia=3.0, infeccioso=2.5),
    }
    conjunto = mo.ui.dropdown(options=list(CONJUNTOS), value=list(CONJUNTOS)[0], label="Datos")
    return CONJUNTOS, conjunto


@app.cell(hide_code=True)
def _(CONJUNTOS, conjunto, mo, serie_semanal):
    datos_ola = CONJUNTOS[conjunto.value]
    casos_ola = serie_semanal(datos_ola["ambito"], datos_ola["columna"], datos_ola["inicio"], datos_ola["fin"])
    _semanas = [d.strftime("%Y-%m-%d") for d in casos_ola.index]
    _i0 = _semanas.index(datos_ola["crecimiento"][0])
    _i1 = _semanas.index(max(s for s in _semanas if s <= datos_ola["crecimiento"][1]))
    ventana = mo.ui.range_slider(0, len(_semanas) - 1, step=1, value=[_i0, _i1], label="Semanas de crecimiento (índices)", show_value=True)
    dur_latente = mo.ui.slider(0.5, 6.0, step=0.1, value=datos_ola["latencia"], label="latencia $1/\\sigma$ (días)", show_value=True)
    dur_infecc = mo.ui.slider(1.0, 10.0, step=0.1, value=datos_ola["infeccioso"], label="periodo infeccioso $1/\\gamma$ (días)", show_value=True)
    return casos_ola, datos_ola, dur_infecc, dur_latente, ventana


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. El número reproductivo a partir de la tasa de crecimiento

    Al inicio de una ola casi todos los que pueden infectarse siguen siendo susceptibles, y el modelo linealizado predice crecimiento exponencial: $C(t) \propto e^{rt}$, o sea $\ln C(t) = \text{const} + r t$. **Una recta en escala logarítmica.** Ajustamos esa recta por mínimos cuadrados en las semanas de crecimiento y usamos las fórmulas de la sesión 1:

    $$
    \text{SIR: } R = 1 + \frac{r}{\gamma}, \qquad \text{SEIR: } R = \left(1 + \frac{r}{\sigma}\right)\left(1 + \frac{r}{\gamma}\right).
    $$

    Dos advertencias:

    * Lo que obtenemos **no es $R_0$**: es el número reproductivo **efectivo al inicio de la ola**, $R_{\text{ef}} = R_0\, s_0 \times (\text{efecto de las medidas y del comportamiento})$. Ni la influenza de 2025 ni la segunda ola de COVID-19 empezaron en una población totalmente susceptible.
    * **$\rho$ no importa aquí** (si es constante): multiplicar los casos por una constante solo sube la recta, no cambia su pendiente. Pero **la escala de tiempo sí importa**: con la misma $r$, distintos supuestos sobre $1/\sigma$ y $1/\gamma$ dan distintos $R$. Los valores iniciales de los deslizadores son ilustrativos y dan intervalos generacionales ($1/\sigma + 1/\gamma$) del orden de los reportados: unos 3.5 días para influenza (Cowling et al., 2009) y unos 5 días para SARS-CoV-2 (Ganyani et al., 2020). Muévelos.
    """)
    return


@app.cell(hide_code=True)
def _(COLOR, GRIS, casos_ola, conjunto, dur_infecc, dur_latente, mo, np, plt, ventana):
    _a, _b = ventana.value
    _t = (casos_ola.index - casos_ola.index[0]).days.to_numpy().astype(float)
    _y = casos_ola.to_numpy().astype(float)
    _sel = slice(_a, _b + 1)
    _m = _y[_sel] > 0
    _pend, _ord = np.polyfit(_t[_sel][_m], np.log(_y[_sel][_m]), 1)
    _r = _pend
    _gam, _sig = 1 / dur_infecc.value, 1 / dur_latente.value
    _R_sir = 1 + _r / _gam
    _R_seir = (1 + _r / _sig) * (1 + _r / _gam)

    _fig, _ax = plt.subplots(figsize=(10.5, 3.8))
    _ax.semilogy(casos_ola.index, _y, "o-", color=COLOR["I"], ms=4, lw=1.2, label="casos confirmados por semana")
    _ax.semilogy(casos_ola.index[_sel], np.exp(_ord + _pend * _t[_sel]), color="#0b0b0b", lw=2, label=f"ajuste exponencial, $r$ = {_r:.4f} por día")
    _ax.axvspan(casos_ola.index[_a], casos_ola.index[_b], color=GRIS, alpha=0.12)
    _ax.set(title=conjunto.value, ylabel="casos por semana (escala log)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _dup = np.log(2) / _r if _r > 0 else np.inf
    mo.vstack([
        conjunto,
        mo.hstack([ventana, dur_latente, dur_infecc], justify="start", wrap=True),
        _fig,
        mo.md(
            f"Tasa de crecimiento $r = {_r:.4f}$ por día (tiempo de duplicación {_dup:.1f} días). "
            f"Con estos supuestos: **SIR $R_{{\\text{{ef}}}} = {_R_sir:.2f}$**, **SEIR $R_{{\\text{{ef}}}} = {_R_seir:.2f}$**. "
            "Cambia la ventana y verás que $r$ (y por lo tanto $R$) depende de qué semanas consideras \"crecimiento exponencial\"."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Ajustar el modelo completo a una ola

    Ahora ajustamos toda la curva. Siguiendo la sección 2, el modelo que podemos estimar tiene tres parámetros libres:

    * $R_{\text{ef}} = R_0 s_0$ (forma de la ola),
    * $K = \rho s_0 N$ (altura: "tamaño efectivo" de la población en unidades de casos reportados),
    * $y_0$, la fracción infectada (de la población susceptible efectiva) al inicio de la ventana,

    y dejamos fijos $\gamma$ (y $\sigma$ en el SEIR) con los deslizadores de arriba. El criterio de ajuste es mínimos cuadrados sobre $\sqrt{C_k}$ (la raíz estabiliza la varianza de conteos tipo Poisson, para que las semanas con muchos casos no dominen el ajuste).
    """)
    return


@app.cell
def _(brentq, least_squares, np, odeint):
    def incidencia_semanal(parametros, n_semanas, gamma, sigma, modelo):
        """Casos esperados por semana: K * (caída semanal de x) en el SIR o SEIR escalado."""
        R_ef, log_K, log_y0 = parametros
        K, y0 = np.exp(log_K), np.exp(log_y0)
        beta = R_ef * gamma                         # beta * s0, en 1/día
        t = np.arange(n_semanas + 1) * 7.0
        if modelo == "SIR":
            def f(z, t):
                x, y = z
                return [-beta * x * y, beta * x * y - gamma * y]
            z0 = [1 - y0, y0]
        else:                                       # SEIR: e(0) en proporción a i(0)
            def f(z, t):
                x, e, y = z
                return [-beta * x * y, beta * x * y - sigma * e, sigma * e - gamma * y]
            e0 = y0 * gamma / sigma
            z0 = [1 - y0 - e0, e0, y0]
        z = odeint(f, z0, t, rtol=1e-8, atol=1e-12)
        return K * np.maximum(-np.diff(z[:, 0]), 0)

    def ajustar_ola(casos, gamma, sigma, modelo):
        """Ajuste por mínimos cuadrados en raíz cuadrada, con varios puntos de arranque."""
        y = np.asarray(casos, float)
        residuos = lambda p: np.sqrt(incidencia_semanal(p, len(y), gamma, sigma, modelo)) - np.sqrt(y)
        cotas = ([1.0001, np.log(0.5 * y.sum()), -25], [20, np.log(1e4 * y.sum()), -0.7])
        mejor = None
        for R in (1.2, 2.0):
            for ly in (-9.0, -5.0):
                try:
                    r = least_squares(residuos, [R, np.log(2 * y.sum()), ly], bounds=cotas)
                except ValueError:
                    continue
                if mejor is None or r.cost < mejor.cost:
                    mejor = r
        return mejor.x

    def tamano_final(R):
        return 0.0 if R <= 1 else brentq(lambda z: z - 1 + np.exp(-R * z), 1e-12, 1.0)

    return ajustar_ola, incidencia_semanal, tamano_final


@app.cell(hide_code=True)
def _(mo):
    modelo_ajuste = mo.ui.dropdown(options=["SIR", "SEIR"], value="SEIR", label="Modelo")
    return (modelo_ajuste,)


@app.cell(hide_code=True)
def _(
    COLOR,
    GRIS,
    POBLACION,
    ajustar_ola,
    casos_ola,
    conjunto,
    datos_ola,
    dur_infecc,
    dur_latente,
    incidencia_semanal,
    mo,
    modelo_ajuste,
    np,
    plt,
    tamano_final,
):
    _y = casos_ola.to_numpy().astype(float)
    _gam, _sig = 1 / dur_infecc.value, 1 / dur_latente.value
    _p = ajustar_ola(_y, _gam, _sig, modelo_ajuste.value)
    _ajuste = incidencia_semanal(_p, len(_y), _gam, _sig, modelo_ajuste.value)
    _R_ef, _K = _p[0], np.exp(_p[1])
    _N = POBLACION[datos_ola["ambito"]]

    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    _ax[0].plot(casos_ola.index, _y, "o", color=COLOR["I"], ms=4, label="datos")
    _ax[0].plot(casos_ola.index, _ajuste, color="#0b0b0b", lw=2, label=f"{modelo_ajuste.value} ajustado")
    _ax[0].set(title=conjunto.value, ylabel="casos por semana")
    _ax[0].legend(fontsize=8)
    _ax[0].tick_params(axis="x", rotation=30)
    _ax[1].axhline(0, color=GRIS, lw=1)
    _ax[1].plot(casos_ola.index, (np.sqrt(_y) - np.sqrt(_ajuste)), "o-", color=GRIS, ms=3, lw=1)
    _ax[1].set(title="Residuos ($\\sqrt{datos} - \\sqrt{ajuste}$)")
    _ax[1].tick_params(axis="x", rotation=30)
    _fig.tight_layout()

    _filas = []
    for _rho in [1.0, 0.1, 0.01, 0.001]:
        _s0 = _K / (_rho * _N)
        _filas.append(
            f"| {_rho:g} | {_s0:.4f} | {_R_ef / _s0:.2f} |" if _s0 <= 1
            else f"| {_rho:g} | {_s0:.2f} (imposible: $s_0 > 1$) | — |"
        )
    mo.vstack([
        modelo_ajuste,
        _fig,
        mo.md(
            f"**Estimaciones:** $R_{{\\text{{ef}}}} = {_R_ef:.3f}$, $K = \\rho s_0 N = {_K:,.0f}$ casos reportados. "
            f"Casos totales que predice el modelo para la ola completa: $K \\cdot z(R_{{\\text{{ef}}}}) = {_K * tamano_final(_R_ef):,.0f}$ "
            f"(observados en la ventana: {_y.sum():,.0f}). La población real es $N = {_N:,}$: **$K$ es {_N / _K:,.0f} veces más pequeño**.\n\n"
            "**Mismo ajuste, historias distintas.** Todas las filas de esta tabla producen *exactamente* la misma curva negra:\n\n"
            "| fracción de reporte $\\rho$ | susceptibles al inicio $s_0 = K/(\\rho N)$ | $R_0 = R_{\\text{ef}}/s_0$ |\n|---:|---:|---:|\n"
            + "\n".join(_filas)
            + "\n\nLos datos de casos no pueden decidir entre ellas. Algunas filas se descartan con conocimiento biológico "
            "(con $\\rho = 1$, es decir, si se confirmara *toda* infección, casi nadie habría sido susceptible y $R_0$ sería absurdo), "
            "pero entre las razonables solo deciden encuestas serológicas o estudios de reporte."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Qué le falta al modelo (mira los residuos).** Las olas reales son **asimétricas**: suben más rápido de lo que bajan (o al revés) y tienen colas largas. Un SIR con parámetros constantes no puede hacer eso. Algunas razones:

    * **Las cosas cambian durante la ola**: comportamiento, medidas de salud pública, vacaciones (la segunda ola de COVID-19 coincidió con las fiestas de fin de año), clima. En el modelo eso sería $\beta(t)$, no $\beta$ constante.
    * **Heterogeneidad**: unas personas tienen muchos más contactos que otras. Los muy conectados se infectan primero, y la ola se frena antes de lo que predice la mezcla homogénea.
    * **Varias cepas a la vez** (en influenza circulan A(H1N1), A(H3N2) y B, con inmunidades distintas).
    * **La ola no termina solo por agotar susceptibles**: en la influenza, el final de la temporada también es estacional. El SIR atribuye todo el descenso a la inmunidad, y por eso su $R_{\text{ef}}$ y su $K$ deben leerse con cuidado.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Vacunación: qué muestran los datos y qué muestra el modelo

    México empezó a vacunar contra COVID-19 el 24 de diciembre de 2020 (personal de salud). El 15 de febrero de 2021 empezó la vacunación de personas de **60 años o más**, y los grupos más jóvenes se abrieron después, hasta mediados de 2021.

    Un error común es mirar la curva de casos y concluir algo sobre la vacuna. Veamos por qué no funciona y qué sí funciona.
    """)
    return


@app.cell(hide_code=True)
def _(COLOR, GRIS, POBLACION, SERIE, mo, pd, plt, serie_semanal, vacunas):
    _v = vacunas.set_index("date")
    _ini, _fin = "2020-06-01", "2022-06-26"
    _casos = serie_semanal("Nacional", "covid_confirmados", _ini, _fin)
    _muertes = serie_semanal("Nacional", "covid_defunciones", _ini, _fin)
    _c60 = serie_semanal("Nacional", "covid_conf_60mas", _ini, _fin)
    _d60 = serie_semanal("Nacional", "covid_def_60mas", _ini, _fin)
    _marca = pd.Timestamp("2021-02-15")

    _fig, _ax = plt.subplots(4, 1, figsize=(10.5, 10), sharex=True)
    _ax[0].plot(_v.index, 100 * _v["people_vaccinated"] / POBLACION["Nacional"], color=COLOR["V"], label="al menos una dosis")
    _ax[0].plot(_v.index, 100 * _v["people_fully_vaccinated"] / POBLACION["Nacional"], color=SERIE[0], label="esquema completo")
    _ax[0].set(title="Cobertura de vacunación (% de la población)", ylabel="%", ylim=(0, 100))
    _ax[0].legend(fontsize=8, loc="upper left")
    _ax[1].plot(_casos.index, _casos, color=COLOR["I"])
    _ax[1].set(title="Casos confirmados por semana", ylabel="casos")
    _ax[2].plot(_muertes.index, _muertes, color="#0b0b0b")
    _ax[2].set(title="Defunciones de casos confirmados por semana", ylabel="defunciones")
    _ax[3].plot(_c60.index, 100 * _c60 / _casos, color=COLOR["I"], label="% de los casos")
    _ax[3].plot(_d60.index, 100 * _d60 / _muertes, color="#0b0b0b", label="% de las defunciones")
    _ax[3].set(title="Proporción de personas de 60 años o más", ylabel="%", ylim=(0, 100))
    _ax[3].legend(fontsize=8, loc="center left")
    for _a in _ax:
        _a.axvline(_marca, color=GRIS, ls=":", lw=1.5)
    _ax[0].text(_marca, 90, "  inicia vacunación de 60+", fontsize=8, va="top")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lo que NO se puede concluir.** Con la vacunación avanzando, la **tercera ola** (julio-septiembre de 2021, variante Delta) fue tan grande como la segunda en casos, y la cuarta (enero de 2022, ómicron) fue la mayor de todas. ¿La vacuna no sirvió? Esa conclusión mezcla todo: llegaron variantes más transmisibles (mayor $R_0$), cambió el comportamiento, cambió la inmunidad por infección previa y cambió cuántas pruebas se hacían. La curva de casos responde a todo eso a la vez. **Comparar "antes y después" de la vacunación en la curva de casos no aísla el efecto de la vacuna.**

    **Lo que sí se puede ver.** La última gráfica compara dos grupos **dentro de la misma epidemia**, al mismo tiempo, con la misma variante y el mismo sistema de pruebas. Antes de febrero de 2021, cerca de dos tercios de las defunciones eran de personas de 60 años o más. Entre febrero y junio de 2021, cuando ese grupo se vacunó primero, su proporción en las defunciones cayó a menos de la mitad, y la proporción de 60+ entre los casos también bajó. Después, cuando se vacunaron los demás grupos, la proporción volvió a subir. Ese patrón es lo que esperaríamos si la vacuna protege, y es difícil de explicar de otra forma (aunque no es una prueba: también podrían haber cambiado los hábitos de prueba por edad). Es el mismo tipo de argumento que se usó en otros países con datos parecidos.

    **Lo que muestra el modelo.** En la sesión 1 vimos que la vacunación reduce $R_{\text{ef}} = R_0 s$, y que con una variante muy transmisible y una vacuna que protege poco contra la *infección*, la cobertura crítica es mayor que 100 %. Usemos la cobertura **real** de México para explorar una ola tipo Delta.

    **El modelo.** SEIR con vacunación desde el 1 de junio de 2021. La cobertura con al menos una dosis, $c(t)$, viene de los datos. Si la vacunación no depende de si alguien ya se infectó, la fracción de no vacunados que se vacuna por día es $u(t) = c'(t)/(1 - c(t))$, y una fracción $e$ (eficacia contra la infección) de los susceptibles vacunados queda protegida:
    $$
    \frac{dS}{dt} = -\beta \frac{S I}{N} - e\, u(t)\, S, \qquad
    \frac{dE}{dt} = \beta \frac{S I}{N} - \sigma E, \qquad
    \frac{dI}{dt} = \sigma E - \gamma I .
    $$
    Al inicio, $S(0)/N = s_0\,(1 - e\, c(0))$, donde $s_0$ es la fracción sin inmunidad por infección previa. **Es un escenario, no un ajuste**: no sabemos $s_0$ ni $e$ (sección 2), así que los dejamos como deslizadores.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_delta = mo.ui.slider(1.5, 8.0, step=0.1, value=5.0, label="$R_0$ de la variante", show_value=True)
    s0_delta = mo.ui.slider(0.2, 1.0, step=0.05, value=0.45, label="$s_0$: sin inmunidad previa", show_value=True)
    e_delta = mo.ui.slider(0.0, 0.95, step=0.05, value=0.6, label="eficacia contra infección $e$", show_value=True)
    return e_delta, r0_delta, s0_delta


@app.cell(hide_code=True)
def _(
    COLOR,
    GRIS,
    POBLACION,
    e_delta,
    mo,
    np,
    odeint,
    pd,
    plt,
    r0_delta,
    s0_delta,
    serie_semanal,
    vacunas,
):
    _ini = pd.Timestamp("2021-06-01")
    _dias = np.arange(0, 181)
    _fechas = _ini + pd.to_timedelta(_dias, unit="D")
    _v = vacunas.set_index("date")["people_vaccinated"].dropna() / POBLACION["Nacional"]
    _cobertura = np.interp(_fechas.values.astype("int64"), _v.index.values.astype("int64"), _v.values)
    _u = np.gradient(_cobertura) / (1 - _cobertura)           # fracción de no vacunados que se vacuna por día
    _gam, _sig = 1 / 2.5, 1 / 3.0
    _beta = r0_delta.value * _gam

    def _seirv(z, t, e):
        s, ex, i, acum = z                                     # acum: infecciones acumuladas
        u = np.interp(t, _dias, _u)
        return [-_beta * s * i - e * u * s, _beta * s * i - _sig * ex, _sig * ex - _gam * i, _beta * s * i]

    _sal = {}
    for _nombre, _e in [("con vacunación", e_delta.value), ("sin vacunación", 0.0)]:
        _z0 = [s0_delta.value * (1 - _e * _cobertura[0]) - 2e-4, 1e-4, 1e-4, 0.0]
        _z = odeint(_seirv, _z0, _dias, args=(_e,), rtol=1e-8, atol=1e-12, hmax=1.0)
        _sal[_nombre] = _z

    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for _nombre, _col in [("sin vacunación", GRIS), ("con vacunación", COLOR["I"])]:
        _ax[0].plot(_fechas, 100 * _beta * _sal[_nombre][:, 0] * _sal[_nombre][:, 2], color=_col, label=_nombre)
        _ax[1].plot(_fechas, r0_delta.value * _sal[_nombre][:, 0], color=_col if _nombre == "sin vacunación" else COLOR["V"], label=_nombre)
    _ax[0].set(title="Modelo: infecciones nuevas", ylabel="% de la población por día")
    _ax[0].legend(fontsize=8)
    _ax[1].axhline(1, color="#0b0b0b", lw=1)
    _ax[1].set(title=r"Modelo: $R_{\mathrm{ef}}(t) = R_0\, S/N$")
    _obs = serie_semanal("Nacional", "covid_confirmados", "2021-06-01", "2021-11-28")
    _ax[2].plot(_obs.index, _obs.values, "o-", color=COLOR["I"], ms=3, lw=1.2)
    _ax[2].set(title="Datos: casos confirmados por semana", ylabel="casos")
    for _a in _ax:
        _a.tick_params(axis="x", rotation=40)
    _fig.tight_layout()

    _inf = {k: 100 * z[-1, 3] for k, z in _sal.items()}
    mo.vstack([
        mo.hstack([r0_delta, s0_delta, e_delta], justify="start", wrap=True),
        _fig,
        mo.md(
            f"Cobertura (al menos una dosis): {100 * _cobertura[0]:.0f} % el 1 de junio y {100 * _cobertura[-1]:.0f} % a finales de noviembre de 2021. "
            f"Con estos supuestos se infecta aproximadamente el **{_inf['sin vacunación']:.0f} %** de la población sin vacunación y el "
            f"**{_inf['con vacunación']:.0f} %** con la vacunación real, entre el 1 de junio y finales de noviembre. "
            "Con $R_0$ alto y $e$ moderada, la vacunación **no evita la ola**, pero la hace más pequeña; "
            "y la protección contra enfermedad grave (que este modelo no incluye) es mayor que la protección contra infección. "
            "Compara la forma con los datos de la derecha, pero recuerda que están en otra escala (casos confirmados, no infecciones)."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. ¿Cuándo deja de valer el modelo? Poblaciones pequeñas

    En la sesión 1 vimos que la EDO es el límite $N \to \infty$ de un proceso aleatorio con personas enteras. ¿Qué tan grande tiene que ser $N$? ¿Qué pasa en una casa de 4 personas, en un salón de 10, en una escuela de 100?

    ### 6.1 El modelo estocástico exacto

    El estado es $(S, I)$ con números enteros ($R = N - S - I$). Hay dos eventos posibles:

    | evento | cambio | tasa |
    |---|---|---|
    | infección | $(S, I) \to (S - 1, I + 1)$ | $a_1 = \beta\,\dfrac{S I}{N - 1}$ |
    | recuperación | $(S, I) \to (S, I - 1)$ | $a_2 = \gamma I$ |

    Usamos $N - 1$ en lugar de $N$ porque nadie se contagia a sí mismo: así, un infeccioso rodeado de $N - 1$ susceptibles contagia a tasa $\beta$ y, en promedio, a $R_0 = \beta/\gamma$ personas, **sin importar el tamaño del grupo**. (Para $N$ grande da igual.)

    **Algoritmo de Gillespie (1977).** Como los eventos son procesos de Poisson independientes:

    1. Calcula las tasas $a_1, a_2$ y su suma $a_0 = a_1 + a_2$.
    2. El tiempo hasta el siguiente evento es exponencial con tasa $a_0$: $\Delta t = -\ln(u_1)/a_0$, con $u_1 \sim \text{Uniforme}(0,1)$.
    3. El evento es una infección con probabilidad $a_1/a_0$ y una recuperación con probabilidad $a_2/a_0$.
    4. Actualiza $(S, I)$ y $t$; repite hasta que $I = 0$.

    No es una aproximación: cada corrida es una realización exacta del proceso de Markov.

    ### 6.2 Dos predicciones exactas que la EDO no puede hacer

    * **Probabilidad de que el caso índice no contagie a nadie.** Al inicio, la infección compite con la recuperación: $P = \dfrac{\gamma}{\gamma + \beta} = \dfrac{1}{1 + R_0}$, para **cualquier** $N$.
    * **Probabilidad de que no haya brote grande** (en una población grande, con un caso inicial). Al inicio la epidemia es un *proceso de ramificación*, y la probabilidad de extinción temprana es $1/R_0$ si $R_0 > 1$. Con $R_0 = 2$, **la mitad de las introducciones se apagan solas**. La EDO, en cambio, predice un brote seguro.
    """)
    return


@app.cell
def _(np):
    def gillespie_sir(N, R0, rng, I0=1):
        """Una realización exacta del SIR estocástico (gamma = 1: tiempo en periodos infecciosos)."""
        S, I, t = N - I0, I0, 0.0
        tiempos, infecciosos = [0.0], [I]
        while I > 0:
            tasa_inf = R0 * S * I / (N - 1)
            tasa_rec = I
            total = tasa_inf + tasa_rec
            t += rng.expovariate(total)
            if rng.random() * total < tasa_inf:
                S -= 1
                I += 1
            else:
                I -= 1
            tiempos.append(t)
            infecciosos.append(I)
        return np.array(tiempos), np.array(infecciosos), N - S      # N - S = total que se infectó

    def distribucion_exacta(N, R0):
        """
        Distribución EXACTA del total infectado (sin simular), para N pequeño.

        Solo importa el orden de los eventos, no sus tiempos: desde (S, I) el siguiente
        evento es una infección con probabilidad a1/(a1 + a2). Recorremos todos los estados
        acumulando la probabilidad de llegar a cada uno.
        """
        prob = {(N - 1, 1): 1.0}
        final = np.zeros(N + 1)
        for total_eventos in range(2 * N):
            nueva = {}
            for (S, I), p in prob.items():
                if I == 0:
                    final[N - S] += p
                    continue
                a1, a2 = R0 * S * I / (N - 1), I
                if S > 0:
                    nueva[(S - 1, I + 1)] = nueva.get((S - 1, I + 1), 0) + p * a1 / (a1 + a2)
                nueva[(S, I - 1)] = nueva.get((S, I - 1), 0) + p * a2 / (a1 + a2)
            prob = nueva
        for (S, I), p in prob.items():
            final[N - S] += p
        return final            # final[k] = P(se infectan k personas en total)

    return distribucion_exacta, gillespie_sir


@app.cell(hide_code=True)
def _(mo):
    r0_gil = mo.ui.slider(0.5, 5.0, step=0.1, value=2.0, label="$R_0$", show_value=True)
    n_gil = mo.ui.dropdown(options={"4 (una casa)": 4, "10 (un salón pequeño)": 10, "100 (una escuela)": 100, "1000 (un pueblo)": 1000},
                           value="4 (una casa)", label="Tamaño de la población $N$")
    return n_gil, r0_gil


@app.cell(hide_code=True)
def _(gillespie_sir, np, odeint, r0_gil, random):
    # Simula para todos los tamaños a la vez (para la tabla comparativa).
    rejilla = np.linspace(0, 20, 401)
    simulaciones = {}
    for _N, _corridas in [(4, 4000), (10, 2000), (100, 400), (1000, 150)]:
        _rng = random.Random(2026 + _N)
        _finales, _media, _muestras = [], np.zeros_like(rejilla), []
        for _k in range(_corridas):
            _t, _I, _Z = gillespie_sir(_N, r0_gil.value, _rng)
            _finales.append(_Z)
            _Ig = _I[np.searchsorted(_t, rejilla, side="right") - 1]
            _media += _Ig / _corridas
            if _k < 25:
                _muestras.append((_t, _I))
        # EDO con las mismas tasas: dS/dt = -R0 S I/(N-1), dI/dt = R0 S I/(N-1) - I.
        _f = lambda y, t, N=_N: [-r0_gil.value * y[0] * y[1] / (N - 1), r0_gil.value * y[0] * y[1] / (N - 1) - y[1]]
        _ode = odeint(_f, [_N - 1, 1.0], np.concatenate([rejilla, [200.0]]), rtol=1e-9, atol=1e-12)
        simulaciones[_N] = dict(finales=np.array(_finales), media=_media, muestras=_muestras,
                                ode_I=_ode[:-1, 1], ode_total=_N - _ode[-1, 0], corridas=_corridas)
    return rejilla, simulaciones


@app.cell(hide_code=True)
def _(
    COLOR,
    GRIS,
    distribucion_exacta,
    mo,
    n_gil,
    np,
    plt,
    r0_gil,
    rejilla,
    simulaciones,
):
    _N = n_gil.value
    _d = simulaciones[_N]
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for _t, _I in _d["muestras"]:
        _ax[0].step(_t, _I, where="post", color=GRIS, lw=0.8, alpha=0.6)
    _ax[0].plot([], [], color=GRIS, lw=0.8, label="25 corridas")
    _ax[0].plot(rejilla, _d["media"], color="#0b0b0b", lw=2, label=f"promedio de {_d['corridas']} corridas")
    _ax[0].plot(rejilla, _d["ode_I"], color=COLOR["I"], lw=2, ls="--", label="EDO")
    _ax[0].set(title=f"Infecciosos, $N = {_N}$", xlabel=r"$\tau$ (periodos infecciosos)", ylabel="personas", xlim=(0, 20 if _N > 10 else 10))
    _ax[0].legend(fontsize=8)

    _Z = _d["finales"]
    if _N <= 10:
        _k = np.arange(1, _N + 1)
        _p = np.array([(_Z == k).mean() for k in _k])
        _ax[1].bar(_k, _p, width=0.7, color=COLOR["R"], label="simulación")
        _ax[1].plot(_k, distribucion_exacta(_N, r0_gil.value)[1:], "D", color="#0b0b0b", ms=6, label="exacta")
        _ax[1].set(xticks=_k)
    else:
        _ax[1].hist(_Z, bins=np.linspace(0, _N, 41), weights=np.ones(len(_Z)) / len(_Z), color=COLOR["R"], rwidth=0.85)
    _ax[1].axvline(_d["ode_total"], color=COLOR["I"], ls="--", lw=2, label=f"EDO: {_d['ode_total']:.2f} personas")
    _ax[1].axvline(_Z.mean(), color="#0b0b0b", lw=2, label=f"promedio simulado: {_Z.mean():.2f}")
    _ax[1].set(title="Total de personas que se infectaron", xlabel="personas (incluye al caso índice)", ylabel="probabilidad")
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()

    _filas = []
    for _n, _s in simulaciones.items():
        _z = _s["finales"]
        _cerca = np.mean(np.abs(_z - _s["ode_total"]) <= 0.1 * _n)
        _p1 = np.mean(_z == 1)
        _filas.append(
            f"| {_n} | {_s['corridas']} | {_s['ode_total']:.2f} | {_z.mean():.2f} | {_p1:.3f} ± {np.sqrt(_p1 * (1 - _p1) / len(_z)):.3f} | {_cerca:.2f} |"
        )
    _R = r0_gil.value
    mo.vstack([
        mo.hstack([n_gil, r0_gil], justify="start"),
        _fig,
        mo.md(
            "| $N$ | corridas | total infectado según la EDO | promedio simulado | $P$(solo el caso índice) ± error estándar | $P$(a menos de $0.1N$ de la EDO) |\n"
            "|---:|---:|---:|---:|---:|---:|\n" + "\n".join(_filas) +
            f"\n\nTeoría: $P$(solo el caso índice) $= 1/(1 + R_0) = {1 / (1 + _R):.3f}$ para todo $N$ "
            "(las simulaciones difieren de este valor por el error de Monte Carlo, del tamaño del error estándar); "
            f"probabilidad de que no haya brote grande cuando $N \\to \\infty$: "
            + (f"$1/R_0 = {1 / _R:.3f}$." if _R > 1 else "1 (con $R_0 \\le 1$ nunca hay brote grande).")
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Qué nos dicen las simulaciones

    1. **Con $N = 4$ la EDO predice "fracciones de persona"** (por ejemplo, 3.2 personas infectadas). La realidad solo admite 1, 2, 3 o 4, y la distribución está lejos de concentrarse en el valor de la EDO.
    2. **El azar domina al inicio, para cualquier $N$.** La probabilidad de que el primer caso no contagie a nadie es $1/(1 + R_0)$ aunque la población sea enorme. La EDO no puede representar eso: arranca con una "fracción" de infeccioso que nunca se extingue.
    3. **La distribución del tamaño final es bimodal para $N$ grande**: o el brote se apaga muy pronto (casi nadie se infecta), o es un brote grande cuyo tamaño sí se parece al de la EDO. **El promedio no se parece a ninguno de los dos**: la EDO no describe "lo que pasa en promedio" sino lo que pasa *si hay un brote grande*.
    4. **La curva promedio es más baja y más ancha que la de la EDO**: mezcla corridas que se apagaron con corridas que ocurrieron en distintos momentos.
    5. **La concordancia mejora con $N$** (última columna de la tabla), pero incluso con $N = 1000$ hay una fracción grande de corridas sin brote.

    **En una casa**, además, la hipótesis de mezcla homogénea es casi cierta (todos conviven con todos), pero la de población grande es falsa. Por eso los estudios de transmisión en hogares no usan EDO: usan el modelo estocástico (o el de cadenas binomiales de Reed y Frost) y estiman la **tasa de ataque secundario** (la fracción de convivientes que se infecta).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. SIS en poblaciones finitas: la extinción es segura

    En la EDO del SIS, con $R_0 > 1$, el equilibrio endémico $i^* = 1 - 1/R_0$ es globalmente estable: la enfermedad **persiste para siempre**. En el modelo estocástico, $I = 0$ es un **estado absorbente** (sin infecciosos nadie puede infectarse), y desde cualquier estado hay una probabilidad positiva de llegar a él. Por lo tanto, **con probabilidad 1 la enfermedad se extingue** en una población finita. La pregunta correcta es **cuánto tarda**.

    La teoría (Nåsell, 1996, 2011) dice que, para $R_0 > 1$, el tiempo medio hasta la extinción crece **exponencialmente** con $N$, aproximadamente como $e^{N c}$ con $c = \ln R_0 - 1 + 1/R_0$. Para poblaciones pequeñas es corto; para poblaciones grandes es astronómico, y entonces el equilibrio endémico de la EDO describe bien un estado *cuasi-estacionario* muy largo.

    Esta simulación tarda unos segundos: presiona el botón.
    """)
    return


@app.cell
def _(np):
    def tiempo_extincion_sis(N, R0, rng, t_max):
        """Tiempo hasta I = 0 en el SIS estocástico, empezando cerca del equilibrio endémico."""
        I = max(1, round(N * (1 - 1 / R0)))
        t = 0.0
        while I > 0 and t < t_max:
            tasa_inf = R0 * (N - I) * I / (N - 1)
            total = tasa_inf + I
            t += rng.expovariate(total)
            I += 1 if rng.random() * total < tasa_inf else -1
        return t if I == 0 else np.inf

    return (tiempo_extincion_sis,)


@app.cell(hide_code=True)
def _(mo):
    r0_sis2 = mo.ui.slider(1.2, 3.0, step=0.1, value=2.0, label="$R_0$", show_value=True)
    boton_sis = mo.ui.run_button(label="Simular extinciones")
    return boton_sis, r0_sis2


@app.cell(hide_code=True)
def _(COLOR, GRIS, boton_sis, mo, np, plt, r0_sis2, random, tiempo_extincion_sis):
    _salida = mo.md("*Presiona el botón para simular.*")
    if boton_sis.value:
        _R = r0_sis2.value
        _rng = random.Random(7)
        _Ns = np.arange(2, 31, 2)
        _t_max = 3000.0
        _medias, _cens = [], []
        for _N in _Ns:
            _ts = np.array([tiempo_extincion_sis(int(_N), _R, _rng, _t_max) for _ in range(40)])
            _cens.append(np.mean(np.isinf(_ts)))
            _medias.append(np.mean(np.where(np.isinf(_ts), _t_max, _ts)))
        _medias = np.array(_medias)
        _c = np.log(_R) - 1 + 1 / _R
        _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
        _ok = np.array(_cens) == 0
        _ax.semilogy(_Ns[_ok], _medias[_ok], "o", color=COLOR["I"], ms=7, label="simulación (40 corridas por $N$)")
        _ax.semilogy(_Ns[~_ok], _medias[~_ok], "^", color=GRIS, ms=7, label=f"al menos una corrida no se extinguió antes de {_t_max:.0f}")
        _j = np.argmax(_ok * _Ns)
        _ax.semilogy(_Ns, _medias[_j] * np.exp(_c * (_Ns - _Ns[_j])), color="#0b0b0b", lw=1.5, ls="--", label=f"pendiente teórica $e^{{cN}}$, $c$ = {_c:.3f}")
        _ax.set(title="Tiempo medio hasta la extinción (SIS)", xlabel="$N$", ylabel=r"$\tau$ (periodos infecciosos)")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _salida = mo.vstack([_fig, mo.md(
            "La EDO dice \"nunca\". El modelo estocástico dice \"siempre, pero cada vez más tarde\": el tiempo crece "
            "exponencialmente con $N$ (en escala logarítmica, una recta). En un hogar o un salón, una infección sin "
            "inmunidad se extingue rápido; para persistir necesita una población grande o reintroducciones desde fuera."
        )])
    mo.vstack([mo.hstack([r0_sis2, boton_sis], justify="start"), _salida])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. Hogares: poblaciones pequeñas conectadas entre sí

    Una ciudad no es una mezcla homogénea de millones de personas: es un conjunto de **hogares pequeños** (donde el contacto es intenso) conectados por contactos **en la comunidad** (menos intensos). Ball, Mollison y Scalia-Tomba (1997) estudiaron este modelo de **dos niveles de mezcla**. Su resultado central: el umbral ya no es $R_0$, sino
    $$
    R_* = R_G \times \mu_h ,
    $$
    donde $R_G$ es el número medio de contagios **fuera** de casa que produce un infectado, y $\mu_h$ es el **tamaño medio del brote dentro de un hogar** iniciado por una persona (incluida ella). Es decir: cada infectado de fuera "siembra" un brote en su casa, y cada miembro de ese brote siembra hogares nuevos. Hay epidemia en la ciudad si y solo si $R_* > 1$.

    $\mu_h$ es justamente lo que calcula el modelo estocástico de la sección 6 (el promedio del total infectado con $N$ = tamaño del hogar), **no** lo que diría la EDO.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r_casa = mo.ui.slider(0.0, 5.0, step=0.1, value=1.5, label="$R$ dentro de casa", show_value=True)
    r_comunidad = mo.ui.slider(0.1, 1.5, step=0.05, value=0.5, label="$R_G$ en la comunidad", show_value=True)
    return r_casa, r_comunidad


@app.cell(hide_code=True)
def _(COLOR, GRIS, mo, np, plt, r_casa, r_comunidad, random):
    def _tamano_brote_casa(h, R, rng):
        """Solo el tamaño final: basta la cadena de saltos, sin tiempos."""
        S, I = h - 1, 1
        while I > 0:
            a1 = R * S * I / (h - 1) if h > 1 else 0.0
            if rng.random() * (a1 + I) < a1:
                S, I = S - 1, I + 1
            else:
                I -= 1
        return h - S

    _rng = random.Random(11)
    _hs = np.arange(1, 9)
    _mu = np.array([np.mean([_tamano_brote_casa(int(h), r_casa.value, _rng) for _ in range(3000)]) for h in _hs])
    _Rstar = r_comunidad.value * _mu
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax[0].plot(_hs, _mu, "o-", color=COLOR["R"], label="simulación (3000 brotes por tamaño)")
    _ax[0].plot(_hs, _hs, color=GRIS, lw=1, ls=":", label="todos se infectan")
    _ax[0].set(title=r"Tamaño medio del brote en casa $\mu_h$", xlabel="personas en el hogar", ylabel="personas")
    _ax[0].legend(fontsize=8)
    _ax[1].plot(_hs, _Rstar, "o-", color=COLOR["I"])
    _ax[1].axhline(1, color="#0b0b0b", lw=1)
    _ax[1].set(title=r"$R_* = R_G\,\mu_h$", xlabel="personas en el hogar", ylabel="$R_*$")
    _fig.tight_layout()
    _h_crit = _hs[_Rstar > 1]
    mo.vstack([mo.hstack([r_casa, r_comunidad], justify="start"), _fig, mo.md(
        f"Con $R_G = {r_comunidad.value:.2f}$ (menos de un contagio por persona fuera de casa) "
        + (f"la epidemia **sí** se propaga en una ciudad de hogares de {_h_crit[0]} o más personas: los hogares **amplifican** la transmisión."
           if len(_h_crit) and r_comunidad.value < 1 else
           ("la epidemia se propaga sin importar el tamaño del hogar." if r_comunidad.value >= 1 else
            "la epidemia no se propaga con hogares de hasta 8 personas."))
        + " Por eso medidas que solo reducen el contacto en la comunidad pueden no bastar si los hogares son grandes, "
        "y por eso aislar al primer caso *dentro* de casa importa."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9. Resumen: cuándo sí y cuándo no

    | Supuesto del modelo | Cuándo se rompe | Qué se ve | Qué se hace |
    |---|---|---|---|
    | **Población grande** ($N \to \infty$) | hogares, salones, hospitales, granjas, el inicio y el final de cualquier brote | extinción por azar, brotes de tamaño muy variable, "fracciones de persona" | modelos estocásticos (Gillespie, Reed-Frost), procesos de ramificación |
    | **Mezcla homogénea** | casi siempre: hogares, escuelas, redes de contactos, superpropagadores | olas que se frenan antes, brotes en grupos | modelos de hogares, de redes, metapoblaciones, estructura por edad |
    | **Parámetros constantes** | cambios de comportamiento, medidas, vacaciones, clima | olas asimétricas, recurrencias estacionales | $\beta(t)$, forzamiento estacional |
    | **Tiempos exponenciales** (salir de $I$ a tasa constante) | las infecciones reales tienen duraciones más regulares | picos más agudos que en el modelo | varias etapas en serie (distribución Erlang) |
    | **Una sola cepa, inmunidad fija** | influenza, SARS-CoV-2 (variantes, pérdida de inmunidad) | reinfecciones, olas sucesivas | SIRS, modelos multicepa |
    | **Los datos son la incidencia** | siempre: subregistro, retrasos, cambios en las pruebas y en la vigilancia | cambios bruscos que no son epidemiológicos | modelo de observación explícito; no confundir $K$ con $N$ |

    **La idea central de las dos sesiones.** Los modelos de EDO describen **densidades** en una población **grande** y **bien mezclada**. Dentro de ese dominio son muy útiles: explican el umbral $R_0 = 1$, la inmunidad de rebaño, la cobertura crítica de vacunación y el tamaño final de una epidemia, y permiten estimar $R_{\text{ef}}$ a partir de datos. Fuera de él (una casa, un salón, el primer caso en una ciudad) hay que usar modelos que cuenten personas.

    ### Referencias

    * Ball, F., Mollison, D. y Scalia-Tomba, G. (1997). Epidemics with two levels of mixing. *The Annals of Applied Probability*, 7(1), 46–89.
    * Cowling, B. J., Fang, V. J., Riley, S., Peiris, J. S. M. y Leung, G. M. (2009). Estimation of the serial interval of influenza. *Epidemiology*, 20(3), 344–347.
    * Ganyani, T. et al. (2020). Estimating the generation interval for coronavirus disease (COVID-19) based on symptom onset data. *Eurosurveillance*, 25(17).
    * Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *The Journal of Physical Chemistry*, 81(25), 2340–2361.
    * Keeling, M. J. y Rohani, P. (2008). *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press.
    * Nåsell, I. (1996). The quasi-stationary distribution of the closed endemic SIS model. *Advances in Applied Probability*, 28(3), 895–932.
    * Nåsell, I. (2011). *Extinction and Quasi-stationarity in the Stochastic Logistic SIS Model*. Springer.
    * Wallinga, J. y Lipsitch, M. (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proceedings of the Royal Society B*, 274, 599–604.
    * Secretaría de Salud, Dirección General de Epidemiología. Datos abiertos. https://www.gob.mx/salud/documentos/datos-abiertos-152127
    * OMS, FluNet. https://www.who.int/tools/flunet
    * Mathieu, E. et al. (2021). A global database of COVID-19 vaccinations. *Nature Human Behaviour*, 5, 947–953 (Our World in Data).
    """)
    return


if __name__ == "__main__":
    app.run()
