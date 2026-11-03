import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Modelos epidémicos · Sesión 1: teoría",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelos epidémicos compartimentales
    ## Sesión 1 · Teoría: qué modelamos, equilibrios, estabilidad, bifurcaciones y $R_0$

    Biología Matemática · Este cuaderno es **interactivo**: mueve los deslizadores y las figuras se recalculan en tu navegador (Python corre localmente gracias a WebAssembly; la primera carga tarda unos segundos).

    **Contenido**

    1. ¿Qué estamos modelando? Los números del modelo **no son personas**.
    2. La incidencia: cómo se escriben los contagios (acción de masas vs. incidencia estándar).
    3. Adimensionalización: de tres parámetros a uno solo.
    4. Modelo SI.
    5. Modelo SIS: el primer equilibrio endémico y una bifurcación transcrítica.
    6. Modelo SIR: un umbral **sin** bifurcación.
    7. SIR con nacimientos y muertes: equilibrio endémico, oscilaciones amortiguadas y bifurcación transcrítica.
    8. SIRS: inmunidad que se pierde.
    9. SEIR: periodo de latencia.
    10. Cómo calcular $R_0$ (cuatro maneras, incluida la matriz de siguiente generación).
    11. Vacunación: cobertura crítica y la bifurcación en el parámetro de vacunación.
    12. Resumen.

    En la **sesión 2** usamos datos reales de México (COVID-19, influenza y vacunación), estimamos parámetros y vemos, con simulaciones estocásticas, cuándo dejan de valer estos modelos.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    return brentq, mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(plt):
    # Estilo de las figuras: colores fijos por compartimento (paleta validada para
    # daltonismo), líneas delgadas y ejes discretos.
    COLOR = {
        "S": "#2a78d6",  # azul
        "I": "#eb6834",  # naranja
        "R": "#1baf7a",  # aqua
        "E": "#eda100",  # amarillo
        "V": "#e87ba4",  # magenta
    }
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
    return COLOR, GRIS


@app.cell
def _(brentq, np, solve_ivp):
    # ------------------------------------------------------------------
    # Los modelos de este cuaderno, ya ADIMENSIONALIZADOS (sección 3):
    # variables = fracciones de la población; tiempo tau = (tiempo) / (periodo
    # infeccioso medio). Cada función regresa dy/dtau.
    # ------------------------------------------------------------------

    def integrar(f, y0, tau_max, *args, n=1500, **opciones):
        """Integra dy/dtau = f(tau, y, *args) en [0, tau_max] y regresa (tau, y)."""
        tau = np.linspace(0, tau_max, n)
        sol = solve_ivp(f, (0, tau_max), y0, t_eval=tau, args=args,
                        rtol=1e-9, atol=1e-12, method="LSODA", **opciones)
        return sol.t, sol.y

    def sis(tau, y, R0):
        (i,) = y
        return [(R0 - 1) * i - R0 * i**2]

    def sir(tau, y, R0):
        s, i = y
        return [-R0 * s * i, R0 * s * i - i]

    def sir_demografia(tau, y, R0, eps):
        s, i = y
        return [eps * (1 - s) - R0 * s * i, R0 * s * i - i]

    def sirs(tau, y, R0, alfa):
        s, i = y
        return [-R0 * s * i + alfa * (1 - s - i), R0 * s * i - i]

    def seir(tau, y, R0, sig):
        s, e, i = y
        return [-R0 * s * i, R0 * s * i - sig * e, sig * e - i]

    def sir_vacunacion(tau, y, R0, nu, tau_v):
        s, i, v = y
        tasa = nu if tau >= tau_v else 0.0          # la campaña empieza en tau_v
        return [-R0 * s * i - tasa * s, R0 * s * i - i, tasa * s]

    def tamano_final(R0, s0=1.0):
        """Fracción z que se infecta en total: raíz positiva de z = s0 (1 - exp(-R0 z))."""
        if R0 * s0 <= 1:
            return 0.0
        return brentq(lambda z: z - s0 * (1 - np.exp(-R0 * z)), 1e-12, 1.0)

    return (
        integrar,
        seir,
        sir,
        sir_demografia,
        sir_vacunacion,
        sirs,
        sis,
        tamano_final,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. ¿Qué estamos modelando? Los números del modelo no son personas

    Pensemos en una población de $N$ personas. En cada instante, cada persona está en **un** estado: susceptible, infecciosa o recuperada. El estado real del sistema es un vector de **números enteros**

    $$
    \big(S(t),\, I(t),\, R(t)\big) \in \{0, 1, \dots, N\}^3, \qquad S + I + R = N,
    $$

    y el sistema cambia por **eventos discretos y aleatorios**: una infección ($S \to S-1$, $I \to I+1$) o una recuperación ($I \to I-1$, $R \to R+1$). Esto es un proceso estocástico (una cadena de Markov en tiempo continuo). Nadie se infecta "un poquito": o te infectas o no.

    El modelo que vamos a estudiar es otra cosa:

    $$
    \frac{dS}{dt} = -\beta \frac{S I}{N}, \qquad
    \frac{dI}{dt} = \beta \frac{S I}{N} - \gamma I, \qquad
    \frac{dR}{dt} = \gamma I .
    $$

    Aquí $S, I, R$ son **números reales que cambian de forma continua**. Si $dI/dt = 0.3$ personas por día, en medio día "aparecen" $0.15$ personas. Eso no existe. Entonces, ¿qué es $I(t)$?

    ### La analogía con la biomasa

    En ecología esto se dice sin pena: en muchos modelos de poblaciones la variable no es "número de peces" sino **biomasa** (kilogramos de peces) o **densidad** (peces por km²). La biomasa es una cantidad continua: se puede dividir, puede valer $0.37$ kg, y fluye entre compartimentos (de presas a depredadores) como un líquido entre recipientes.

    En los modelos epidémicos pasa exactamente lo mismo, aunque casi nunca se dice:

    > $S(t)$, $I(t)$ y $R(t)$ son **cantidades continuas de "masa poblacional"** en cada estado: densidades, o fracciones de la población cuando dividimos entre $N$. **No son conteos de personas.** Funcionan como aproximación a los conteos solo cuando la población es grande.

    ### ¿Por qué funciona la aproximación? La ley de los grandes números

    Thomas Kurtz (1970, 1971) demostró el resultado que justifica todo esto. Sea $X_N(t)$ el proceso estocástico con $N$ personas. Si las tasas de los eventos dependen solo de las **fracciones** $X_N/N$ (como aquí: la tasa de infección es $\beta S I / N = N \cdot \beta\, s\, i$ con $s = S/N$, $i = I/N$), entonces para cualquier tiempo final $T$ fijo

    $$
    \sup_{0 \le t \le T} \left| \frac{X_N(t)}{N} - x(t) \right| \xrightarrow[N \to \infty]{} 0 ,
    $$

    donde $x(t)$ es la solución de la ecuación diferencial. Además, las fluctuaciones alrededor de $x(t)$ son de tamaño $1/\sqrt{N}$:

    | $N$ | tamaño típico de las fluctuaciones relativas $\sim 1/\sqrt{N}$ |
    |---:|:---|
    | 10 | 32 % |
    | 100 | 10 % |
    | 10 000 | 1 % |
    | 1 000 000 | 0.1 % |

    Así que la EDO describe **el límite de una población infinita**. Para una ciudad funciona bien; para una casa de 4 personas, no (eso lo estudiaremos con detalle en la sesión 2).

    ### Consecuencias prácticas (que conviene tener siempre presentes)

    1. **$I(t) = 0.3$ personas no significa nada como conteo**, pero sí como densidad: "0.3 infecciosos por cada $N$" es perfectamente razonable si $N$ representa una población grande.
    2. **En la EDO la enfermedad nunca se extingue en tiempo finito.** Si $I(0) > 0$, entonces $I(t) > 0$ para todo $t$ (decae exponencialmente pero nunca llega a cero). Un modelo de rabia en zorros llegó a predecir densidades de $10^{-18}$ zorros por km² que luego "rebrotaban": Denis Mollison (1991) lo bautizó como el problema del **"atto-zorro"**. En una población real, cuando $I$ baja de 1 la enfermedad simplemente desaparece.
    3. **La EDO ni siquiera es exactamente el promedio del modelo estocástico.** Si promediamos el proceso aleatorio, $\frac{d}{dt}\mathbb{E}[I] = \frac{\beta}{N}\,\mathbb{E}[S I] - \gamma\,\mathbb{E}[I]$, y $\mathbb{E}[SI] \neq \mathbb{E}[S]\,\mathbb{E}[I]$ (la diferencia es la covarianza). La EDO aparece al despreciar esa covarianza, que es pequeña solo si $N$ es grande.
    4. **Lo que se mide no es $S$, $I$ ni $R$.** Los datos son casos *reportados*, que dependen de cuánta gente se hace una prueba. Entre el modelo y los datos siempre hay un **modelo de observación** (sesión 2).

    La siguiente figura simula el modelo estocástico (personas enteras, eventos aleatorios) varias veces y lo compara con la EDO, con el mismo $R_0 = 2.5$ y empezando con 1 % de infecciosos. Cambia $N$.
    """)
    return


@app.cell
def _(np):
    def sir_estocastico(N, R0, I0, tau_max, corridas, rng, dt=0.02):
        """
        SIR estocástico con personas enteras (aproximación binomial en pasos dt).

        En cada paso, cada susceptible se infecta con probabilidad 1 - exp(-R0 * I/N * dt)
        y cada infeccioso se recupera con probabilidad 1 - exp(-dt) (gamma = 1: el tiempo
        está medido en periodos infecciosos). Regresa tau y la matriz I[corrida, paso].
        """
        pasos = int(round(tau_max / dt))
        S = np.full(corridas, N - I0)
        I = np.full(corridas, I0)
        p_recuperar = 1 - np.exp(-dt)
        historia = np.empty((corridas, pasos + 1), dtype=int)
        historia[:, 0] = I
        for k in range(pasos):
            p_infectar = 1 - np.exp(-R0 * I / N * dt)
            nuevos = rng.binomial(S, p_infectar)
            recuperados = rng.binomial(I, p_recuperar)
            S = S - nuevos
            I = I + nuevos - recuperados
            historia[:, k + 1] = I
        return np.linspace(0, pasos * dt, pasos + 1), historia

    return (sir_estocastico,)


@app.cell(hide_code=True)
def _(mo):
    tam_poblacion = mo.ui.dropdown(
        options={"10": 10, "100": 100, "1 000": 1000, "10 000": 10000, "100 000": 100000},
        value="100",
        label="Tamaño de la población $N$",
    )
    return (tam_poblacion,)


@app.cell(hide_code=True)
def _(COLOR, GRIS, integrar, mo, np, plt, sir, sir_estocastico, tam_poblacion):
    _N = tam_poblacion.value
    _R0, _corridas = 2.5, 30
    _I0 = max(1, round(0.01 * _N))
    _tau, _I = sir_estocastico(_N, _R0, _I0, 15, _corridas, np.random.default_rng(1))
    _t_ode, _y = integrar(sir, [1 - _I0 / _N, _I0 / _N], 15, _R0)

    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for _k in range(_corridas):
        _ax[0].plot(_tau, _I[_k], color=GRIS, lw=0.8, alpha=0.6)
        _ax[1].plot(_tau, _I[_k] / _N, color=GRIS, lw=0.8, alpha=0.6)
    _ax[0].plot(_t_ode, _N * _y[1], color=COLOR["I"], label="EDO")
    _ax[1].plot(_t_ode, _y[1], color=COLOR["I"], label="EDO")
    _ax[0].plot([], [], color=GRIS, lw=0.8, label=f"{_corridas} simulaciones")
    _ax[0].set(title="Número de personas infecciosas", xlabel=r"tiempo $\tau$ (periodos infecciosos)", ylabel="personas")
    _ax[1].set(title="Fracción infecciosa $I/N$", xlabel=r"tiempo $\tau$ (periodos infecciosos)", ylabel="fracción")
    _ax[0].legend()
    _fig.tight_layout()

    _extintas = np.mean(_I[:, -1] == 0) * 100
    _nunca = np.mean(_I.max(axis=1) <= _I0) * 100
    mo.vstack([
        tam_poblacion,
        _fig,
        mo.md(
            f"Con $N = {_N:,}$ la EDO predice un pico de **{_N * _y[1].max():,.2f} personas** "
            f"infecciosas a la vez. En las simulaciones, en el {_nunca:.0f} % de las corridas la "
            f"infección nunca superó su valor inicial ({_I0} {'persona' if _I0 == 1 else 'personas'}). "
            "Con $N$ pequeño las corridas no se parecen a la EDO ni entre ellas; al crecer $N$ "
            "se concentran alrededor de la curva naranja."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. La incidencia: cómo se escriben los contagios

    El término que hace interesante a un modelo epidémico es la **incidencia**: cuántas infecciones nuevas ocurren por unidad de tiempo. Hay que construirlo, no adivinarlo.

    **Hipótesis de mezcla homogénea.** Cada persona tiene la misma probabilidad de encontrarse con cualquier otra (como moléculas en un gas bien agitado; por eso a veces se le llama "ley de acción de masas", tomada de la cinética química).

    **Construcción paso a paso.** Supongamos que:

    * cada persona tiene, en promedio, $c$ contactos por unidad de tiempo;
    * un contacto entre un infeccioso y un susceptible transmite la infección con probabilidad $p$;
    * como la mezcla es homogénea, la fracción de los contactos de un infeccioso que caen en un susceptible es $S/N$.

    Entonces un infeccioso produce $c \cdot p \cdot \frac{S}{N}$ infecciones por unidad de tiempo, y los $I$ infecciosos producen

    $$
    \text{incidencia} = c\,p\,\frac{S}{N}\,I = \beta\,\frac{S\,I}{N}, \qquad \beta = c\,p .
    $$

    Esta es la **incidencia estándar** (o *dependiente de la frecuencia*). El parámetro $\beta$ tiene unidades de $1/\text{tiempo}$: es la tasa de contactos *efectivos* (los que transmiten) de una persona.

    **¿Y si el número de contactos depende de la densidad?** Si la gente está más apretada, quizá se encuentra más. Si $c$ es proporcional a la densidad, $c = \kappa N / A$ ($A$ es el área), entonces

    $$
    \text{incidencia} = \kappa p \frac{N}{A}\frac{S}{N} I = \beta' S\,I .
    $$

    Esta es la **acción de masas** propiamente dicha (o *incidencia dependiente de la densidad*); ahora $\beta'$ tiene unidades de $1/(\text{personas}\cdot\text{tiempo})$.

    **¿Importa la diferencia? Sí, y mucho.** Con recuperación a tasa $\gamma$, un infeccioso dura en promedio $1/\gamma$ y, en una población totalmente susceptible, produce

    $$
    R_0 = \frac{\beta}{\gamma} \quad\text{(incidencia estándar)}, \qquad
    R_0 = \frac{\beta' N}{\gamma} \quad\text{(acción de masas)}.
    $$

    Con acción de masas, $R_0$ **crece con el tamaño de la población**: existe un **tamaño crítico** $N_c = \gamma/\beta'$ por debajo del cual la enfermedad no puede invadir. Con incidencia estándar no existe tal umbral: una enfermedad con $R_0 > 1$ invade igual una ciudad que un pueblo.

    ¿Cuál es la correcta? Depende del sistema. Para enfermedades de transmisión sexual o respiratorias en humanos, el número de contactos casi no depende de la densidad de población (uno no saluda al doble de gente porque la ciudad sea el doble de densa), y la incidencia estándar suele describir mejor los datos. Para muchas enfermedades de fauna silvestre ocurre lo contrario. McCallum, Barlow y Hone (2001) revisan cómo distinguirlas con datos. **Esta elección vuelve a aparecer con fuerza cuando la población es pequeña (una casa)**: ¿una persona en una casa de 10 contagia más que en una casa de 4? Lo retomamos en la sesión 2.

    En lo que sigue usamos la **incidencia estándar**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Adimensionalización: de tres parámetros a uno solo

    El SIR tiene tres parámetros: $\beta$, $\gamma$ y $N$. Adimensionalizar es **elegir escalas naturales** para las variables y el tiempo, de modo que el modelo quede escrito con el menor número posible de combinaciones de parámetros sin unidades.

    **Receta.**

    1. Escribe las unidades de cada cantidad: $S, I, R$ en personas; $t$ en días; $\beta$ y $\gamma$ en 1/día; $N$ en personas.
    2. Elige una escala para cada variable. Lo natural es $N$ para las poblaciones y $1/\gamma$ (el periodo infeccioso medio) para el tiempo:
    $$
    s = \frac{S}{N}, \qquad i = \frac{I}{N}, \qquad r = \frac{R}{N}, \qquad \tau = \gamma\, t .
    $$
    3. Sustituye usando la regla de la cadena. Como $S = N s$ y $t = \tau/\gamma$,
    $$
    \frac{dS}{dt} = \frac{d(Ns)}{d\tau}\,\frac{d\tau}{dt} = N\gamma\,\frac{ds}{d\tau} .
    $$
    Entonces $\frac{dS}{dt} = -\beta \frac{SI}{N}$ se vuelve $N\gamma\,\frac{ds}{d\tau} = -\beta\,\frac{(Ns)(Ni)}{N} = -\beta N\, s\, i$, es decir,
    $$
    \frac{ds}{d\tau} = -\frac{\beta}{\gamma}\, s\, i .
    $$
    4. Haz lo mismo con las demás ecuaciones:
    $$
    \boxed{\;\frac{ds}{d\tau} = -R_0\, s\, i, \qquad \frac{di}{d\tau} = R_0\, s\, i - i, \qquad \frac{dr}{d\tau} = i, \qquad R_0 = \frac{\beta}{\gamma}\;}
    $$

    **Qué ganamos.**

    * **Un solo parámetro.** Toda la dinámica depende de $R_0$. Dos enfermedades con el mismo $R_0$ tienen epidemias de **la misma forma**; solo cambia la escala de tiempo ($1/\gamma$) y la escala de población ($N$). La figura de abajo lo muestra.
    * **El parámetro de bifurcación aparece solo.** En las secciones siguientes veremos que todo cambia cuando $R_0$ cruza 1.
    * **El número de parámetros es una propiedad del modelo, no de cómo lo escribimos.** El teorema $\Pi$ de Buckingham lo explica: 3 parámetros con 2 unidades independientes (personas y tiempo) dejan $3 - 2 = 1$ grupo adimensional.

    **Y lo que perdimos (¡importante!).** $N$ **desapareció** del modelo. El modelo adimensional es idéntico para una casa de 4 personas y para un país de 126 millones. Eso es coherente con la sección 1: la EDO es el límite $N \to \infty$ y, por construcción, no sabe nada del tamaño de la población. Todo lo que dependa de $N$ (la extinción por azar, la variabilidad entre brotes) es invisible para este modelo.

    En la sesión 2 la adimensionalización tendrá otro uso: nos dirá **qué combinaciones de parámetros pueden estimarse con datos y cuáles no**.

    Las dos enfermedades de la figura tienen el mismo $R_0$ pero periodos infecciosos y poblaciones distintas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_adim = mo.ui.slider(1.2, 5.0, step=0.1, value=2.0, label="$R_0$ (ambas)", show_value=True)
    dur_a = mo.ui.slider(2, 15, step=1, value=3, label="A: $1/\\gamma$ (días)", show_value=True)
    dur_b = mo.ui.slider(2, 15, step=1, value=10, label="B: $1/\\gamma$ (días)", show_value=True)
    return dur_a, dur_b, r0_adim


@app.cell(hide_code=True)
def _(COLOR, dur_a, dur_b, mo, np, plt, r0_adim, solve_ivp):
    def _sir_dimensional(t, y, beta, gamma, N):
        S, I = y
        return [-beta * S * I / N, beta * S * I / N - gamma * I]

    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for _nombre, _dur, _N, _col in [("A", dur_a.value, 1_000, COLOR["S"]), ("B", dur_b.value, 1_000_000, COLOR["I"])]:
        _gamma = 1 / _dur
        _beta = r0_adim.value * _gamma
        _t = np.linspace(0, 400, 4000)
        _sol = solve_ivp(_sir_dimensional, (0, 400), [_N * 0.999, _N * 0.001], t_eval=_t, args=(_beta, _gamma, _N), rtol=1e-9)
        _lab = f"{_nombre}: $1/\\gamma$ = {_dur} d, $N$ = {_N:,}"
        _ax[0].plot(_sol.t, _sol.y[1] / _N, color=_col, label=_lab)
        _ax[1].plot(_gamma * _sol.t, _sol.y[1] / _N, color=_col, label=_lab, lw=3 if _nombre == "A" else 1.5, ls="-" if _nombre == "A" else "--")
    _ax[0].set(title="Variables con dimensiones", xlabel="días", ylabel="$I/N$", xlim=(0, 30 * max(dur_a.value, dur_b.value)))
    _ax[1].set(title="Variables adimensionales", xlabel=r"$\tau = \gamma t$ (periodos infecciosos)", ylabel="$i$", xlim=(0, 30))
    _ax[0].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.hstack([r0_adim, dur_a, dur_b], justify="start", wrap=True), _fig,
               mo.md("En días las epidemias se ven distintas; en $\\tau$ las curvas son **la misma** (se enciman), aunque una ocurre en una población mil veces mayor.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Modelo SI: todos terminan infectados

    El modelo más simple: no hay recuperación ($\gamma = 0$). Con $S + I = N$ y $i = I/N$,

    $$
    \frac{di}{dt} = \beta\, i\,(1 - i).
    $$

    Es la **ecuación logística**. Su solución es $i(t) = \dfrac{i_0}{i_0 + (1-i_0)e^{-\beta t}}$.

    * Equilibrios: $i^* = 0$ e $i^* = 1$.
    * Estabilidad (derivada del lado derecho $f(i) = \beta i(1-i)$): $f'(i) = \beta(1 - 2i)$, así que $f'(0) = \beta > 0$ (inestable) y $f'(1) = -\beta < 0$ (estable).
    * No hay ningún parámetro que cambie el retrato: para todo $\beta > 0$ la infección llega a toda la población. **No hay bifurcación** (no hay umbral). Aquí no podemos adimensionalizar el tiempo con $1/\gamma$ porque $\gamma = 0$; usamos $\tau = \beta t$ y no queda ningún parámetro.

    Describe infecciones de las que nunca se sale (por ejemplo, algunas infecciones virales persistentes) durante una escala de tiempo corta comparada con la vida de los hospederos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Modelo SIS: el primer equilibrio endémico

    Ahora los infecciosos se recuperan a tasa $\gamma$ pero **no quedan inmunes**: vuelven a ser susceptibles (gonorrea, muchas infecciones bacterianas, el resfriado común).

    $$
    \frac{dS}{dt} = -\beta\frac{SI}{N} + \gamma I, \qquad \frac{dI}{dt} = \beta\frac{SI}{N} - \gamma I .
    $$

    **Paso 1: reducir la dimensión.** Sumando, $\frac{d}{dt}(S + I) = 0$: la población total es constante. Con $S = N - I$ basta una ecuación. Dividiendo entre $N$:
    $$
    \frac{di}{dt} = \beta(1 - i)\,i - \gamma i .
    $$

    **Paso 2: adimensionalizar** con $\tau = \gamma t$ y $R_0 = \beta/\gamma$:
    $$
    \frac{di}{d\tau} = R_0 (1-i)\, i - i = (R_0 - 1)\, i - R_0\, i^2 \;\equiv\; f(i).
    $$

    **Paso 3: equilibrios.** $f(i) = i\,[(R_0 - 1) - R_0 i] = 0$ da
    $$
    i^*_0 = 0 \quad\text{(libre de enfermedad)}, \qquad i^*_1 = 1 - \frac{1}{R_0} \quad\text{(endémico)}.
    $$
    El equilibrio endémico tiene sentido biológico ($i^*_1 > 0$) **solo si $R_0 > 1$**.

    **Paso 4: estabilidad.** En una dimensión, un equilibrio es estable si $f'(i^*) < 0$ e inestable si $f'(i^*) > 0$. Con $f'(i) = (R_0 - 1) - 2R_0\, i$:
    $$
    f'(0) = R_0 - 1, \qquad f'\!\left(1 - \tfrac{1}{R_0}\right) = (R_0 - 1) - 2(R_0 - 1) = -(R_0 - 1).
    $$

    | | $R_0 < 1$ | $R_0 > 1$ |
    |---|---|---|
    | $i^* = 0$ | **estable** | inestable |
    | $i^* = 1 - 1/R_0$ | no existe (negativo), inestable | **estable** |

    **Paso 5: la bifurcación.** En $R_0 = 1$ las dos ramas de equilibrios ($i^* = 0$ e $i^* = 1 - 1/R_0$) **se cruzan e intercambian su estabilidad**. Eso es una **bifurcación transcrítica**. De hecho, cerca de $R_0 = 1$, con $\mu = R_0 - 1$ la ecuación es $\frac{di}{d\tau} = \mu\, i - (1 + \mu)\, i^2 \approx \mu\, i - i^2$: la **forma normal** de la bifurcación transcrítica (Strogatz, cap. 3).

    **Solución explícita.** Como es otra logística, con $i_\infty = 1 - 1/R_0$:
    $$
    i(\tau) = \frac{i_\infty}{1 + \left(\frac{i_\infty}{i_0} - 1\right) e^{-(R_0 - 1)\tau}} \qquad (R_0 \ne 1).
    $$
    Observa que la tasa de crecimiento inicial es $R_0 - 1$ (en unidades de $\tau$), es decir, $r = \gamma(R_0 - 1) = \beta - \gamma$ en días. Esta relación entre **tasa de crecimiento** y $R_0$ será clave para estimar $R_0$ con datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_sis = mo.ui.slider(0.2, 5.0, step=0.05, value=2.0, label="$R_0$", show_value=True)
    return (r0_sis,)


@app.cell(hide_code=True)
def _(COLOR, GRIS, integrar, mo, np, plt, r0_sis, sis):
    _R0 = r0_sis.value
    _fig, _ax = plt.subplots(1, 3, figsize=(11, 3.5))

    # (a) Línea de fase: f(i) contra i.
    _i = np.linspace(0, 1, 400)
    _f = (_R0 - 1) * _i - _R0 * _i**2
    _ax[0].axhline(0, color=GRIS, lw=1)
    _ax[0].plot(_i, _f, color=COLOR["I"])
    _equil = [0.0] + ([1 - 1 / _R0] if _R0 > 1 else [])
    for _e in _equil:
        _estable = ((_R0 - 1) - 2 * _R0 * _e) < 0
        _ax[0].plot(_e, 0, "o", ms=9, mfc=COLOR["I"] if _estable else "white", mec=COLOR["I"], mew=2, zorder=3)
    _ax[0].set(title=r"Línea de fase: $f(i) = di/d\tau$", xlabel="$i$", ylabel="$f(i)$")

    # (b) Trayectorias desde varias condiciones iniciales.
    for _i0 in [0.01, 0.2, 0.5, 0.9]:
        _t, _y = integrar(sis, [_i0], 20, _R0)
        _ax[1].plot(_t, _y[0], color=COLOR["I"], lw=1.5)
    if _R0 > 1:
        _ax[1].axhline(1 - 1 / _R0, color=GRIS, ls=":", lw=1.5)
    _ax[1].set(title="Trayectorias", xlabel=r"$\tau$", ylabel="$i(\\tau)$", ylim=(-0.02, 1))

    # (c) Diagrama de bifurcación.
    _r = np.linspace(0.2, 5, 400)
    _ax[2].plot(_r[_r <= 1], 0 * _r[_r <= 1], color=COLOR["I"], label="estable")
    _ax[2].plot(_r[_r >= 1], 0 * _r[_r >= 1], color=COLOR["I"], ls="--", lw=1.5, label="inestable")
    _ax[2].plot(_r[_r >= 1], 1 - 1 / _r[_r >= 1], color=COLOR["I"])
    _ax[2].plot(_r[_r <= 1], 1 - 1 / _r[_r <= 1], color=GRIS, ls="--", lw=1.2, label="inestable y sin sentido ($i^*<0$)")
    _ax[2].axvline(_R0, color=GRIS, lw=1)
    _ax[2].set(title="Diagrama de bifurcación", xlabel="$R_0$", ylabel="$i^*$", ylim=(-1, 1))
    _ax[2].legend(fontsize=8, loc="lower right")
    _fig.tight_layout()

    mo.vstack([r0_sis, _fig, mo.md(
        "Círculos llenos: equilibrios estables; vacíos: inestables. "
        + (f"Con $R_0 = {_R0:.2f}$ todas las trayectorias van al equilibrio endémico $i^* = {1 - 1/_R0:.3f}$."
           if _R0 > 1 else f"Con $R_0 = {_R0:.2f} < 1$ la infección desaparece desde cualquier condición inicial.")
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Modelo SIR: un umbral **sin** bifurcación

    Ahora los recuperados quedan inmunes para siempre (sarampión, varicela; y como aproximación, una ola de una enfermedad cuya inmunidad dura más que la ola). Adimensionalizado (sección 3), y usando $r = 1 - s - i$ para quedarnos con dos ecuaciones:

    $$
    \frac{ds}{d\tau} = -R_0\, s\, i, \qquad \frac{di}{d\tau} = R_0\, s\, i - i = i\,(R_0 s - 1).
    $$

    ### 6.1 Equilibrios: una recta entera

    $\frac{di}{d\tau} = 0$ si $i = 0$ o $s = 1/R_0$; $\frac{ds}{d\tau} = 0$ si $s = 0$ o $i = 0$. Las dos se cumplen a la vez **si y solo si $i = 0$**, con $s$ arbitraria. Los equilibrios son

    $$
    \{(s^*, 0) : 0 \le s^* \le 1\}:
    $$

    **un continuo de equilibrios** (una recta), no puntos aislados. Tiene sentido: en cuanto no queda nadie infeccioso, nada cambia, sin importar cuántos susceptibles queden. **No existe equilibrio endémico**: toda epidemia SIR termina con $i = 0$.

    ### 6.2 Linealización

    El jacobiano es
    $$
    J(s, i) = \begin{pmatrix} -R_0\, i & -R_0\, s \\ R_0\, i & R_0\, s - 1 \end{pmatrix},
    \qquad
    J(s^*, 0) = \begin{pmatrix} 0 & -R_0\, s^* \\ 0 & R_0\, s^* - 1 \end{pmatrix}.
    $$
    Es triangular, así que sus valores propios son los elementos de la diagonal:
    $$
    \lambda_1 = 0, \qquad \lambda_2 = R_0\, s^* - 1 .
    $$

    * $\lambda_1 = 0$ **siempre**. Su vector propio es $(1, 0)$: la dirección *a lo largo* de la recta de equilibrios. Moverse sobre la recta no cambia nada, por eso no hay ni atracción ni repulsión en esa dirección. Un equilibrio con un valor propio cero se llama **no hiperbólico**, y el teorema de linealización (Hartman–Grobman) **no se puede usar**.
    * $\lambda_2 = R_0 s^* - 1$ controla la dirección transversal (si una pequeña cantidad de infecciosos crece o decrece). Al número $R_0 s$ se le llama **número reproductivo efectivo**, $R_{\text{ef}} = R_0\, s$: cuántos contagios produce un infeccioso cuando solo una fracción $s$ es susceptible. Los equilibrios con $s^* < 1/R_0$ atraen en la dirección transversal; los que tienen $s^* > 1/R_0$ repelen.

    ### 6.3 La herramienta que sí funciona: una cantidad conservada

    Dividiendo las dos ecuaciones (donde $i \ne 0$):
    $$
    \frac{di}{ds} = \frac{i (R_0 s - 1)}{-R_0\, s\, i} = -1 + \frac{1}{R_0\, s}.
    $$
    Integrando respecto a $s$: $i = -s + \frac{1}{R_0}\ln s + C$. Es decir, la función
    $$
    \boxed{\;V(s, i) = s + i - \frac{1}{R_0}\ln s\;}
    $$
    es **constante a lo largo de cada trayectoria** (una *integral primera*). Las órbitas son las curvas de nivel de $V$. Con eso sabemos todo:

    * **El pico.** $i$ crece mientras $R_0 s > 1$ y decrece después. El máximo ocurre cuando $s = 1/R_0$: el famoso **umbral de inmunidad de rebaño**. Igualando $V(s_0, i_0) = V(1/R_0, i_{\max})$:
    $$
    i_{\max} = s_0 + i_0 - \frac{1}{R_0} - \frac{1}{R_0}\ln(R_0\, s_0) \;\xrightarrow{s_0 \to 1,\, i_0 \to 0}\; 1 - \frac{1 + \ln R_0}{R_0}.
    $$
    * **El tamaño final.** Cuando $\tau \to \infty$, $i \to 0$ y $s \to s_\infty$. Igualando $V(s_0, i_0) = V(s_\infty, 0)$ con $s_0 \to 1$, $i_0 \to 0$, se obtiene $\ln s_\infty = R_0 (s_\infty - 1)$. Con $z = 1 - s_\infty$ (la fracción que se infectó en total):
    $$
    \boxed{\;z = 1 - e^{-R_0 z}\;}
    $$
    la **ecuación del tamaño final** (Kermack y McKendrick, 1927). $z = 0$ siempre es solución; hay una solución positiva **si y solo si $R_0 > 1$** (la curva $1 - e^{-R_0 z}$ tiene pendiente $R_0$ en el origen, así que corta a la recta $z$ en un punto positivo solo si $R_0 > 1$).
    * **La epidemia termina con susceptibles.** $s_\infty > 0$ siempre: la epidemia no se apaga porque se acaben los susceptibles, sino porque se acaban los infecciosos (cada uno contagia en promedio menos de uno cuando $s < 1/R_0$). Además la epidemia **se pasa** del umbral: $s_\infty < 1/R_0$ ("overshoot").

    ### 6.4 Entonces, ¿hay bifurcación en el SIR?

    **No, no en el sentido usual.** Una bifurcación es un cambio cualitativo en el conjunto de equilibrios o en su estabilidad: que aparezcan o desaparezcan equilibrios, o que dos ramas intercambien su estabilidad (como en el SIS). En el SIR:

    * el conjunto de equilibrios es **el mismo para todo $R_0$**: la recta $i = 0$; no aparece ningún equilibrio nuevo al cruzar $R_0 = 1$;
    * no hay equilibrio endémico, ni para $R_0 < 1$ ni para $R_0 > 1$;
    * los equilibrios no son hiperbólicos ($\lambda_1 = 0$ siempre), así que la teoría estándar de bifurcaciones (que se apoya en equilibrios hiperbólicos aislados) no aplica.

    Lo que cambia en $R_0 = 1$ es un **umbral en el transitorio**: si una pequeña introducción de infecciosos crece antes de desaparecer ($R_0 s_0 > 1$) o desaparece directamente ($R_0 s_0 < 1$). Lo que sí tiene forma de bifurcación transcrítica es la **ecuación algebraica del tamaño final** $z = 1 - e^{-R_0 z}$ (una ecuación para el *resultado* de la epidemia, no un sistema dinámico): sus ramas $z = 0$ y $z > 0$ se cruzan en $R_0 = 1$. Algunos autores llaman a esto una "bifurcación degenerada"; lo importante es entender por qué el SIR es distinto del SIS: **el SIR no tiene un estado endémico al cual bifurcar**. Para tenerlo hace falta reponer susceptibles (nacimientos, sección 7, o pérdida de inmunidad, sección 8).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_sir = mo.ui.slider(0.5, 5.0, step=0.05, value=2.5, label="$R_0$", show_value=True)
    return (r0_sir,)


@app.cell(hide_code=True)
def _(COLOR, GRIS, integrar, mo, np, plt, r0_sir, sir, tamano_final):
    _R0 = r0_sir.value
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) Plano fase.
    _ax[0].plot([0, 1], [1, 0], color=GRIS, lw=1)                     # borde s + i = 1
    _iniciales = [(0.999, 0.001), (0.8, 0.001), (0.6, 0.001), (0.95, 0.05), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    for _s0, _i0 in _iniciales:
        _t, _y = integrar(sir, [_s0, _i0], 60, _R0)
        _ax[0].plot(_y[0], _y[1], color=COLOR["I"], lw=1.4)
        _ax[0].plot(_s0, _i0, "o", ms=4, color=COLOR["I"])
    _umbral = min(1 / _R0, 1)
    _ax[0].plot([0, _umbral], [0, 0], color="#0b0b0b", lw=4, solid_capstyle="butt", label="equilibrios que atraen ($R_0 s^* < 1$)")
    if _umbral < 1:
        _ax[0].plot([_umbral, 1], [0, 0], color="#0b0b0b", lw=2, ls=(0, (2, 2)), label="equilibrios que repelen ($R_0 s^* > 1$)")
        _ax[0].axvline(_umbral, color=GRIS, ls=":", lw=1.5)
        _ax[0].text(_umbral + 0.01, 0.92, "$s = 1/R_0$\n(pico de $i$)", fontsize=8, va="top")
    _ax[0].set(title="Plano fase $(s, i)$", xlabel="$s$ (susceptibles)", ylabel="$i$ (infecciosos)", xlim=(0, 1.02), ylim=(-0.03, 1.02))
    _ax[0].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.75))

    # (b) Series de tiempo.
    _t, _y = integrar(sir, [0.999, 0.001], 40, _R0)
    _ax[1].plot(_t, _y[0], color=COLOR["S"], label="$s$")
    _ax[1].plot(_t, _y[1], color=COLOR["I"], label="$i$")
    _ax[1].plot(_t, 1 - _y[0] - _y[1], color=COLOR["R"], label="$r$")
    if _R0 > 1:
        _ax[1].axhline(1 / _R0, color=GRIS, ls=":", lw=1.5)
        _ax[1].text(39, 1 / _R0 + 0.02, "$s = 1/R_0$", ha="right", fontsize=8)
    _ax[1].set(title="Series de tiempo ($s_0 = 0.999$)", xlabel=r"$\tau$", ylabel="fracción")
    _ax[1].legend(loc="center right")
    _fig.tight_layout()

    _z = tamano_final(_R0)
    _imax = 1 - (1 + np.log(_R0)) / _R0 if _R0 > 1 else 0
    mo.vstack([r0_sir, _fig, mo.md(
        f"Con $R_0 = {_R0:.2f}$: pico teórico $i_{{\\max}} = {_imax:.3f}$ (numérico: {_y[1].max():.3f}); "
        f"tamaño final $z = {_z:.3f}$ (numérico: {1 - _y[0][-1]:.3f}); "
        f"quedan sin infectarse $s_\\infty = {1 - _z:.3f}$"
        + (f", por debajo del umbral $1/R_0 = {1/_R0:.3f}$." if _R0 > 1 else ".")
    )])
    return


@app.cell(hide_code=True)
def _(COLOR, GRIS, mo, np, plt, r0_sir, tamano_final):
    _R0 = r0_sir.value
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _z = np.linspace(0, 1, 300)
    _ax[0].plot(_z, _z, color=GRIS, label="$z$")
    _ax[0].plot(_z, 1 - np.exp(-_R0 * _z), color=COLOR["R"], label=f"$1 - e^{{-R_0 z}}$, $R_0 = {_R0:.2f}$")
    _zf = tamano_final(_R0)
    _ax[0].plot([0, _zf], [0, _zf], "o", color=COLOR["R"], ms=7)
    _ax[0].set(title="Solución gráfica del tamaño final", xlabel="$z$", ylabel="")
    _ax[0].legend(fontsize=8)

    _r = np.linspace(0.5, 5, 300)
    _zr = np.array([tamano_final(x) for x in _r])
    _imax = np.where(_r > 1, 1 - (1 + np.log(_r)) / _r, 0)
    _ax[1].plot(_r, _zr, color=COLOR["R"], label="tamaño final $z$")
    _ax[1].plot(_r, _imax, color=COLOR["I"], label=r"pico $i_{\max}$")
    _ax[1].axvline(_R0, color=GRIS, lw=1)
    _ax[1].set(title="El umbral $R_0 = 1$", xlabel="$R_0$", ylabel="fracción de la población")
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "A la izquierda: las soluciones de $z = 1 - e^{-R_0 z}$ son los cruces. A la derecha: el tamaño final "
        "como función de $R_0$ tiene la forma de una bifurcación transcrítica, pero es la gráfica de una ecuación "
        "algebraica para el **resultado** de la epidemia, no un cambio en los equilibrios del sistema dinámico."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. SIR con nacimientos y muertes: aparece el equilibrio endémico

    Para que una enfermedad inmunizante persista hace falta **reponer susceptibles**. Añadimos nacimientos (todos susceptibles) a tasa $\mu N$ y muertes naturales a tasa $\mu$ en cada compartimento (así $N$ es constante):

    $$
    \frac{dS}{dt} = \mu N - \beta\frac{SI}{N} - \mu S, \qquad
    \frac{dI}{dt} = \beta\frac{SI}{N} - (\gamma + \mu) I, \qquad
    \frac{dR}{dt} = \gamma I - \mu R .
    $$

    **Adimensionalización.** Ahora un infeccioso deja de serlo a tasa $\gamma + \mu$ (se recupera o muere), así que la escala de tiempo natural es $1/(\gamma + \mu)$. Con $s = S/N$, $i = I/N$, $\tau = (\gamma + \mu)t$:

    $$
    \frac{ds}{d\tau} = \varepsilon(1 - s) - R_0\, s\, i, \qquad \frac{di}{d\tau} = R_0\, s\, i - i,
    \qquad R_0 = \frac{\beta}{\gamma + \mu}, \qquad \varepsilon = \frac{\mu}{\gamma + \mu}.
    $$

    Quedan **dos** parámetros adimensionales. $\varepsilon$ es el cociente entre la duración de la infección y la duración de la vida: para una infección de una semana en humanos que viven 70 años, $\varepsilon \approx 1/3600$. Es un parámetro **muy pequeño**, y eso tendrá consecuencias.

    ### 7.1 Equilibrios

    De $\frac{di}{d\tau} = i(R_0 s - 1) = 0$: $i = 0$ o $s = 1/R_0$.

    * Si $i = 0$, de $\frac{ds}{d\tau} = 0$ sale $s = 1$: el **equilibrio libre de enfermedad** $E_0 = (1, 0)$.
    * Si $s = 1/R_0$, de $\frac{ds}{d\tau} = 0$: $\varepsilon(1 - 1/R_0) = R_0 \cdot \frac{1}{R_0}\, i$, es decir, el **equilibrio endémico**
    $$
    E_1 = \left(\frac{1}{R_0},\; \varepsilon\left(1 - \frac{1}{R_0}\right)\right),
    $$
    que es biológicamente válido ($i^* > 0$) **si y solo si $R_0 > 1$**.

    ### 7.2 Estabilidad

    $$
    J(s, i) = \begin{pmatrix} -\varepsilon - R_0 i & -R_0 s \\ R_0 i & R_0 s - 1 \end{pmatrix}.
    $$

    **En $E_0$:** $J = \begin{pmatrix} -\varepsilon & -R_0 \\ 0 & R_0 - 1\end{pmatrix}$, triangular, con valores propios $-\varepsilon < 0$ y $R_0 - 1$. Estable si $R_0 < 1$; punto silla si $R_0 > 1$.

    **En $E_1$:** sustituyendo $s^* = 1/R_0$ e $i^* = \varepsilon(R_0 - 1)/R_0$,
    $$
    J(E_1) = \begin{pmatrix} -\varepsilon R_0 & -1 \\ \varepsilon(R_0 - 1) & 0 \end{pmatrix},
    \qquad \operatorname{tr} J = -\varepsilon R_0 < 0, \qquad \det J = \varepsilon(R_0 - 1).
    $$
    En dos dimensiones, un equilibrio es estable si y solo si $\operatorname{tr} J < 0$ y $\det J > 0$. Aquí eso ocurre **exactamente cuando $R_0 > 1$**: el equilibrio endémico es estable siempre que existe.

    **Bifurcación transcrítica en $R_0 = 1$.** Cuando $R_0 = 1$, $E_1 = (1, 0) = E_0$: las dos ramas se cruzan y se intercambian la estabilidad, igual que en el SIS. La diferencia con la sección 6 es que ahora sí hay un equilibrio endémico al cual "bifurcar".

    ### 7.3 Oscilaciones amortiguadas

    Los valores propios en $E_1$ son
    $$
    \lambda_\pm = \frac{-\varepsilon R_0 \pm \sqrt{\varepsilon^2 R_0^2 - 4\varepsilon(R_0 - 1)}}{2}.
    $$
    Como $\varepsilon$ es pequeño, $\varepsilon^2 R_0^2 < 4\varepsilon (R_0-1)$ y el discriminante es **negativo**: los valores propios son complejos, $E_1$ es un **foco estable** y la aproximación al equilibrio es mediante **oscilaciones amortiguadas** (epidemias recurrentes cada vez más pequeñas). Para $\varepsilon$ pequeño,
    $$
    \lambda_\pm \approx -\frac{\varepsilon R_0}{2} \pm i\sqrt{\varepsilon (R_0 - 1)}, \qquad
    \text{periodo} \approx \frac{2\pi}{\sqrt{\varepsilon(R_0 - 1)}} \;\text{(en } \tau\text{)}.
    $$
    En unidades con dimensiones, $T \approx 2\pi\sqrt{A\, D}$, donde $D = 1/(\gamma + \mu)$ es la duración de la infección y $A = 1/(\mu(R_0-1))$ es la **edad media a la que la gente se infecta** (Anderson y May, 1991). Para el sarampión antes de la vacuna ($R_0 \approx 15$, $D \approx 2$ semanas, esperanza de vida de 70 años) da $T \approx 2.7$ años, del orden de las epidemias bienales que se observaban en Inglaterra.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_dem = mo.ui.slider(0.5, 10.0, step=0.1, value=5.0, label="$R_0$", show_value=True)
    log_eps = mo.ui.slider(-3.0, -1.0, step=0.1, value=-2.0, label="$\\log_{10}\\varepsilon$", show_value=True)
    return log_eps, r0_dem


@app.cell(hide_code=True)
def _(COLOR, GRIS, integrar, log_eps, mo, np, plt, r0_dem, sir_demografia):
    _R0, _eps = r0_dem.value, 10 ** log_eps.value
    _tau_max = min(4000, 30 / (_eps * max(_R0, 1)))
    _t, _y = integrar(sir_demografia, [0.999, 0.001], _tau_max, _R0, _eps, n=6000)

    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    _ax[0].semilogy(_t, np.maximum(_y[1], 1e-12), color=COLOR["I"])
    if _R0 > 1:
        _ax[0].axhline(_eps * (1 - 1 / _R0), color=GRIS, ls=":", lw=1.5)
    _ax[0].set(title="Infecciosos (escala log)", xlabel=r"$\tau$", ylabel="$i$", ylim=(1e-9, 1))

    _ax[1].plot(_y[0], _y[1], color=COLOR["I"], lw=1.2)
    if _R0 > 1:
        _ax[1].plot(1 / _R0, _eps * (1 - 1 / _R0), "o", color="#0b0b0b", ms=6)
    _ax[1].set(title="Plano fase", xlabel="$s$", ylabel="$i$", yscale="log", ylim=(1e-9, 1))

    _r = np.linspace(0.5, 10, 300)
    _ax[2].plot(_r[_r <= 1], 0 * _r[_r <= 1], color=COLOR["I"], label="estable")
    _ax[2].plot(_r[_r >= 1], 0 * _r[_r >= 1], color=COLOR["I"], ls="--", lw=1.5, label="inestable")
    _ax[2].plot(_r[_r >= 1], (1 - 1 / _r[_r >= 1]), color=COLOR["I"])
    _ax[2].plot(_r[_r <= 1], (1 - 1 / _r[_r <= 1]), color=GRIS, ls="--", lw=1.2)
    _ax[2].axvline(_R0, color=GRIS, lw=1)
    _ax[2].set(title=r"Bifurcación: $i^*/\varepsilon$ contra $R_0$", xlabel="$R_0$", ylabel=r"$i^*/\varepsilon$", ylim=(-1, 1))
    _ax[2].legend(fontsize=8, loc="lower right")
    _fig.tight_layout()

    if _R0 > 1:
        _J = np.array([[-_eps * _R0, -1], [_eps * (_R0 - 1), 0]])
        _lam = np.linalg.eigvals(_J)
        _txt = (f"Valores propios en $E_1$: $\\lambda = {_lam[0].real:.4f} \\pm {abs(_lam[0].imag):.4f}\\,i$. "
                + (f"Periodo de las oscilaciones $\\approx 2\\pi/{abs(_lam[0].imag):.4f} = {2*np.pi/abs(_lam[0].imag):.0f}$ unidades de $\\tau$."
                   if abs(_lam[0].imag) > 0 else "Valores propios reales: no hay oscilaciones."))
    else:
        _txt = "Con $R_0 < 1$ solo existe $E_0$ y es estable: la infección desaparece."
    mo.vstack([mo.hstack([r0_dem, log_eps], justify="start", wrap=True), _fig, mo.md(
        _txt + " Observa en la escala logarítmica que entre olas $i$ baja a valores minúsculos: en una población "
        "real de tamaño finito la enfermedad se extinguiría ahí (el atto-zorro otra vez)."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. SIRS: la inmunidad se pierde

    Otra forma de reponer susceptibles: los recuperados pierden la inmunidad a tasa $\omega$ (duración media de la inmunidad $1/\omega$). Es el caso de la influenza, del SARS-CoV-2 y de los coronavirus estacionales.

    $$
    \frac{dS}{dt} = -\beta\frac{SI}{N} + \omega R, \qquad
    \frac{dI}{dt} = \beta\frac{SI}{N} - \gamma I, \qquad
    \frac{dR}{dt} = \gamma I - \omega R .
    $$

    Con $\tau = \gamma t$, $R_0 = \beta/\gamma$, $\alpha = \omega/\gamma$ (duración de la infección entre duración de la inmunidad) y $r = 1 - s - i$:
    $$
    \frac{ds}{d\tau} = -R_0\, s\, i + \alpha(1 - s - i), \qquad \frac{di}{d\tau} = R_0\, s\, i - i .
    $$

    **Equilibrios.** Otra vez $i = 0$ o $s = 1/R_0$. Con $i = 0$: $s = 1$ ($E_0$). Con $s = 1/R_0$: $i = \alpha(1 - 1/R_0 - i)$, es decir,
    $$
    i^* = \frac{\alpha}{1 + \alpha}\left(1 - \frac{1}{R_0}\right),
    $$
    positivo si y solo si $R_0 > 1$.

    **Estabilidad.** $J = \begin{pmatrix} -R_0 i - \alpha & -R_0 s - \alpha \\ R_0 i & R_0 s - 1 \end{pmatrix}$.

    * En $E_0 = (1, 0)$: $J = \begin{pmatrix} -\alpha & -R_0 - \alpha \\ 0 & R_0 - 1\end{pmatrix}$, valores propios $-\alpha$ y $R_0 - 1$.
    * En $E_1$: $J = \begin{pmatrix} -R_0 i^* - \alpha & -1 - \alpha \\ R_0 i^* & 0\end{pmatrix}$, con $\operatorname{tr} J = -(R_0 i^* + \alpha) < 0$ y $\det J = (1+\alpha)R_0 i^* = \alpha (R_0 - 1)$, positivo si y solo si $R_0 > 1$.

    Otra **bifurcación transcrítica** en $R_0 = 1$. Lo bonito es que el SIRS **contiene a los dos modelos anteriores** como casos límite:

    * $\alpha \to 0$ (inmunidad permanente): $i^* \to 0$, recuperamos el SIR.
    * $\alpha \to \infty$ (la inmunidad se pierde al instante): $i^* \to 1 - 1/R_0$, recuperamos el SIS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_sirs = mo.ui.slider(0.5, 6.0, step=0.1, value=2.5, label="$R_0$", show_value=True)
    log_alfa = mo.ui.slider(-3.0, 1.0, step=0.1, value=-1.5, label="$\\log_{10}\\alpha$", show_value=True)
    return log_alfa, r0_sirs


@app.cell(hide_code=True)
def _(COLOR, GRIS, integrar, log_alfa, mo, np, plt, r0_sirs, sirs):
    _R0, _alfa = r0_sirs.value, 10 ** log_alfa.value
    _t, _y = integrar(sirs, [0.999, 0.001], 400, _R0, _alfa, n=4000)
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax[0].plot(_t, _y[1], color=COLOR["I"], label="$i$")
    _ax[0].plot(_t, _y[0], color=COLOR["S"], label="$s$")
    if _R0 > 1:
        _ax[0].axhline(_alfa / (1 + _alfa) * (1 - 1 / _R0), color=GRIS, ls=":", lw=1.5)
    _ax[0].set(title=f"SIRS con $\\alpha = {_alfa:.3g}$", xlabel=r"$\tau$", ylabel="fracción")
    _ax[0].legend()

    _r = np.linspace(0.5, 6, 300)
    _azules = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    for _a, _c in zip([0.01, 0.1, 1, 10], _azules):
        _ax[1].plot(_r, np.where(_r > 1, _a / (1 + _a) * (1 - 1 / _r), 0), color=_c, lw=1.5, label=f"$\\alpha = {_a:g}$")
    _ax[1].plot(_r, np.where(_r > 1, 1 - 1 / _r, 0), color=_azules[-1], lw=2.5, label=r"$\alpha \to \infty$ (SIS)")
    _ax[1].plot(_r, np.where(_r > 1, _alfa / (1 + _alfa) * (1 - 1 / _r), 0), color=COLOR["I"], lw=2, ls="--", label="tu $\\alpha$")
    _ax[1].axvline(_R0, color=GRIS, lw=1)
    _ax[1].set(title="Equilibrio estable $i^*$ contra $R_0$", xlabel="$R_0$", ylabel="$i^*$")
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.hstack([r0_sirs, log_alfa], justify="start", wrap=True), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9. SEIR: periodo de latencia

    Muchas infecciones tienen un periodo **latente**: la persona ya está infectada pero todavía no contagia. Añadimos el compartimento $E$ (expuestos), del que se sale a tasa $\sigma$ (latencia media $1/\sigma$):

    $$
    \frac{dS}{dt} = -\beta\frac{SI}{N}, \qquad
    \frac{dE}{dt} = \beta\frac{SI}{N} - \sigma E, \qquad
    \frac{dI}{dt} = \sigma E - \gamma I, \qquad
    \frac{dR}{dt} = \gamma I .
    $$

    Adimensionalizado ($\tau = \gamma t$, $R_0 = \beta/\gamma$, $\tilde\sigma = \sigma/\gamma$):
    $\;\dfrac{ds}{d\tau} = -R_0 s i,\;\; \dfrac{de}{d\tau} = R_0 s i - \tilde\sigma e,\;\; \dfrac{di}{d\tau} = \tilde\sigma e - i.$

    **Dos hechos importantes.**

    1. **El tamaño final es el mismo que en el SIR.** Como $\frac{d}{d\tau}\ln s = -R_0 i$, integrando, $\ln\frac{s_\infty}{s_0} = -R_0\int_0^\infty i\,d\tau$, y $\int_0^\infty i\,d\tau$ es la fracción total que pasó por $I$, es decir $1 - s_\infty$ (si al inicio casi todos son susceptibles). Llegamos otra vez a $z = 1 - e^{-R_0 z}$. **La latencia no cambia cuántos se infectan, sino qué tan rápido.**
    2. **La epidemia crece más despacio.** Linealizando en el equilibrio libre de enfermedad ($s = 1$), el subsistema de infectados en unidades con dimensiones es
    $$
    \frac{d}{dt}\begin{pmatrix} E \\ I\end{pmatrix} = \begin{pmatrix} -\sigma & \beta \\ \sigma & -\gamma\end{pmatrix}\begin{pmatrix} E \\ I\end{pmatrix}.
    $$
    La tasa de crecimiento $r$ es el valor propio dominante, raíz de $r^2 + (\sigma + \gamma) r + \sigma(\gamma - \beta) = 0$:
    $$
    r = \frac{-(\sigma + \gamma) + \sqrt{(\sigma - \gamma)^2 + 4\sigma\beta}}{2}.
    $$
    Despejando $\beta$ de la ecuación característica se obtiene una fórmula muy útil: $(\sigma + r)(\gamma + r) = \sigma\beta$, es decir,
    $$
    \boxed{\;R_0 = \frac{\beta}{\gamma} = \left(1 + \frac{r}{\sigma}\right)\left(1 + \frac{r}{\gamma}\right)\;}
    \qquad\text{(en el SIR: } R_0 = 1 + r/\gamma\text{)}.
    $$
    **Con la misma tasa de crecimiento observada $r$, el SEIR implica un $R_0$ mayor que el SIR.** El $R_0$ que se estima con datos depende del modelo que se suponga. Lo usaremos en la sesión 2.

    La estabilidad del equilibrio libre de enfermedad sale de la matriz de arriba: $\operatorname{tr} = -(\sigma + \gamma) < 0$ y $\det = \sigma(\gamma - \beta)$, positivo si y solo si $\beta < \gamma$, es decir $R_0 < 1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_seir = mo.ui.slider(1.1, 6.0, step=0.1, value=2.5, label="$R_0$", show_value=True)
    sig_seir = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="$\\tilde\\sigma = \\sigma/\\gamma$", show_value=True)
    return r0_seir, sig_seir


@app.cell(hide_code=True)
def _(COLOR, integrar, mo, np, plt, r0_seir, seir, sig_seir, sir, tamano_final):
    _R0, _sig = r0_seir.value, sig_seir.value
    _t1, _y1 = integrar(sir, [0.999, 0.001], 80, _R0, n=3000)
    _t2, _y2 = integrar(seir, [0.999, 0.0, 0.001], 80, _R0, _sig, n=3000)
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax[0].plot(_t1, _y1[1], color=COLOR["I"], label="SIR: $i$")
    _ax[0].plot(_t2, _y2[2], color=COLOR["I"], ls="--", label="SEIR: $i$")
    _ax[0].plot(_t2, _y2[1], color=COLOR["E"], ls="--", label="SEIR: $e$")
    _ax[0].set(title="Infecciosos", xlabel=r"$\tau$", ylabel="fracción")
    _ax[0].legend()
    _ax[1].plot(_t1, 1 - _y1[0], color=COLOR["R"], label="SIR")
    _ax[1].plot(_t2, 1 - _y2[0], color=COLOR["R"], ls="--", label="SEIR")
    _ax[1].axhline(tamano_final(_R0), color="#9a9993", ls=":", lw=1.5)
    _ax[1].set(title="Fracción que ya se infectó, $1 - s$", xlabel=r"$\tau$")
    _ax[1].legend()
    _fig.tight_layout()
    _r_sir = _R0 - 1
    _r_seir = (-(_sig + 1) + np.sqrt((_sig - 1) ** 2 + 4 * _sig * _R0)) / 2
    mo.vstack([mo.hstack([r0_seir, sig_seir], justify="start", wrap=True), _fig, mo.md(
        f"Tasa de crecimiento inicial (por periodo infeccioso): SIR $r = {_r_sir:.3f}$; SEIR $r = {_r_seir:.3f}$. "
        f"Ambos llegan al mismo tamaño final $z = {tamano_final(_R0):.3f}$ (línea punteada)."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 10. Cómo calcular $R_0$

    **Definición.** $R_0$ es el número esperado de infecciones secundarias que produce **un** individuo infeccioso típico durante todo su periodo infeccioso, en una población **totalmente susceptible**. Es adimensional (un número de personas por persona).

    ### 10.1 Directamente: tasa × duración

    En el SIR un infeccioso contagia a tasa $\beta$ durante un tiempo medio $1/\gamma$: $R_0 = \beta/\gamma$. Con mortalidad natural, el tiempo medio es $1/(\gamma + \mu)$. En el SEIR con demografía hay que sobrevivir la latencia (probabilidad $\frac{\sigma}{\sigma + \mu}$) y luego contagiar durante $\frac{1}{\gamma+\mu}$:
    $$
    R_0 = \frac{\sigma}{\sigma + \mu}\cdot\frac{\beta}{\gamma + \mu}.
    $$
    Funciona cuando el ciclo de infección es simple; con varios tipos de infecciosos (por edad, por vía de transmisión, hospederos y vectores) se complica.

    ### 10.2 Como umbral de estabilidad

    El equilibrio libre de enfermedad cambia de estabilidad cuando una cantidad cruza 1. Cuidado: el jacobiano te da **una** cantidad umbral, pero no necesariamente $R_0$. Por ejemplo, $\det J < 0$ puede dar la condición "$R_0^2 > 1$", o "$\beta\sigma > \gamma(\sigma+\mu)$ ..." escrita de mil formas equivalentes. Todas marcan el mismo umbral, pero solo una tiene la interpretación de "contagios por caso".

    ### 10.3 La matriz de siguiente generación

    El método general (Diekmann, Heesterbeek y Metz, 1990; van den Driessche y Watmough, 2002):

    1. Quédate solo con los compartimentos **infectados**, $x = (x_1, \dots, x_m)$ (en el SEIR, $x = (E, I)$).
    2. Escribe $\frac{dx}{dt} = \mathcal{F}(x) - \mathcal{V}(x)$, donde $\mathcal{F}$ son las **infecciones nuevas** (entradas a compartimentos infectados desde fuera de ellos) y $\mathcal{V}$ son **todas las demás transiciones** (salidas por recuperación o muerte, y paso de un compartimento infectado a otro).
    3. Calcula los jacobianos en el equilibrio libre de enfermedad: $F = D\mathcal{F}(x_0)$, $V = D\mathcal{V}(x_0)$.
    4. La **matriz de siguiente generación** es $K = F V^{-1}$, y
    $$
    R_0 = \rho(F V^{-1}) \quad\text{(su radio espectral: el valor propio de mayor módulo)}.
    $$

    **Interpretación.** $(V^{-1})_{jk}$ es el tiempo esperado que pasa en el compartimento $j$ un individuo que entró en el compartimento $k$; $F_{ij}$ es la tasa a la que alguien en $j$ produce infecciones nuevas en $i$. Entonces $K_{ik}$ es el **número esperado de infecciones nuevas en el compartimento $i$ producidas por alguien que entró al compartimento $k$**. El radio espectral es la tasa de multiplicación por generación a largo plazo.

    **Ejemplo: SEIR con demografía.** $\mathcal{F} = \begin{pmatrix}\beta S I/N \\ 0\end{pmatrix}$, $\mathcal{V} = \begin{pmatrix} (\sigma + \mu)E \\ -\sigma E + (\gamma + \mu) I\end{pmatrix}$. En el equilibrio libre de enfermedad ($S = N$):
    $$
    F = \begin{pmatrix} 0 & \beta \\ 0 & 0\end{pmatrix}, \qquad
    V = \begin{pmatrix} \sigma + \mu & 0 \\ -\sigma & \gamma + \mu\end{pmatrix}, \qquad
    V^{-1} = \frac{1}{(\sigma+\mu)(\gamma+\mu)}\begin{pmatrix} \gamma + \mu & 0 \\ \sigma & \sigma + \mu\end{pmatrix},
    $$
    $$
    K = F V^{-1} = \begin{pmatrix} \dfrac{\beta\sigma}{(\sigma+\mu)(\gamma+\mu)} & \dfrac{\beta}{\gamma + \mu} \\ 0 & 0 \end{pmatrix}
    \quad\Longrightarrow\quad R_0 = \frac{\beta\sigma}{(\sigma+\mu)(\gamma+\mu)},
    $$
    lo mismo que obtuvimos "a mano" en 10.1. La celda siguiente lo hace con `sympy` y luego lo aplica a un caso donde el método directo no es obvio.
    """)
    return


@app.cell
def _():
    import sympy as sp

    def siguiente_generacion(F, V):
        """Matriz de siguiente generación K = F V^{-1} y sus valores propios."""
        K = (F * V.inv()).applyfunc(sp.factor)
        return K, [sp.factor(v) for v in K.eigenvals()]

    beta, sigma, gamma, mu = sp.symbols("beta sigma gamma mu", positive=True)
    F_seir = sp.Matrix([[0, beta], [0, 0]])
    V_seir = sp.Matrix([[sigma + mu, 0], [-sigma, gamma + mu]])
    K_seir, valores_seir = siguiente_generacion(F_seir, V_seir)
    return K_seir, siguiente_generacion, sp, valores_seir


@app.cell(hide_code=True)
def _(K_seir, mo, sp, valores_seir):
    mo.md(rf"""
    **SEIR con demografía (sympy):** $K = {sp.latex(K_seir)}$, con valores propios ${', '.join(sp.latex(v) for v in valores_seir)}$. El radio espectral es el no nulo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Ejemplo donde el método brilla: una enfermedad transmitida por vector** (como el dengue, endémico en buena parte de México). Hospederos humanos ($N_h$) y mosquitos ($N_v$). Un mosquito infeccioso infecta humanos a tasa $\beta_h$; un humano infeccioso infecta mosquitos a tasa $\beta_v N_v/N_h$ (más mosquitos por persona, más picaduras). Los humanos se recuperan a tasa $\gamma$ y los mosquitos mueren a tasa $\mu_v$ (nunca se recuperan). Compartimentos infectados: $x = (I_h, I_v)$.
    $$
    \frac{dI_h}{dt} = \beta_h \frac{S_h}{N_h} I_v - \gamma I_h, \qquad
    \frac{dI_v}{dt} = \beta_v \frac{S_v}{N_h} I_h - \mu_v I_v .
    $$
    No hay ningún compartimento que contagie a su mismo tipo: un humano produce mosquitos infectados, y un mosquito produce humanos infectados. La matriz $K$ es **anti-diagonal**:
    """)
    return


@app.cell
def _(siguiente_generacion, sp):
    beta_h, beta_v, gam, mu_v, N_h, N_v = sp.symbols("beta_h beta_v gamma mu_v N_h N_v", positive=True)
    F_vector = sp.Matrix([[0, beta_h], [beta_v * N_v / N_h, 0]])
    V_vector = sp.Matrix([[gam, 0], [0, mu_v]])
    K_vector, valores_vector = siguiente_generacion(F_vector, V_vector)
    return K_vector, valores_vector


@app.cell(hide_code=True)
def _(K_vector, mo, sp, valores_vector):
    mo.md(rf"""
    $$K = {sp.latex(K_vector)}, \qquad \text{{valores propios: }} {', '.join(sp.latex(v) for v in valores_vector)}.$$

    El elemento $K_{{21}} = \frac{{\beta_v N_v}}{{\gamma N_h}}$ es cuántos mosquitos infecta un humano durante su infección; $K_{{12}} = \frac{{\beta_h}}{{\mu_v}}$ es cuántos humanos infecta un mosquito durante su vida. Entonces
    $$
    R_0 = \sqrt{{\frac{{\beta_h}}{{\mu_v}}\cdot\frac{{\beta_v N_v}}{{\gamma N_h}}}} ,
    $$
    la **media geométrica** de las dos "medias generaciones". Algunos textos (por ejemplo, la tradición de Ross y Macdonald) llaman $R_0$ al producto sin raíz: el número de humanos infectados por un humano después de un ciclo completo humano → mosquito → humano. Las dos definiciones tienen el **mismo umbral** ($R_0 = 1$), pero valores distintos: hay que saber cuál usa cada artículo.

    Observa que aquí $R_0$ **sí depende de una densidad**: el número de mosquitos por persona, $N_v/N_h$. Por eso el control del vector (descacharrización, larvicidas) reduce $R_0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10.4 A partir de la tasa de crecimiento (lo que haremos con datos)

    Al inicio de una epidemia, casi todos son susceptibles y el sistema linealizado predice crecimiento exponencial, $I(t) \propto e^{rt}$. Ya vimos que
    $$
    \text{SIR: } R_0 = 1 + \frac{r}{\gamma}, \qquad \text{SEIR: } R_0 = \left(1 + \frac{r}{\sigma}\right)\left(1 + \frac{r}{\gamma}\right).
    $$
    En general (Wallinga y Lipsitch, 2007), si $g(a)$ es la densidad del **intervalo generacional** (el tiempo entre que alguien se infecta y que infecta a otro),
    $$
    R_0 = \frac{1}{\int_0^\infty e^{-r a} g(a)\, da}.
    $$
    Moraleja: **los datos te dan $r$; para convertir $r$ en $R_0$ necesitas saber la escala de tiempo de la transmisión**, que viene de estudios clínicos o de contactos, no de la curva epidémica. Lo veremos en la sesión 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 11. Vacunación

    ### 11.1 Vacunar al nacer: la cobertura crítica

    En el SIR con demografía, supongamos que una fracción $p$ de los recién nacidos se vacuna (con una vacuna perfecta) y entra directamente a $R$. Adimensionalizado:
    $$
    \frac{ds}{d\tau} = \varepsilon(1 - p - s) - R_0\, s\, i, \qquad \frac{di}{d\tau} = R_0\, s\, i - i .
    $$

    * Equilibrio libre de enfermedad: $E_0 = (1 - p, 0)$. Su jacobiano es $\begin{pmatrix} -\varepsilon & -R_0(1-p) \\ 0 & R_0(1-p) - 1\end{pmatrix}$: estable si y solo si $R_0(1 - p) < 1$.
    * Equilibrio endémico: $s^* = 1/R_0$, $i^* = \varepsilon\left(1 - p - \frac{1}{R_0}\right)$, que existe si y solo si $R_0(1-p) > 1$.

    Ahora **el parámetro de bifurcación es la cobertura $p$**: hay una bifurcación transcrítica en
    $$
    \boxed{\;p_c = 1 - \frac{1}{R_0}\;}
    $$
    la **cobertura crítica**. Si vacunamos a más de $p_c$, la enfermedad se elimina (aunque no todos estén protegidos: eso es la **inmunidad de rebaño**).

    ### 11.2 Vacunas imperfectas

    Si la vacuna protege completamente a una fracción $e$ de los vacunados y no hace nada en los demás (vacuna "todo o nada" con eficacia $e$), la fracción efectivamente protegida es $e\,p$ y
    $$
    p_c = \frac{1}{e}\left(1 - \frac{1}{R_0}\right).
    $$
    **Si $e < 1 - 1/R_0$, entonces $p_c > 1$: ni vacunando al 100 % se elimina la enfermedad.** Con $R_0 = 5$ y $e = 0.7$, por ejemplo, $p_c = 0.8/0.7 \approx 1.14$. Esa es la razón matemática por la que vacunas que protegen poco contra la *infección* (aunque protejan mucho contra la enfermedad grave) no producen inmunidad de rebaño. (Hay otro tipo de vacuna imperfecta, la "con fugas" o *leaky*, que reduce la probabilidad de infección en cada contacto; da el mismo umbral pero distinta dinámica.)

    ### 11.3 Vacunar durante una epidemia

    En el SIR sin demografía, una campaña que inmuniza a los susceptibles a tasa $\nu$ (en unidades de $1/\gamma$) a partir de $\tau_v$:
    $$
    \frac{ds}{d\tau} = -R_0\, s\, i - \nu\, s, \qquad \frac{di}{d\tau} = R_0\, s\, i - i, \qquad \frac{dv}{d\tau} = \nu\, s .
    $$
    En la figura, $\nu$ es la fracción de los susceptibles que se vacuna por periodo infeccioso y $\tau_v$ el inicio de la campaña. Ahora $R_{\text{ef}}(\tau) = R_0\, s(\tau)$ baja por dos razones: por los contagios y por la vacunación. La epidemia empieza a bajar cuando $R_{\text{ef}} < 1$. Esto es lo que usaremos en la sesión 2 con los datos de vacunación de México.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    r0_vac = mo.ui.slider(1.1, 15.0, step=0.1, value=3.0, label="$R_0$", show_value=True)
    efic = mo.ui.slider(0.3, 1.0, step=0.01, value=0.9, label="eficacia $e$", show_value=True)
    return efic, r0_vac


@app.cell(hide_code=True)
def _(COLOR, GRIS, efic, mo, np, plt, r0_vac):
    _R0, _e = r0_vac.value, efic.value
    _pc = (1 - 1 / _R0) / _e
    _p = np.linspace(0, 1, 400)
    _istar = 1 - _e * _p - 1 / _R0            # i*/epsilon
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax[0].plot(_p[_istar >= 0], _istar[_istar >= 0], color=COLOR["I"], label="endémico (estable)")
    _ax[0].plot(_p[_istar <= 0], _istar[_istar <= 0], color=GRIS, ls="--", lw=1.2, label="sin sentido ($i^*<0$)")
    _ax[0].plot(_p[_istar >= 0], 0 * _p[_istar >= 0], color=COLOR["I"], ls="--", lw=1.5, label="libre de enfermedad (inestable)")
    _ax[0].plot(_p[_istar <= 0], 0 * _p[_istar <= 0], color=COLOR["I"], lw=2, label="libre de enfermedad (estable)")
    if _pc <= 1:
        _ax[0].axvline(_pc, color=GRIS, ls=":", lw=1.5)
    _ax[0].set(title="Bifurcación en la cobertura $p$", xlabel="cobertura de vacunación $p$", ylabel=r"$i^*/\varepsilon$")
    _ax[0].legend(fontsize=7.5)

    _R = np.linspace(1.01, 20, 400)
    for _ee, _c in zip([1.0, 0.9, 0.7, 0.5], ["#104281", "#256abf", "#3987e5", "#86b6ef"]):
        _ax[1].plot(_R, (1 - 1 / _R) / _ee, color=_c, lw=1.8, label=f"$e = {_ee:g}$")
    _ax[1].axhline(1, color=GRIS, lw=1)
    _ax[1].plot(_R0, _pc, "o", color=COLOR["I"], ms=7)
    _ax[1].set(title="Cobertura crítica $p_c$", xlabel="$R_0$", ylabel="$p_c$", ylim=(0, 1.6))
    _ax[1].text(19.5, 1.03, "imposible por encima de 1", ha="right", fontsize=8)
    _ax[1].legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    mo.vstack([mo.hstack([r0_vac, efic], justify="start", wrap=True), _fig, mo.md(
        f"Con $R_0 = {_R0:.1f}$ y eficacia $e = {_e:.2f}$: $p_c = {_pc:.3f}$"
        + (" — **mayor que 1: no se puede eliminar la enfermedad solo vacunando.**" if _pc > 1 else ".")
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    r0_camp = mo.ui.slider(1.5, 6.0, step=0.1, value=2.5, label="$R_0$", show_value=True)
    nu_camp = mo.ui.slider(0.0, 0.3, step=0.005, value=0.1, label="vacunación $\\nu$", show_value=True)
    tv_camp = mo.ui.slider(0, 15, step=0.5, value=2, label="inicio $\\tau_v$", show_value=True)
    return nu_camp, r0_camp, tv_camp


@app.cell(hide_code=True)
def _(COLOR, GRIS, integrar, mo, nu_camp, plt, r0_camp, sir_vacunacion, tv_camp):
    _R0, _nu, _tv = r0_camp.value, nu_camp.value, tv_camp.value
    _t0, _y0 = integrar(sir_vacunacion, [0.999, 0.001, 0.0], 40, _R0, 0.0, _tv, n=3000, max_step=0.05)
    _t1, _y1 = integrar(sir_vacunacion, [0.999, 0.001, 0.0], 40, _R0, _nu, _tv, n=3000, max_step=0.05)
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax[0].plot(_t0, _y0[1], color=GRIS, label="sin vacunación")
    _ax[0].plot(_t1, _y1[1], color=COLOR["I"], label="con vacunación")
    _ax[0].axvline(_tv, color=GRIS, ls=":", lw=1.2)
    _ax[0].set(title="Infecciosos $i$", xlabel=r"$\tau$", ylabel="fracción")
    _ax[0].legend()
    _ax[1].plot(_t0, _R0 * _y0[0], color=GRIS, label="sin vacunación")
    _ax[1].plot(_t1, _R0 * _y1[0], color=COLOR["V"], label="con vacunación")
    _ax[1].axhline(1, color="#0b0b0b", lw=1)
    _ax[1].set(title=r"Número reproductivo efectivo $R_{\mathrm{ef}} = R_0\, s$", xlabel=r"$\tau$")
    _ax[1].legend()
    _fig.tight_layout()
    _z0 = 1 - _y0[0][-1]
    _z1 = 1 - _y1[0][-1] - _y1[2][-1]
    mo.vstack([mo.hstack([r0_camp, nu_camp, tv_camp], justify="start", wrap=True), _fig, mo.md(
        f"Fracción que se infecta: **{_z0:.3f}** sin vacunación y **{_z1:.3f}** con vacunación "
        f"(se vacunó el {100 * _y1[2][-1]:.1f} % de la población). Prueba a empezar la campaña tarde: "
        "si llega después del pico casi no cambia el tamaño de la ola."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 12. Resumen

    | Modelo | Variables (adim.) | Parámetros adim. | Equilibrios | Condición | ¿Bifurcación? |
    |---|---|---|---|---|---|
    | SI | $i$ | ninguno | $0$ (inest.), $1$ (est.) | — | no |
    | SIS | $i$ | $R_0 = \beta/\gamma$ | $0$; $\;1 - 1/R_0$ | endémico si $R_0 > 1$ | transcrítica en $R_0 = 1$ |
    | SIR | $s, i$ | $R_0 = \beta/\gamma$ | recta $\{(s^*, 0)\}$ | brote si $R_0 s_0 > 1$ | **no** (umbral; no hiperbólico) |
    | SIR + demografía | $s, i$ | $R_0 = \frac{\beta}{\gamma+\mu}$, $\varepsilon = \frac{\mu}{\gamma+\mu}$ | $(1, 0)$; $\left(\frac{1}{R_0}, \varepsilon(1 - \frac{1}{R_0})\right)$ | endémico (foco) si $R_0 > 1$ | transcrítica en $R_0 = 1$ |
    | SIRS | $s, i$ | $R_0 = \beta/\gamma$, $\alpha = \omega/\gamma$ | $(1, 0)$; $\left(\frac{1}{R_0}, \frac{\alpha}{1+\alpha}(1 - \frac{1}{R_0})\right)$ | endémico si $R_0 > 1$ | transcrítica en $R_0 = 1$ |
    | SEIR | $s, e, i$ | $R_0 = \beta/\gamma$, $\tilde\sigma = \sigma/\gamma$ | recta $\{(s^*, 0, 0)\}$ | brote si $R_0 s_0 > 1$ | no (como el SIR) |
    | Vacunación al nacer | $s, i$ | $R_0$, $\varepsilon$, $p$, $e$ | $(1 - p, 0)$; endémico | endémico si $p < p_c$ | transcrítica en $p_c = \frac{1}{e}(1 - \frac{1}{R_0})$ |

    **Ideas que hay que llevarse.**

    1. Las variables son **densidades continuas** (como la biomasa), no personas. El modelo es el límite de población infinita del proceso aleatorio real.
    2. **Adimensionalizar** reduce los parámetros a los que de verdad importan ($R_0$, $\varepsilon$, $\alpha$, …) y deja claro que la EDO no sabe nada del tamaño $N$.
    3. El **umbral $R_0 = 1$** aparece en todos los modelos, pero solo es una bifurcación cuando existe un equilibrio endémico (SIS, SIR con demografía, SIRS, vacunación).
    4. **$R_0$ se calcula** con la matriz de siguiente generación; con datos, se infiere de la tasa de crecimiento **más** una suposición sobre la escala de tiempo de la transmisión.

    ### Referencias

    * Anderson, R. M. y May, R. M. (1991). *Infectious Diseases of Humans: Dynamics and Control*. Oxford University Press.
    * Brauer, F., Castillo-Chavez, C. y Feng, Z. (2019). *Mathematical Models in Epidemiology*. Springer.
    * Diekmann, O., Heesterbeek, J. A. P. y Metz, J. A. J. (1990). On the definition and the computation of the basic reproduction ratio $R_0$ in models for infectious diseases in heterogeneous populations. *Journal of Mathematical Biology*, 28, 365–382.
    * Hethcote, H. W. (2000). The mathematics of infectious diseases. *SIAM Review*, 42(4), 599–653.
    * Keeling, M. J. y Rohani, P. (2008). *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press.
    * Kermack, W. O. y McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. *Proceedings of the Royal Society A*, 115, 700–721.
    * Kurtz, T. G. (1970). Solutions of ordinary differential equations as limits of pure jump Markov processes. *Journal of Applied Probability*, 7(1), 49–58.
    * McCallum, H., Barlow, N. y Hone, J. (2001). How should pathogen transmission be modelled? *Trends in Ecology & Evolution*, 16(6), 295–300.
    * Mollison, D. (1991). Dependence of epidemic and population velocities on basic parameters. *Mathematical Biosciences*, 107, 255–287.
    * Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*, 2.ª ed. Westview Press (cap. 3: bifurcaciones).
    * van den Driessche, P. y Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180, 29–48.
    * Wallinga, J. y Lipsitch, M. (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proceedings of the Royal Society B*, 274, 599–604.
    """)
    return


if __name__ == "__main__":
    app.run()
