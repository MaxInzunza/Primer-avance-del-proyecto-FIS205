import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from potenciales import potencial_morse, potencial_armonico
from schrodinger import resolver_schrodinger_1d, estados_ligados


def superposicion(autofunciones, indices_estados, coeficientes):

    psi_0 = np.zeros(autofunciones.shape[0], dtype=complex)

    for i, n in enumerate(indices_estados):
        psi_0 += coeficientes[i] * autofunciones[:, n]

    return psi_0


def evo_superposicion(energias, autofunciones, indices_estados, coeficientes, t, hbar=1):
    
    psi_t = np.zeros(autofunciones.shape[0], dtype=complex)

    for i, n in enumerate(indices_estados):
        fase = np.exp(-1j * energias[n] * t / hbar)
        psi_t += coeficientes[i] * autofunciones[:, n] * fase

    return psi_t


def densidad_probabilidad(psi_t):
    
    return np.abs(psi_t) ** 2


# Para poder utilizar las funciones de arriba en otros archivos sin que corran el codigo completo (Uso funciones de acá en la parte 4)
if __name__ == "__main__":

    # Parametros usados
    D_e = 8
    a = 0.9
    x_e = 0
    masa = 1
    hbar = 1


    # Malla espacial
    x_min = -2
    x_max = 8
    N = 1200
    x = np.linspace(x_min, x_max, N)

    # ==========================================================
    # Potenciales y shrodinger para Morse, HO
    # ==========================================================
    V_morse = potencial_morse(x, D_e, a, x_e, desplazado=True)
    V_ho = potencial_armonico(x, D_e, a, x_e, desplazado=True, masa=masa)


    energias_morse, psi_morse = resolver_schrodinger_1d(x, V_morse, masa=masa, hbar=hbar)
    energias_morse_lig, psi_morse_lig = estados_ligados(energias_morse, psi_morse, umbral=0)


    energias_ho, psi_ho = resolver_schrodinger_1d(x, V_ho, masa=masa, hbar=hbar)


    # ==========================================================
    # Superposicion de estados
    # ==========================================================

    # Usando tres estados:
    # Psi(x,0) = c0 * psi_0 + c1 * psi_1 + c2 * psi_2

    indices_estados = [0, 1, 2]
    coeficientes = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)], dtype=complex)

    # Psi_0 para Morse y HO
    psi0_morse = superposicion(psi_morse_lig, indices_estados, coeficientes)
    psi0_ho = superposicion(psi_ho, indices_estados, coeficientes)

    # ==========================================================
    # Graficos estados estacionarios Morse y HO
    # ==========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    escala = 0.5

    # Panel Morse
    axes[0].plot(x, V_morse, color="black", linewidth=2, label="Potencial de Morse")
    for n in indices_estados:
        E = energias_morse_lig[n]
        psi_n = psi_morse_lig[:, n]
        axes[0].hlines(E, x[0], x[-1], colors="gray", linestyles=":", linewidth=1)
        axes[0].plot(x, E + escala * psi_n, label=f"n={n}")
    axes[0].set_title("Estados estacionarios en Morse")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Energía / amplitud")
    axes[0].set_xlim(-1.2, 4.5)
    axes[0].set_ylim(-9, 6)
    axes[0].grid(True)
    axes[0].legend()

    # Panel armónico
    axes[1].plot(x, V_ho, color="black", linewidth=2, label="Potencial armónico")
    for n in indices_estados:
        E = energias_ho[n]
        psi_n = psi_ho[:, n]
        axes[1].hlines(E, x[0], x[-1], colors="gray", linestyles=":", linewidth=1)
        axes[1].plot(x, E + escala * psi_n, label=f"n={n}")
    axes[1].set_title("Estados estacionarios en HO")
    axes[1].set_xlabel("x")
    axes[1].set_xlim(-1.2, 4.5)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


    # Parametros usados para el slider (Graficos desde t=0 a t=10 interactivo)
    t_min = 0
    t_max = 10
    t_ini = 0

    # Prob de densidad inicial t = 0 para Morse y HO
    psi_t_morse = evo_superposicion(energias_morse_lig, psi_morse_lig, indices_estados, coeficientes, t_ini, hbar=hbar)
    dens_morse = densidad_probabilidad(psi_t_morse)

    psi_t_ho = evo_superposicion(energias_ho, psi_ho, indices_estados, coeficientes, t_ini, hbar=hbar)
    dens_ho = densidad_probabilidad(psi_t_ho)

    # ==========================================================
    # Grafico comparativo para la densidad de prob para Morse y HO en t = [0,10]
    # ==========================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plt.subplots_adjust(bottom=0.22)

    # Panel Morse
    linea_morse, = axes[0].plot(x, dens_morse, linewidth=2, label="Morse")
    axes[0].set_title(f"Morse: t = {t_ini:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$|\Psi(x,t)|^2$")
    axes[0].set_xlim(-1.2, 4.5)
    axes[0].grid(True)
    axes[0].legend()

    # Panel armónico
    linea_ho, = axes[1].plot(x, dens_ho, linewidth=2, linestyle="--", label="Armónico")
    axes[1].set_title(f"Armónico: t = {t_ini:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_xlim(-1.2, 4.5)
    axes[1].grid(True)
    axes[1].legend()

    # Fijar escala vertical para evitar errores en el grafico
    y_max = 1.1 * max(np.max(dens_morse), np.max(dens_ho))
    axes[0].set_ylim(0, y_max)
    axes[1].set_ylim(0, y_max)

    # Eje del slider
    ax_slider = plt.axes([0.18, 0.08, 0.64, 0.04])

    slider_t = Slider(ax=ax_slider, label='Tiempo t', valmin=t_min, valmax=t_max, valinit=t_ini, valstep=0.05)

    # ==========================================================
    # Funcion para actualizar la densidad de prob con sliders
    # ==========================================================

    # Esta funcion solamente esta hecha para que se pueda ver visualmente el cambio a distintos tiempos y poder comparar mejor entre Morse y HO    
    def actualizar(val):
        t = slider_t.val

        # Recalcular densidad de prob en Morse
        psi_t_morse = evo_superposicion(energias_morse_lig, psi_morse_lig, indices_estados, coeficientes, t, hbar=hbar)
        dens_morse = densidad_probabilidad(psi_t_morse)

        # Recalcular densidad de prob en HO
        psi_t_ho = evo_superposicion(energias_ho, psi_ho, indices_estados, coeficientes, t, hbar=hbar)
        dens_ho = densidad_probabilidad(psi_t_ho)

        # Actualizar curvas
        linea_morse.set_ydata(dens_morse)
        linea_ho.set_ydata(dens_ho)

        # Actualizar títulos
        axes[0].set_title(f"Morse: t = {t:.2f}")
        axes[1].set_title(f"Armónico: t = {t:.2f}")

        fig.canvas.draw_idle()

    slider_t.on_changed(actualizar)

    plt.show()

    # ==========================================================
    # Comparacion densidad de prob para Morse y HO en tiempo especifico
    # ==========================================================


    t_comp = 4.5

    psi_t_morse = evo_superposicion(energias_morse_lig, psi_morse_lig, indices_estados, coeficientes, t_comp, hbar=hbar)
    dens_morse = densidad_probabilidad(psi_t_morse)

    psi_t_ho = evo_superposicion(energias_ho, psi_ho, indices_estados, coeficientes, t_comp, hbar=hbar)
    dens_ho = densidad_probabilidad(psi_t_ho)

    plt.figure(figsize=(10, 5))
    plt.plot(x, dens_morse, linewidth=2, label="Morse")
    plt.plot(x, dens_ho, linewidth=2, linestyle="--", label="Armónico")
    plt.xlabel("x")
    plt.ylabel(r"$|\Psi(x,t)|^2$")
    #plt.title(f"Comparación de densidad de probabilidad en t = {t_comp}") 
    plt.xlim(-1.2, 4.5)
    plt.grid(True)
    plt.legend()
    plt.show()
