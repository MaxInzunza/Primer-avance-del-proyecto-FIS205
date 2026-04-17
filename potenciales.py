import numpy as np


def potencial_morse(x, D_e, a, x_e, desplazado=True):
    
    V = D_e * (1 - np.exp(-a * (x - x_e)))**2

    if desplazado:
        V = V - D_e

    return V


def potencial_armonico(x, D_e, a, x_e, desplazado=True, masa=1):
    """
    Potencial armónico cerca del mínimo del Morse.

        k = 2 * D_e * a^2

    Entonces:
        V_HO(x) = 0.5 * k * (x - x_e)^2

    """
    k = 2 * D_e * a**2
    V = 0.5 * k * (x - x_e)**2

    if desplazado:
        V = V - D_e

    return V


def frecuencia_armonica_equivalente(D_e, a, masa=1):
    """
        omega = sqrt(k / m)
              = a * sqrt(2 * D_e / m)
    """
    return a * np.sqrt(2 * D_e / masa)


def energias_armonicas_analiticas(n_max, D_e, a, masa=1, hbar=1, desplazado=True):
    """

        E_n = hbar * omega * (n + 1/2)

    Si desplazado=True:
        E_n = -D_e + hbar * omega * (n + 1/2)

    """
    omega = frecuencia_armonica_equivalente(D_e, a, masa=masa)
    n = np.arange(n_max + 1)
    E = hbar * omega * (n + 0.5)

    if desplazado:
        E = E - D_e

    return E

# Algunas funciones para la parte 2 (RK4)

def fuerza_morse(x, D_e, a, x_e):
    
    exp_term = np.exp(-a * (x - x_e))
    dVdx_morse = 2 * a * D_e * exp_term * (1 - exp_term)

    return -dVdx_morse


def energia_clasica(x, p, D_e, a, x_e, masa=1, desplazado=True):
    
    # E = p^2 / (2m) + V(x)
    
    V = potencial_morse(x, D_e, a, x_e, desplazado=desplazado)
    T = p**2 / (2 * masa)
    return T + V