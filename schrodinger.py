import numpy as np
from scipy.linalg import eigh_tridiagonal


def construir_hamiltoniano(x, V, masa=1.0, hbar=1.0):
    """
    Construye el Hamiltoniano 1D

    H = T + V
    T = -(hbar^2 / 2m) d^2/dx^2

    """
    dx = x[1] - x[0]

    coef = hbar**2 / (2 * masa * dx**2)

    # Parte cinética 
    diagonal = 2 * coef + V
    subdiagonal = -coef * np.ones(len(x) - 1)

    return diagonal, subdiagonal, dx


def normalizar_funciones_onda(x, psi):
    """
    Normaliza las funciones de onda |psi|^2 dx = 1
    """
    psi_norm = np.zeros_like(psi)

    for n in range(psi.shape[1]):
        norma = np.sqrt(np.trapezoid(np.abs(psi[:, n])**2, x))
        psi_norm[:, n] = psi[:, n] / norma

    return psi_norm


def resolver_schrodinger_1d(x, V, masa=1, hbar=1):
  
    diagonal, subdiagonal, dx = construir_hamiltoniano(x, V, masa=masa, hbar=hbar)

    energias, autofunciones = eigh_tridiagonal(diagonal, subdiagonal)
    autofunciones = normalizar_funciones_onda(x, autofunciones)

    return energias, autofunciones


def estados_ligados(energias, autofunciones, umbral=0):
    """
    Filtra los estados ligados, definidos por E < umbral.
    
    Morse con umbral en 0
    """
    mascara = energias < umbral
    return energias[mascara], autofunciones[:, mascara]