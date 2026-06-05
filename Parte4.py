import numpy as np
import matplotlib.pyplot as plt

# Funciones ya hechas en archivos anteriores
from potenciales import potencial_morse
from schrodinger import resolver_schrodinger_1d, estados_ligados
from Parte2 import integrar_rk4_morse
from Parte3 import superposicion, evo_superposicion


def ve_x(x, psi_t):
    """
    Calcula:
        <x>(t) = integral [ Psi*(x,t) * x * Psi(x,t) dx ]
    """
    integrando = np.conjugate(psi_t) * x * psi_t
    return np.real(np.trapezoid(integrando, x))


def ve_p(x, psi_t, hbar=1):
    """
    Calcula:
        <p>(t) = integral [ Psi*(x,t) (-i hbar d/dx) Psi(x,t) dx ]
    """
    dpsi_dx = np.gradient(psi_t, x)
    integrando = np.conjugate(psi_t) * (-1j * hbar * dpsi_dx)
    return np.real(np.trapezoid(integrando, x))


# Obtiene ⟨x⟩(t) y ⟨p⟩(t) para un rango de tiempo
def trayectoriaQ(x, energias, autofunciones, indices_estados, coeficientes, tiempos, hbar=1):
    
    x_esperado = np.zeros(len(tiempos))
    p_esperado = np.zeros(len(tiempos))

    for i, t in enumerate(tiempos):
        psi_t = evo_superposicion(energias, autofunciones, indices_estados, coeficientes, t, hbar=hbar)

        x_esperado[i] = ve_x(x, psi_t)
        p_esperado[i] = ve_p(x, psi_t, hbar=hbar)

    return x_esperado, p_esperado

# Por si necesito usar alguna función de esta parte en el futuro
if __name__ == "__main__":

    
    # Parametros
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

    
    # Potencial de Morse y schrodinger para Morse
    V_morse = potencial_morse(x, D_e, a, x_e, desplazado=True)

    energias_morse, psi_morse = resolver_schrodinger_1d(x, V_morse, masa=masa, hbar=hbar)
    energias_morse_lig, psi_morse_lig = estados_ligados(energias_morse, psi_morse, umbral=0.0)

   
    # Superposición
    indices_estados = [0, 1, 2]
    coeficientes = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)], dtype=complex)

    psi0 = superposicion(psi_morse_lig, indices_estados, coeficientes)

    # Rango de tiempo utilizado
    dt = 0.01 # Paso
    t_max = 10
    tiempos = np.arange(0, t_max + dt, dt)

   
    # Valores esperados (posición y momentum) en distintos tiempos ==> Trayectoria
    
    x_q, p_q = trayectoriaQ(x, energias_morse_lig, psi_morse_lig, indices_estados, coeficientes, tiempos, hbar=hbar)


    # ==========================================================
    # Comparación Clasica vs Cuantica (con gráficos)
    # ==========================================================

    # Trayectoria clásica desde los valores esperados cuánticos en t=0
    x0_cl = x_q[0]
    p0_cl = p_q[0]

    
    # Trayectoria clasica a partir del rk4 para el morse en Parte2.py
    t_cl, x_cl, p_cl, E_cl = integrar_rk4_morse(x0_cl, p0_cl, t_max, dt, D_e, a, x_e, masa=masa)


    
    # Gráfico <x>(t) vs x_cl(t)
    plt.figure(figsize=(10, 5))
    plt.plot(tiempos, x_q, linewidth=2, label=r"$\langle x \rangle (t)$ cuántico")
    plt.plot(t_cl, x_cl, linewidth=2, linestyle="--", label=r"$x_{cl}(t)$ clásico")
    plt.xlabel("Tiempo")
    plt.ylabel("Posición")
    #plt.title("Comparación de posición: cuántico vs clásico")
    plt.grid(True)
    plt.legend()
    plt.show()

    
    # Gráfico <p>(t) vs p_cl(t)
    plt.figure(figsize=(10, 5))
    plt.plot(tiempos, p_q, linewidth=2, label=r"$\langle p \rangle (t)$ cuántico")
    plt.plot(t_cl, p_cl, linewidth=2, linestyle="--", label=r"$p_{cl}(t)$ clásico")
    plt.xlabel("Tiempo")
    plt.ylabel("Momento")
    #plt.title("Comparación de momento: cuántico vs clásico")
    plt.grid(True)
    plt.legend()
    plt.show()

    
    # Espacio de fase
    plt.figure(figsize=(7, 6))
    plt.plot(x_q, p_q, linewidth=2, label=r"Cuántico: $\langle p \rangle$ vs $\langle x \rangle$")
    plt.plot(x_cl, p_cl, linewidth=2, linestyle="--", label="Clásico: p vs x")
    plt.xlabel("Posición")
    plt.ylabel("Momento")
    #plt.title("Comparación cuántica vs clásica en el espacio de fase")
    plt.grid(True)
    plt.legend()
    plt.show()
