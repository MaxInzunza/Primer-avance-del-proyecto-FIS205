import numpy as np
import matplotlib.pyplot as plt
from potenciales import potencial_morse, potencial_armonico
from schrodinger import resolver_schrodinger_1d, estados_ligados


def momento_dipolar_cuadrado(x, psi_ini, psi_fi):
    int = np.conjugate(psi_fi) * x * psi_ini
    md = np.trapezoid(int, x)
    
    return np.abs(md)**2


def perfil_lorentziano(E_malla, delta_E, gamma):
    num = gamma/2
    den = (E_malla - delta_E)**2 + (gamma / 2)**2
    
    return (1 / np.pi) * (num / den)


def generar_espectro_IR(x, energias, autofunciones, E_malla, gamma=0.1):

    espectro_tot = np.zeros_like(E_malla)
    n_estados = len(energias)
    
    for v in range(n_estados - 1):
        
        delta_E = energias[v+1] - energias[v]
        
        psi_v = autofunciones[:, v]
        psi_v1 = autofunciones[:, v+1]
        intensidad = momento_dipolar_cuadrado(x, psi_v, psi_v1)
        
        espectro_tot += intensidad * perfil_lorentziano(E_malla, delta_E, gamma)
        
    return espectro_tot


if __name__ == "__main__":
    
    
    # Parámetros (Mismos que en la sotras partes)
    D_e = 8
    a = 0.9
    x_e = 0
    masa = 1
    hbar = 1

    # Para el espectro IR (ancho)
    gamma = 0.15 
    
    # Malla espacial
    x_min = -2
    x_max = 8
    N = 1200
    x = np.linspace(x_min, x_max, N)
    
    
    # Potenciales Morse, Ho y Energías
    
    V_morse = potencial_morse(x, D_e, a, x_e, desplazado=True)
    V_ho = potencial_armonico(x, D_e, a, x_e, desplazado=True, masa=masa)
    
    # MORSE
    energias_morse, psi_morse = resolver_schrodinger_1d(x, V_morse, masa=masa, hbar=hbar)
    energias_morse_lig, psi_morse_lig = estados_ligados(energias_morse, psi_morse, umbral=0)
    
    # HO recortado a niveles de Morse (ligados)
    energias_ho, psi_ho = resolver_schrodinger_1d(x, V_ho, masa=masa, hbar=hbar)
    n_ligados = len(energias_morse_lig)
    energias_ho = energias_ho[:n_ligados]
    psi_ho = psi_ho[:, :n_ligados]

    
    # Espectro IR
    
    E_malla = np.linspace(0.5, 4.5, 1000)
    
    # Calcular espectros
    espectro_morse = generar_espectro_IR(x, energias_morse_lig, psi_morse_lig, E_malla, gamma=gamma)
    espectro_ho = generar_espectro_IR(x, energias_ho, psi_ho, E_malla, gamma=gamma)
    
    # Para que el maximo sea siempre =1
    espectro_morse = espectro_morse / np.max(espectro_morse)
    espectro_ho = espectro_ho / np.max(espectro_ho)

    
    # Gráficos Comparativos
    
    plt.figure(figsize=(10, 6))
    
    # Graficar espectro del Oscilador Armónico
    plt.plot(E_malla, espectro_ho, '--', linewidth=2, color='orange', label="Oscilador Armónico")
    
    # Graficar espectro del Potencial de Morse
    plt.plot(E_malla, espectro_morse, '-', linewidth=2, color='blue', label="Potencial de Morse")
    
    plt.title("Espectro Infrarrojo Simulado (Regla de selección $\Delta v = +1$)")
    plt.xlabel("Energía de Transición $\Delta E$")
    plt.ylabel("Intensidad")
    plt.legend()
    plt.grid(True)
    plt.xlim(0.5, 4.5)
    
    
    plt.tight_layout()
    plt.show()