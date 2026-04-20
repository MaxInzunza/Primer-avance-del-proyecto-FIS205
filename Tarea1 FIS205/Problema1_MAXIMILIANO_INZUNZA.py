############################################################################################
# Inciso A y B

import numpy as np

# MATRICES DE PAULI

sx = np.array([[0, 1],
               [1, 0]], dtype=complex)

sy = np.array([[0, -1j],
               [1j, 0]], dtype=complex)

sz = np.array([[1, 0],
               [0, -1]], dtype=complex)

I2 = np.eye(2, dtype=complex)


# FUNCION: dimension del espacio de Hilbert
def dimension_hilbert(N):
    return 2**N

# FUNCION: producto de Kronecker de una lista de operadores
def kron_n(lista_operadores):

    resultado = lista_operadores[0]
    for op in lista_operadores[1:]:
        resultado = np.kron(resultado, op)
    return resultado



# FUNCION: operador de un solo sitio
def operador_un_sitio(op, sitio, N):

    operadores = []
    for i in range(N):
        operadores.append(I2.copy())

    operadores[sitio] = op
    return kron_n(operadores)



# FUNCION: operador de dos sitios
def operador_dos_sitios(op1, sitio1, op2, sitio2, N):
   
    operadores = []
    for _ in range(N):
        operadores.append(I2.copy())

    operadores[sitio1] = op1
    operadores[sitio2] = op2
    return kron_n(operadores)



# FUNCION: Hamiltoniano de Ising transversal
def hamiltoniano_ising_transversal(N, J, B):
    
    dim = dimension_hilbert(N)
    H = np.zeros((dim, dim), dtype=complex)

    # Termino de interaccion entre vecinos:
    for i in range(N - 1):
        H += J * operador_dos_sitios(sx, i, sx, i + 1, N)

    # Termino de campo transversal: 
    for i in range(N):
        H += B * operador_un_sitio(sz, i, N)

    return H


N = 2
J = 1.0
B = 0.5

dim = dimension_hilbert(N)
H = hamiltoniano_ising_transversal(N, J, B)

print(f"N = {N}")
print(f"Dimension del espacio de Hilbert = {dim}")
print(f"Forma de H = {H.shape}")
print("Hamiltoniano:")
print(H)


############################################################################################
# Inciso C

import matplotlib.pyplot as plt
from scipy.linalg import expm

# Estado base |down>
ket_down = np.array([0, 1], dtype=complex)


def estado_inicial_down(N):
    return kron_n([ket_down.copy() for i in range(N)])


def evolucion_probabilidad_retorno(N, J, B, dt, n_pasos):
    H = hamiltoniano_ising_transversal(N, J, B)
    psi0 = estado_inicial_down(N)

    U = expm(-1j * H * dt)

    tiempos = np.arange(n_pasos + 1) * dt
    pr = np.zeros(n_pasos + 1)

    psi = psi0.copy()

    for k in range(n_pasos + 1):
        amp = np.vdot(psi0, psi)
        pr[k] = np.abs(amp)**2

        if k < n_pasos:
            psi = U @ psi

    return tiempos, pr


def graficar_probabilidad_retorno(N=4, J=1.0, dt=0.05, n_pasos=400):
    razones = [0.01, 1, 100]
    etiquetas = [r"$B/J = 0.01$", r"$B/J = 1$", r"$B/J = 100$"]

    plt.figure(figsize=(9, 5))

    for razon, etiqueta in zip(razones, etiquetas):
        B = razon * J
        tiempos, p = evolucion_probabilidad_retorno(N, J, B, dt, n_pasos)
        plt.plot(tiempos, p, label=etiqueta)

    plt.xlabel("t")
    plt.ylabel(r"$p(t)=|\langle \Psi(t)|\Psi(0)\rangle|^2$")
    plt.title(f"Probabilidad de retorno para el modelo de Ising transversal (N={N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


graficar_probabilidad_retorno(N=4, J=1.0, dt=0.05, n_pasos=400)


############################################################################################
# Inciso D

from time import perf_counter

def medir_tiempos_hamiltoniano(N_valores, J=1.0, B=1.0, repeticiones=3):
    tiempos_construccion = []
    tiempos_diagonalizacion = []

    for N in N_valores:
        tiempos_c = []
        tiempos_d = []

        for i in range(repeticiones):
            t0 = perf_counter()
            H = hamiltoniano_ising_transversal(N, J, B)
            t1 = perf_counter()

            np.linalg.eigh(H)
            t2 = perf_counter()

            tiempos_c.append(t1 - t0)
            tiempos_d.append(t2 - t1)

        tiempos_construccion.append(np.mean(tiempos_c))
        tiempos_diagonalizacion.append(np.mean(tiempos_d))

    return np.array(tiempos_construccion), np.array(tiempos_diagonalizacion)

N_valores = np.array([4, 5, 6, 7, 8])

tiempos_construccion, tiempos_diagonalizacion = medir_tiempos_hamiltoniano(N_valores, J=1.0, B=1.0, repeticiones=3)

for N, tc, td in zip(N_valores, tiempos_construccion, tiempos_diagonalizacion):
    print(f"N = {N} | construcción = {tc:.6f} s | diagonalización = {td:.6f} s")

############################################################################################
# Inciso E

def graficar_tiempos_hamiltoniano(N_valores, tiempos_construccion, tiempos_diagonalizacion):
    plt.figure(figsize=(9, 5))
    plt.plot(N_valores, tiempos_construccion, "o-", label="Construcción de H")
    plt.plot(N_valores, tiempos_diagonalizacion, "s-", label="Diagonalización de H")
    plt.xlabel("N")
    plt.ylabel("Tiempo (s)")
    plt.title("Tiempo de cálculo para construir y diagonalizar el Hamiltoniano")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

graficar_tiempos_hamiltoniano(N_valores, tiempos_construccion, tiempos_diagonalizacion)


############################################################################################
# Inciso F

# t(N) = A*exp(b*N)

def estimar_tiempos(N_valores, tiempos_construccion, tiempos_diagonalizacion, N_objetivo):
    tiempos_totales = tiempos_construccion + tiempos_diagonalizacion

    coef_c = np.polyfit(N_valores, np.log(tiempos_construccion), 1) 
    coef_d = np.polyfit(N_valores, np.log(tiempos_diagonalizacion), 1)
    coef_t = np.polyfit(N_valores, np.log(tiempos_totales), 1)

    b_c, logA_c = coef_c
    b_d, logA_d = coef_d
    b_t, logA_t = coef_t

    A_c = np.exp(logA_c)
    A_d = np.exp(logA_d)
    A_t = np.exp(logA_t)

    N_objetivo = np.array(N_objetivo)

    est_construccion = A_c * np.exp(b_c * N_objetivo)
    est_diagonalizacion = A_d * np.exp(b_d * N_objetivo)
    est_total = A_t * np.exp(b_t * N_objetivo)

    return est_construccion, est_diagonalizacion, est_total


def convertir_tiempo(tiempo):
    minuto = 60
    hora = 60 * minuto
    dia = 24 * hora
    anio = 365 * dia

    if tiempo < minuto:
        return f"{tiempo:.2f} s"
    elif tiempo < hora:
        return f"{tiempo/minuto:.2f} min"
    elif tiempo < dia:
        return f"{tiempo/hora:.2f} h"
    elif tiempo < anio:
        return f"{tiempo/dia:.2f} días"
    else:
        return f"{tiempo/anio:.2e} años"


N_objetivo = np.array([20, 50, 100])

est_construccion, est_diagonalizacion, est_total = estimar_tiempos(N_valores, tiempos_construccion, tiempos_diagonalizacion, N_objetivo)

for N, tc, td, tt in zip(N_objetivo, est_construccion, est_diagonalizacion, est_total):
    print(f"N = {N}")
    print(f"  Construcción estimada (s)   = {tc:.6e} s")
    print(f"  Diagonalización estimada (s)= {td:.6e} s")
    print(f"  Tiempo total estimado (s)   = {tt:.6e} s")
    print(f"  Tiempo total aproximado  = {convertir_tiempo(tt)}")
    print()

############################################################################################
# Inciso G

edad_universo = 4.3e17  

for N, tt in zip(N_objetivo, est_total):
    razon = tt / edad_universo

    print(f"Comparacion para N = {N}")
    print(f"  Tiempo total estimado = {tt:.6e} s")
    print(f"  Tiempo aproximado     = {convertir_tiempo(tt)}")
    print(f"  Edad del universo     = {edad_universo:.6e} s")
    print(f"  Razón t / edad_universo = {razon:.6e}")

    if tt < edad_universo:
        print("  -> La simulación tomaría menos que la edad del universo.")
    elif tt > edad_universo:
        print("  -> La simulación tomaría más que la edad del universo.")
    else:
        print("  -> La simulación tomaría aproximadamente la edad del universo.")

    print()