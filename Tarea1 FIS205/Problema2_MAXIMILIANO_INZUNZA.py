############################################################################################
# INCISO A
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def generar_senal(N, fs, f1, f2):
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    return t, x


N = 1000
fs = 1000
f1 = 50
f2 = 120

t, x = generar_senal(N, fs, f1, f2)


""""
# Grafico de la señal discreta

plt.figure(figsize=(9, 4))
plt.plot(t, x)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Señal compuesta por dos frecuencias")
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# INCISO B

def dft_directa(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        suma = 0j
        for n in range(N):
            suma += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
        X[k] = suma

    return X

X = dft_directa(x)

# INCISO C

modulo_X = np.abs(X)
k = np.arange(len(X))

plt.figure(figsize=(9, 4))
plt.plot(k, modulo_X)
plt.xlabel("k")
plt.ylabel(r"$|X_k|$")
plt.title("Espectro de la señal")
plt.grid(True)
plt.tight_layout()
plt.show()

# INCISO D

X_fft = np.fft.fft(x)
print(X_fft)

# INCISO E

def medir_tiempos_dft_fft(N_valor, fs=1000, f1=50, f2=120):
    tiempos_dft = []
    tiempos_fft = []

    for N in N_valor:
        t, x = generar_senal(N, fs, f1, f2)

        t0 = perf_counter()
        X_dft = dft_directa(x)
        t1 = perf_counter()

        t2 = perf_counter()
        X_fft = np.fft.fft(x)
        t3 = perf_counter()

        tiempos_dft.append(t1 - t0)
        tiempos_fft.append(t3 - t2)

        print(f"N = {N}")
        print(f"  Tiempo DFT directa = {t1 - t0:.6e} s")
        print(f"  Tiempo FFT         = {t3 - t2:.6e} s")
        print()

    return np.array(tiempos_dft), np.array(tiempos_fft)

# N_valores = np.array([10**2, 10**3, 10**4, 10**5])
N_valores = np.array([10**2, 10**3]) # Lo dejé hasta 10**3 solo para poder correr el codigo rapido

tiempos_dft, tiempos_fft = medir_tiempos_dft_fft(N_valores)


"""
Estos tiempos me arrojó cuando hice correr el codigo hasta 10**5

N = 100
  Tiempo DFT directa = 6.961400e-03 s
  Tiempo FFT         = 1.138000e-04 s

N = 1000
  Tiempo DFT directa = 6.634182e-01 s
  Tiempo FFT         = 8.490000e-05 s

N = 10000
  Tiempo DFT directa = 6.695099e+01 s
  Tiempo FFT         = 3.776000e-04 s

N = 100000
  Tiempo DFT directa = 6.765021e+03 s -----> se demoro casi 2 horas en terminar
  Tiempo FFT         = 4.518400e-03 s
"""

# INCISO F

plt.figure(figsize=(9, 5))
plt.plot(N_valores, tiempos_dft, "o-", label="DFT directa")
plt.plot(N_valores, tiempos_fft, "s-", label="FFT")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("Tiempo de ejecución (s)")
plt.title("Comparación de tiempos: DFT directa vs FFT")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# INCISO G

# Usando los valores que obtuve la vez que corri el codigo hasta 10**5

N_val = np.array([10**2, 10**3, 10**4, 10**5])

tiempos_dft_g = np.array([6.961400e-03, 6.634182e-01, 6.695099e+01, 6.765021e+03])

tiempos_fft_g = np.array([1.138000e-04, 8.490000e-05, 3.776000e-04, 4.518400e-03])


def estimar_exponente(N, tiempos):
    coef = np.polyfit(np.log10(N), np.log10(tiempos), 1)
    pendiente = coef[0]
    intercepto = coef[1]
    return pendiente, intercepto


p_dft, b_dft = estimar_exponente(N_val, tiempos_dft_g)
p_fft, b_fft = estimar_exponente(N_val, tiempos_fft_g)

print(f"Exponente experimental DFT directa: {p_dft:.4f}")
print(f"Exponente experimental FFT: {p_fft:.4f}")

N_fit = np.linspace(N_val.min(), N_val.max(), 300)

tiempo_fit_dft = 10**b_dft * N_fit**p_dft
tiempo_fit_fft = 10**b_fft * N_fit**p_fft

plt.figure(figsize=(9, 5))
plt.loglog(N_val, tiempos_dft_g, "o", label="DFT directa (datos)")
plt.loglog(N_fit, tiempo_fit_dft, "-", label=fr"DFT ajuste: $t \propto N^{{{p_dft:.2f}}}$")

plt.loglog(N_val, tiempos_fft_g, "s", label="FFT (datos)")
plt.loglog(N_fit, tiempo_fit_fft, "-", label=fr"FFT ajuste: $t \propto N^{{{p_fft:.2f}}}$")

plt.xlabel("N")
plt.ylabel("Tiempo de ejecución (s)")
plt.title("Gráfico log-log y exponente de escalamiento")
plt.grid(True, which="major", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# INCISO H

def tiempo_dft_ajustado(N):
    return 10**b_dft * N**p_dft

def tiempo_fft_ajustado(N):
    return 10**b_fft * N**p_fft

N_busqueda = np.arange(1, 1000000)
razon = tiempo_dft_ajustado(N_busqueda) / tiempo_fft_ajustado(N_busqueda)

N_100 = N_busqueda[np.where(razon >= 100)[0][0]]

print("")
print(f"Valor aproximado de N para que la FFT sea al menos 100 veces más rápida: N ≈ {N_100}")
print(f"Razón en ese punto = {tiempo_dft_ajustado(N_100) / tiempo_fft_ajustado(N_100):.2f}")