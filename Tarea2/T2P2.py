import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# INCISO B

# Constantes
K = 0.307075
Z_A = 0.55509
I = 75e-6
me = 0.511
mp = 938.272
z = 1
rho = 1

# Bethe-Bloch
def sp(E):
    gamma = 1 + E / mp
    beta_sq = 1 - 1 / gamma**2

    Tmax = 2 * me * beta_sq * gamma**2 / (1 + 2 * gamma * me / mp + (me / mp)**2)

    L = 0.5 * np.log(2*me * beta_sq * gamma**2 * Tmax / I**2) - beta_sq

    return K * z**2 * Z_A * (1 / beta_sq) * L * rho

# Calculo del rango CSDA
def integr(E):
    return 1 / sp(E)

def rango(E0):
    resultado = integrate.quad(integr, 1, E0)

    return resultado[0]

energias = [50, 150, 250]
nist = [2.227, 15.77, 37.94]

print("E (MeV)   Rango calc (cm)   NIST (cm)")
for E, r in zip(energias, nist):
    Rc = rango(E)
    print(E, "      ", round(Rc, 3), "          ", r)

""""
E (MeV)   Rango calc (cm)   NIST (cm)
50        2.22            2.227
150        15.765            15.77
250        37.929            37.94
"""""

# Grafica del poder de frenado
E = np.linspace(1, 250, 500)
S = sp(E)

plt.plot(E, S)
plt.xlabel("Energia (MeV)")
plt.ylabel("Poder de frenado (MeV/cm)")
plt.title("Poder de frenado de protones en agua")
plt.grid(True)
plt.show()


# INCISO C


# PARAMETROS 
E_inicial = 150       
N_protones = 10000    
paso = 0.01           


x_max = 20            
n_bins = int(x_max/paso)
dosis = np.zeros(n_bins)

# SIMULACION DE CADA PROTON
for i in range(N_protones):

    E = E_inicial
    x = 0

    while E > 1 and x < x_max:

        dEdx = sp(E)
        perdida = dEdx * paso     # delta E = (-dE/dx) * delta x

        bin_actual = int(x/paso)
        if bin_actual < n_bins:
            dosis[bin_actual] +=  perdida

        E -=  perdida
        x +=  paso

# normalizar
dosis = dosis/N_protones

# EJE DE PROFUNDIDAD
profundidad = np.zeros(n_bins)
for j in range(n_bins):
    profundidad[j] = j * paso

# PARA EL P*** DE BRAGG (XDD)
indice_pico = np.argmax(dosis) 
posicion_pico = profundidad[indice_pico] 

print("Posicion del pico de Bragg:", posicion_pico, "cm")
print("Rango CSDA del inciso (b):  15.765 cm (referencia)")

# GRAFIC0
plt.figure(figsize=(8, 5))
plt.plot(profundidad, dosis, color="blue")
plt.axvline(posicion_pico, color="red", linestyle="--",
            label="Pico de Bragg = " + str(round(posicion_pico, 2)) + " cm")
plt.xlabel("Profundidad en agua (cm)")
plt.ylabel("Dosis depositada (MeV/cm)")
plt.title("Pico de Bragg - Protones de 150 MeV (sin fluctuacion)")
plt.legend()
plt.grid(True)
plt.show()



# INCISO D


# PARAMETROS (mismos q en el inciso c)
E_inicial = 150
N_protones = 10000
paso = 0.01
x_max = 20
n_bins = int(x_max/paso)

# DESVIACION DEL STRAGGLING 
def sigma_straggling(E, dx):
    gamma = 1 + E / mp

    beta_sq = 1 - 1 / gamma**2

    omega_sq = K * z**2 * Z_A * rho * dx
    sigma_sq = omega_sq * (1 - beta_sq / 2) / beta_sq

    return np.sqrt(sigma_sq)

# SIMULACION SIN FLUCTUACION 
dosis_sin = np.zeros(n_bins)
rangos_sin = []
for i in range(N_protones):
    E = E_inicial
    x = 0
    while E > 1 and x < x_max:
        perdida = sp(E) * paso
        bin_actual = int(x / paso)
        if bin_actual < n_bins:
            dosis_sin[bin_actual] +=  perdida
        E -= perdida
        x += paso
    rangos_sin.append(x)

dosis_sin = dosis_sin/N_protones

# SIMULACION CON FLUCTUACIÓN
dosis_con = np.zeros(n_bins)
rangos_con = []

for i in range(N_protones):
    E = E_inicial
    x = 0
    while E > 1 and x < x_max:
        perdida = sp(E) * paso

        # fluctuación gaussiana
        sigma = sigma_straggling(E, paso)
        perdida = perdida + np.random.normal(0, sigma)

        if perdida < 0:
            perdida = 0

        bin_actual = int(x / paso)
        if bin_actual < n_bins:
            dosis_con[bin_actual] += perdida

        E -=  perdida
        x += paso
    rangos_con.append(x)   #

dosis_con = dosis_con/N_protones

# PROFUNDIDAD
profundidad = np.zeros(n_bins)
for j in range(n_bins):
    profundidad[j] = j*paso

# ENSANCHAMIENTO (sigma_R)
sigma_R = np.std(rangos_con)
rango_medio = np.mean(rangos_con)

print("Rango medio (con straggling):", round(rango_medio, 3), "cm")
print("Ensanchamiento sigma_R:", round(sigma_R, 4), "cm")

# GRAFICA COMPARATIVA

plt.figure(figsize=(8, 5))
plt.plot(profundidad, dosis_sin, color="blue", label="Sin fluctuacion")
plt.plot(profundidad, dosis_con, color="red", label="Con straggling")
plt.xlabel("Profundidad en agua (cm)")
plt.ylabel("Dosis depositada (MeV/cm)")
plt.title("Pico de Bragg: con y sin straggling (150 MeV)")
plt.legend()
plt.grid(True)
plt.savefig("bragg_d.png")
plt.show()
