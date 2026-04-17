import numpy as np
import matplotlib.pyplot as plt

from potenciales import (potencial_morse, potencial_armonico, energias_armonicas_analiticas)

from schrodinger import resolver_schrodinger_1d, estados_ligados


# =====================================
# Parámetros
# =====================================

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


# =====================================
# Construcción de los potenciales
# =====================================

V_morse = potencial_morse(x, D_e, a, x_e, desplazado=True)
V_ho = potencial_armonico(x, D_e, a, x_e, desplazado=True, masa=masa)


# =====================================
# Resolver Schrodinger para Morse y Energías
# =====================================

energias_morse, psi_morse = resolver_schrodinger_1d(x, V_morse, masa=masa, hbar=hbar)
energias_morse_lig, psi_morse_lig = estados_ligados(energias_morse, psi_morse, umbral=0.0)


# =====================================
# Resolver Schrodinger para armónico
# =====================================

energias_ho_num, psi_ho_num = resolver_schrodinger_1d(x, V_ho, masa=masa, hbar=hbar)


# =====================================
# Niveles energeticos Morse vs HO
# =====================================

n_comp = min(4, len(energias_morse_lig))
energias_ho_ana = energias_armonicas_analiticas(n_comp - 1, D_e, a, masa=masa, hbar=hbar, desplazado=True)


# =====================================
# Comparación numérica entre niveles 
# =====================================
print("=" * 50)
print("COMPARACIÓN MORSE vs OSCILADOR ARMÓNICO")
print("=" * 50)
print()

print("Primeros niveles:")
print("-" * 70)
print(" n   E_Morse(num)    E_HO(num)      E_HO(analítico)")
print("-" * 70)

for n in range(n_comp):
    print(f"{n:2d}   {energias_morse_lig[n]:12.6f}   {energias_ho_num[n]:12.6f}   {energias_ho_ana[n]:16.6f}")

print()
print("Espaciamientos entre niveles:")
print("-" * 80)

for n in range(n_comp - 1):
    dE_morse = energias_morse_lig[n + 1] - energias_morse_lig[n]
    dE_ho = energias_ho_ana[n + 1] - energias_ho_ana[n]

    print(f"Delta E Morse ({n}->{n+1}) = {dE_morse:10.6f}    |    Delta E HO ({n}->{n+1}) = {dE_ho:10.6f}")


# =====================================
# Gráfico Morse vs HO
# =====================================

plt.figure(figsize=(9, 6))
plt.plot(x, V_morse, label="Potencial de Morse", linewidth=2)
plt.plot(x, V_ho, "--", label="Oscilador armónico equivalente", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Límite de disociación (Morse)")
plt.xlim(-1.2, 4.5)
plt.ylim(-9, 6)
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Comparación de potenciales: Morse vs armónico")
plt.legend()
plt.grid(True)
plt.show()


# =====================================
# ráfico niveles de energía Morse vs HO
# =====================================

plt.figure(figsize=(8, 6))

for n in range(n_comp):
    # Niveles de Morse
    plt.hlines(energias_morse_lig[n], 0.8, 1.2, colors="blue", linewidth=2)

    # Niveles armónicos analíticos
    plt.hlines(energias_ho_ana[n], 1.8, 2.2, colors="red", linewidth=2)

plt.xlim(0.4, 2.6)
plt.ylim(-9, 6)
plt.xticks([1, 2], ["Morse", "Armónico"])
plt.ylabel("Energía")
plt.title("Comparación de niveles de energía")
plt.grid(True, axis="y")
plt.show()


# =====================================
# Gráfico autofunciones Morse vs HO
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

escala = 0.5

# --- Para potencial de Morse ---

axes[0].plot(x, V_morse, color="black", linewidth=2, label="Morse")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)

for n in range(n_comp):
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

# --- Para el HO ---

axes[1].plot(x, V_ho, color="black", linewidth=2, label="Armónico")

for n in range(n_comp):
    E = energias_ho_num[n]
    psi_n = psi_ho_num[:, n]

    axes[1].hlines(E, x[0], x[-1], colors="gray", linestyles=":", linewidth=1)
    axes[1].plot(x, E + escala * psi_n, label=f"n={n}")

axes[1].set_title("Estados estacionarios en el armónico")
axes[1].set_xlabel("x")
axes[1].set_xlim(-1.2, 4.5)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
