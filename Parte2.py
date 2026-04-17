import numpy as np
import matplotlib.pyplot as plt

from potenciales import potencial_morse, fuerza_morse, energia_clasica


def derivadas_morse(t, estado, D_e, a, x_e, masa=1.0):
    """
        dx/dt = p / m
        dp/dt = F(x)

    donde:
        F(x) = - dV/dx
    """
    x, p = estado

    dxdt = p / masa
    dpdt = fuerza_morse(x, D_e, a, x_e)

    return np.array([dxdt, dpdt], dtype=float)


def rk4_paso(func, t, estado, dt, D_e, a, x_e, masa=1):

    k1 = func(t, estado, D_e, a, x_e, masa)
    k2 = func(t + 0.5 * dt, estado + 0.5 * dt * k1, D_e, a, x_e, masa)
    k3 = func(t + 0.5 * dt, estado + 0.5 * dt * k2, D_e, a, x_e, masa)
    k4 = func(t + dt, estado + dt * k3, D_e, a, x_e, masa)

    return estado + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrar_rk4_morse(x0, p0, t_max, dt, D_e, a, x_e, masa=1):
    
    n_pasos = int(t_max / dt) + 1 #desde t0 a tn

    t = np.zeros(n_pasos)
    x = np.zeros(n_pasos)
    p = np.zeros(n_pasos)
    E = np.zeros(n_pasos)

    estado = np.array([x0, p0], dtype=float)

    x[0] = x0
    p[0] = p0
    E[0] = energia_clasica(x0, p0, D_e, a, x_e, masa=masa, desplazado=True)

    for i in range(1, n_pasos): #no contamos paso 0 
        t[i] = t[i - 1] + dt
        estado = rk4_paso(derivadas_morse, t[i - 1], estado, dt, D_e, a, x_e, masa)

        x[i] = estado[0]
        p[i] = estado[1]
        E[i] = energia_clasica(x[i], p[i], D_e, a, x_e, masa=masa, desplazado=True)

    return t, x, p, E


# ==========================================================
# Parametros
# ==========================================================

D_e = 8
a = 0.9
x_e = 0
masa = 1

# Condiciones iniciales
x0 = 1
p0 = 0

# Paso y tmax
dt = 0.01
t_max = 40


# ==========================================================
# RK4
# ==========================================================

t, x, p, E = integrar_rk4_morse(x0, p0, t_max, dt, D_e, a, x_e, masa=masa)


# ==========================================================
# Grafico: Potencial de morse y energía
# ==========================================================
x_pot = np.linspace(-1.5, 5, 1200)
V_pot = potencial_morse(x_pot, D_e, a, x_e, desplazado=True)

plt.figure(figsize=(9, 6))
plt.plot(x_pot, V_pot, label="Potencial de Morse", linewidth=2)
plt.axhline(E[0], color="red", linestyle="--", linewidth=1.5, label=f"Energía total = {E[0]:.3f}")
plt.axhline(0, color="black", linestyle=":", linewidth=1.2, label="Límite de disociación")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Pozo de Morse y energía clásica de la trayectoria")
plt.xlim(-1.2, 4.5)
plt.ylim(-9, 3)
plt.grid(True)
plt.legend()
plt.show()


# ==========================================================
# Grafico: x vs t
# ==========================================================
plt.figure(figsize=(9, 5))
plt.plot(t, x, linewidth=1.8)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Posición en función del tiempo")
plt.grid(True)
plt.show()


# ==========================================================
# Grafico: p vs t
# ==========================================================
plt.figure(figsize=(9, 5))
plt.plot(t, p, linewidth=1.8)
plt.xlabel("t")
plt.ylabel("p(t)")
plt.title("Momento en función del tiempo")
plt.grid(True)
plt.show()