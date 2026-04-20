import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constantes
k_B = 1.380649e-23          # J/K
m_H2 = 3.32e-27             # kg
epsilon = 33.3 * k_B        # J
sigma = 0.296e-9            # m

# Parámetros iniciales de la simulación
N = 125
Lx = 10e-6                  # m
Ly = 10e-6                  # m
Lz = 10e-6                  # m
T0 = 300.0                  # K
dt = 1e-12                  # s
n_steps = 2000
seed = 12345

def inicializar_posiciones(N, Lx, Ly, Lz, perturbacion=0.10, seed=None):
    rng = np.random.default_rng(seed)

    n_lado = int(np.ceil(N ** (1 / 3)))

    # Tomando el punto central de una celda de longitud L_i / n_lado para evitar los bordes
    xs = np.linspace(Lx / (2 * n_lado), Lx - Lx / (2 * n_lado), n_lado)
    ys = np.linspace(Ly / (2 * n_lado), Ly - Ly / (2 * n_lado), n_lado)
    zs = np.linspace(Lz / (2 * n_lado), Lz - Lz / (2 * n_lado), n_lado)

    # Combinaciones dentro de toda la red
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]

    dx = Lx / n_lado
    dy = Ly / n_lado
    dz = Lz / n_lado

    # Ajuste automatico para la perturbacion dependiendo del ancho de las celdas
    delta = perturbacion * min(dx, dy, dz)

    # Valores random entre +-delta para todas las particulas
    pos += rng.uniform(-delta, delta, size=pos.shape)

    # Manteniendo las particulas en el rango permitido (entre 0 y L_i)
    pos[:, 0] = np.clip(pos[:, 0], 0, Lx)
    pos[:, 1] = np.clip(pos[:, 1], 0, Ly)
    pos[:, 2] = np.clip(pos[:, 2], 0, Lz)

    return pos

def temperatura_desde_velocidades(vel, m=m_H2, k_B=k_B):
    N_local = vel.shape[0]
    return m * np.sum(vel**2) / (3 * N_local * k_B)


def inicializar_velocidades(N, T, m=m_H2, k_B=k_B, seed=None):
    rng = np.random.default_rng(seed)

    sigma_v = np.sqrt(k_B * T / m)
    vel = rng.normal(0, sigma_v, size=(N, 3))

    vel -= np.mean(vel, axis=0)

    T_inst = temperatura_desde_velocidades(vel, m, k_B)
    if T_inst > 0:
        vel *= np.sqrt(T / T_inst)

    return vel

def paso_gas_ideal(pos, vel, dt, Lx, Ly, Lz):
    pos_new = pos + vel * dt
    vel_new = vel.copy()

    for eje, L in enumerate([Lx, Ly, Lz]):
        mask_izq = pos_new[:, eje] < 0
        #devolviendo la particula para adentro de la caja la misma distancia que se alejó de la pared de la caja
        pos_new[mask_izq, eje] = -pos_new[mask_izq, eje]
        vel_new[mask_izq, eje] *= -1

        mask_der = pos_new[:, eje] > L
        pos_new[mask_der, eje] = 2 * L - pos_new[mask_der, eje]
        vel_new[mask_der, eje] *= -1

    return pos_new, vel_new

##################################################################################

def energia_cinetica(vel, m=m_H2):
    return 0.5 * m * np.sum(vel**2)


def presion_ideal(N, T, Lx, Ly, Lz, k_B=k_B):
    V = Lx * Ly * Lz
    return N * k_B * T / V


def rapidez(vel):
    return np.linalg.norm(vel, axis=1)


def maxwell_boltzmann_3d(v, T, m=m_H2, k_B=k_B):
    pref = 4 * np.pi * (m / (2 * np.pi * k_B * T))**1.5
    return pref * v**2 * np.exp(-m * v**2 / (2 * k_B * T))


def ejecutar_simulacion_gas_ideal(N, Lx, Ly, Lz, T0, dt, n_steps, seed=None):
    # tomamos caso con distintas seeds
    if seed is None:  
        seed_pos = None
        seed_vel = None

    else:
        seed_pos = seed
        seed_vel = seed +1 

    pos = inicializar_posiciones(N, Lx, Ly, Lz, seed=seed_pos)
    vel = inicializar_velocidades(N, T0, seed=seed_vel)

    tiempos = np.arange(n_steps) * dt
    T_hist = np.zeros(n_steps)
    P_hist = np.zeros(n_steps)
    Ec_hist = np.zeros(n_steps)
    Ep_hist = np.zeros(n_steps)

    ultimas_rapideces = []

    for i in range(n_steps):
        pos, vel = paso_gas_ideal(pos, vel, dt, Lx, Ly, Lz)

        T_inst = temperatura_desde_velocidades(vel)
        Ec_inst = energia_cinetica(vel)
        P_inst = presion_ideal(N, T_inst, Lx, Ly, Lz)

        T_hist[i] = T_inst
        P_hist[i] = P_inst
        Ec_hist[i] = Ec_inst
        Ep_hist[i] = 0

        if i >= max(0, n_steps - 1000):
            ultimas_rapideces.extend(rapidez(vel))

    return {"pos": pos, "vel": vel, "t": tiempos, "T": T_hist, "P": P_hist, "Ec": Ec_hist, "Ep": Ep_hist, "rapideces": np.array(ultimas_rapideces), "Lx": Lx, "Ly": Ly, "Lz": Lz}


fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(bottom=0.30)

ax_slider_T = plt.axes([0.15, 0.20, 0.70, 0.03])
ax_slider_Lx = plt.axes([0.15, 0.15, 0.70, 0.03])
ax_slider_Ly = plt.axes([0.15, 0.10, 0.70, 0.03])
ax_slider_Lz = plt.axes([0.15, 0.05, 0.70, 0.03])

slider_T = Slider(ax_slider_T, "T (K)", 50, 600, valinit=T0)
slider_Lx = Slider(ax_slider_Lx, "Lx (um)", 5, 30, valinit=Lx * 1e6)
slider_Ly = Slider(ax_slider_Ly, "Ly (um)", 5, 30, valinit=Ly * 1e6)
slider_Lz = Slider(ax_slider_Lz, "Lz (um)", 5, 30, valinit=Lz * 1e6)



# Genera graficos a a partir de los sliders usados
def redibujar(T_val, Lx_um, Ly_um, Lz_um):
    resultado = ejecutar_simulacion_gas_ideal(N=N, Lx=Lx_um * 1e-6, Ly=Ly_um * 1e-6, Lz=Lz_um * 1e-6, T0=T_val, dt=dt, n_steps=n_steps, seed=seed)

    t_ps = resultado["t"] * 1e12
    T_hist = resultado["T"]
    P_hist = resultado["P"]
    Ec_hist = resultado["Ec"]
    Ep_hist = resultado["Ep"]
    v_data = resultado["rapideces"]

    T_ref = np.mean(T_hist[-min(1000, len(T_hist)):])

    for ax in axs.ravel():
        ax.clear()

    axs[0, 0].plot(t_ps, T_hist)
    axs[0, 0].set_title("Temperatura T(t)")
    axs[0, 0].set_xlabel("t (ps)")
    axs[0, 0].set_ylabel("T (K)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t_ps, P_hist)
    axs[0, 1].set_title("Presión P(t)")
    axs[0, 1].set_xlabel("t (ps)")
    axs[0, 1].set_ylabel("P (Pa)")
    axs[0, 1].grid(True)

    axs[1, 0].plot(t_ps, Ec_hist, label="Energía cinética")
    axs[1, 0].plot(t_ps, Ep_hist, label="Energía potencial")
    axs[1, 0].set_title("Energía")
    axs[1, 0].set_xlabel("t (ps)")
    axs[1, 0].set_ylabel("E (J)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].hist(v_data, bins=35, density=True, alpha=0.7, label="Simulación")
    v_grid = np.linspace(0, max(v_data) * 1.1, 300)
    axs[1, 1].plot(v_grid, maxwell_boltzmann_3d(v_grid, T_ref), label="Maxwell-Boltzmann")
    axs[1, 1].set_title("Distribución de velocidades")
    axs[1, 1].set_xlabel("Rapidez (m/s)")
    axs[1, 1].set_ylabel("Densidad")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.canvas.draw_idle()


# para actualizar graficos, llama a redibujar y genera graficos nuevos
def actualizar(val):
    redibujar(slider_T.val, slider_Lx.val, slider_Ly.val, slider_Lz.val)

# Actualiza todo

slider_T.on_changed(actualizar)
slider_Lx.on_changed(actualizar)
slider_Ly.on_changed(actualizar)
slider_Lz.on_changed(actualizar)

redibujar(T0, Lx * 1e6, Ly * 1e6, Lz * 1e6)
plt.show()