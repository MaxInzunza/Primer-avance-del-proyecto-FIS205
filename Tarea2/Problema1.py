import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# PROBLEMA 1, TAREA 2 FIS205 !!!!!!


# ---> INCISO B 

m = 1
x0 = 1
v0 = 0

N = 3000
Nt = 1000
t_ini = 0
t_fin = 10
sigma = 0.02

t = np.linspace(t_ini, t_fin, Nt)

def oscilador_amortiguado(t, y, gamma, k):
    x = y[0]
    v = y[1]

    dxdt = v
    dvdt = -gamma * v - k * x

    return [dxdt, dvdt]

def generar_senal(gamma, k, t):
    y0 = [x0, v0]

    sol = solve_ivp(oscilador_amortiguado, [t[0], t[-1]], y0, t_eval=t, args=(gamma, k))

    x = sol.y[0]
    return x

X = []
Y = []

for i in range(N):
    gamma = np.random.uniform(0.05, 1)
    k = np.random.uniform(1, 5)

    x = generar_senal(gamma, k, t)

    ruido = np.random.normal(0, sigma, Nt)
    x_obs = x + ruido

    X.append(x_obs)
    Y.append([gamma, k])

X = np.array(X)
Y = np.array(Y)

print("Forma de X:", X.shape)
print("Forma de Y:", Y.shape)

plt.figure(figsize=(10, 6))

for i in range(5):
    plt.plot(t, X[i], label=f"$\gamma$={Y[i,0]:.2f}, k={Y[i,1]:.2f}")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Algunas señales simuladas del oscilador amortiguado")
plt.legend()
plt.grid(True)
plt.show()


# ---> INCISO C

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)


# RMSE = sqrt(MSE), obteniendo RMSE para gamma y k por separado
def calcular_rmse(y_real, y_pred):
    rmse_gamma = np.sqrt(mean_squared_error(y_real[:, 0], y_pred[:, 0]))
    rmse_k = np.sqrt(mean_squared_error(y_real[:, 1], y_pred[:, 1]))
    return rmse_gamma, rmse_k

# Modelo: Random Forest
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, Y_train)
Y_pred_rf = modelo_rf.predict(X_test)

rmse_gamma_rf, rmse_k_rf = calcular_rmse(Y_test, Y_pred_rf)

# Modelo: MLPRegressor
scaler_X = StandardScaler()
X_trainS = scaler_X.fit_transform(X_train)
X_testS = scaler_X.transform(X_test)



scaler_Y = StandardScaler()
Y_trainS = scaler_Y.fit_transform(Y_train)

modelo_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
modelo_mlp.fit(X_trainS, Y_trainS)
Y_pred_mlp_scaled = modelo_mlp.predict(X_testS)

Y_pred_mlp = scaler_Y.inverse_transform(Y_pred_mlp_scaled)
# ----------------------

rmse_gamma_mlp, rmse_k_mlp = calcular_rmse(Y_test, Y_pred_mlp)

print("Resultados finales")
print("------------------")
print("Random Forest")
print("RMSE_gamma =", rmse_gamma_rf)
print("RMSE_k     =", rmse_k_rf)

print()

print("MLPRegressor")
print("RMSE_gamma =", rmse_gamma_mlp)
print("RMSE_k     =", rmse_k_mlp)

"""""
Resultados finales
------------------
Random Forest
RMSE_gamma = 0.016547794016972088
RMSE_k     = 0.048812814656722645

MLPRegressor
RMSE_gamma = 0.11802346212054216
RMSE_k     = 0.13972084487113895

"""""

# ---> Inciso D 

# Igual que antes solo que ahora aplicado a distintos sigmas dentro deu n for

sigmas = [0, 0.01, 0.02, 0.05, 0.1]

X_limpio = []
Y_base = []

for i in range(N):
    gamma = np.random.uniform(0.05, 1)
    k = np.random.uniform(1, 5)

    x = generar_senal(gamma, k, t)

    X_limpio.append(x)
    Y_base.append([gamma, k])

X_limpio = np.array(X_limpio)
Y_base = np.array(Y_base)

print("Forma de X_limpio:", X_limpio.shape)
print("Forma de Y_base  :", Y_base.shape)

# Pa guardar los datos (RF y MLP)

rmse_gamma_rf_train = []
rmse_k_rf_train = []
rmse_gamma_rf_test = []
rmse_k_rf_test = []

rmse_gamma_mlp_train = []
rmse_k_mlp_train = []
rmse_gamma_mlp_test = []
rmse_k_mlp_test = []


for sigmaR in sigmas:
    print("Usando sigma =", sigmaR)

    ruido = np.random.normal(0, sigmaR, X_limpio.shape)
    X_ruido = X_limpio + ruido

    X_train, X_test, Y_train, Y_test = train_test_split(X_ruido, Y_base, test_size=0.2, random_state=42)

   
    # Modelo: Random Forest
    
    modelo_rf = RandomForestRegressor( n_estimators=100, random_state=42)

    modelo_rf.fit(X_train, Y_train)

    Y_pred_rf_train = modelo_rf.predict(X_train)
    Y_pred_rf_test = modelo_rf.predict(X_test)

    rmse_gamma_train, rmse_k_train = calcular_rmse(Y_train, Y_pred_rf_train)
    rmse_gamma_test, rmse_k_test = calcular_rmse(Y_test, Y_pred_rf_test)

    rmse_gamma_rf_train.append(rmse_gamma_train)
    rmse_k_rf_train.append(rmse_k_train)
    rmse_gamma_rf_test.append(rmse_gamma_test)
    rmse_k_rf_test.append(rmse_k_test)

    
    # Modelo: MLPRegressor
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)

    modelo_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    modelo_mlp.fit(X_train_scaled, Y_train_scaled)

    Y_pred_mlp_train_scaled = modelo_mlp.predict(X_train_scaled)
    Y_pred_mlp_test_scaled = modelo_mlp.predict(X_test_scaled)

    Y_pred_mlp_train = scaler_Y.inverse_transform(Y_pred_mlp_train_scaled)
    Y_pred_mlp_test = scaler_Y.inverse_transform(Y_pred_mlp_test_scaled)
    

    rmse_gamma_train, rmse_k_train = calcular_rmse(Y_train, Y_pred_mlp_train)
    rmse_gamma_test, rmse_k_test = calcular_rmse(Y_test, Y_pred_mlp_test)

    rmse_gamma_mlp_train.append(rmse_gamma_train)
    rmse_k_mlp_train.append(rmse_k_train)
    rmse_gamma_mlp_test.append(rmse_gamma_test)
    rmse_k_mlp_test.append(rmse_k_test)


# Graficos RMSE de gamma y k en funcion de los sigmas

plt.figure(figsize=(8, 5))
plt.plot(sigmas, rmse_gamma_rf_test, "o-", label="Random Forest")
plt.plot(sigmas, rmse_gamma_mlp_test, "s-", label="MLPRegressor")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$RMSE_\gamma$")
plt.title(r"RMSE$_\gamma$ en función del ruido")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(sigmas, rmse_k_rf_test, "o-", label="Random Forest")
plt.plot(sigmas, rmse_k_mlp_test, "s-", label="MLPRegressor")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$RMSE_k$")
plt.title(r"RMSE$_k$ en función del ruido")
plt.grid(True)
plt.legend()
plt.show()