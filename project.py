import numpy as np
from scipy.optimize import minimize
# from scipy.interpolate import interp1d
# from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('airfoil_lookup.csv')

def load_airfoil_data(alpha):
    """Retrieve Cl and Cd from airfoil lookup table for a given angle of attack."""
    nearest_idx = (df['Alpha'] - alpha).abs().idxmin()
    cl = df.loc[nearest_idx, 'Cl']
    cd = df.loc[nearest_idx, 'Cd']
    return cl, cd

def bem_solver(c, beta, r, V_inf, Omega, rho, B, R, dr):
    # Initialize induction factors
    a = 0.0
    a_prime = 0.0
    max_iterations = 1000
    tolerance = 1e-3

    def compute_a_and_a_prime(a, a_prime, c, beta, r, V_inf, Omega, B, R):
        # Flow angle
        phi = np.arctan2(V_inf * (1 - a), Omega * r * (1 + a_prime))
        # print(phi)
        # print(a_prime)
        # Relative velocity
        V_rel = np.sqrt(V_inf**2 * (1 - a)**2 + Omega**2 * r**2 * (1 + a_prime)**2)
        
        # Angle of attack
        alpha = np.degrees(phi - beta)
        
        # Get coefficients
        # print(alpha)
        C_L, C_D = load_airfoil_data(alpha)
        # print(C_L, C_D)
        # Local solidity
        sigma = (B * c) / (2 * np.pi * r)
        
        eps = 1e-6
        # Prandtl tip loss factor
        f = ((-B * (R - r + eps))/ (2 * r * np.sin(phi) + eps))
        exp_f = np.exp(f)

        exp_f = min(exp_f, 1.0 - eps)
        F = (2/np.pi) * np.arccos(exp_f)

        # print(((C_L * np.cos(phi) + C_D * np.sin(phi))))
        k_a = (4 * F * np.sin(phi)**2) / (sigma * (C_L * np.cos(phi) + C_D * np.sin(phi)))
        a_new = 1 / (k_a + 1)
        k_a_prime = (4 * F * np.sin(phi) * np.cos(phi)) / (sigma * (C_L * np.sin(phi) - C_D * np.cos(phi)))
        a_prime_new = 1 / (k_a_prime - 1)
        other_ret_values = [V_rel, phi, C_L, C_D]
        return a_new, a_prime_new, other_ret_values

    for _ in range(max_iterations):
        a_new, a_prime_new, _ = compute_a_and_a_prime(a, a_prime, c, beta, r, V_inf, Omega, B, R)
        # print(f"a = {a_new}, a' = {a_prime_new}")
        
        if abs(a - a_new) < tolerance and abs(a_prime - a_prime_new) < tolerance:
            a = a_new
            a_prime = a_prime_new
            # run one  more time to compute values for the converged a and a'
            _, _, [V_rel, phi, C_L, C_D] = compute_a_and_a_prime(a, a_prime, c, beta, r, V_inf, Omega, B, R)
            break

        a = a_new
        a_prime = a_prime_new
    
    else:
        raise RuntimeError("Max iterations exceeded for a and a' computation")

    # Calculate torque and thrust
    dQ = 0.5 * rho * V_rel**2 * c * B * (C_L * np.sin(phi) - C_D * np.cos(phi)) * r * dr
    dT = 0.5 * rho * V_rel**2 * c * B * (C_L * np.cos(phi) + C_D * np.sin(phi)) * dr
    # print(f"dQ = {dQ}, dT = {dT}")
    return dQ, dT, V_rel


def constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr):
    x_reshaped = x.reshape(-1, 2)  # Reshape to (num_radii, 2)
    c = x_reshaped[:, 0]
    beta = x_reshaped[:, 1]
    cons = []
    Q = 0
    T = 0
    for i in range(len(r)):
        dQ, dT, _ = bem_solver(c[i], beta[i], r[i], V_inf, Omega, rho, B, R, dr)
        Q += dQ
        T += dT
        C_limit = 0.178 * np.exp(-3.083*(r[i]-0.109)) + 0.049
        beta_limit = 3.412 * np.exp(-3.514*(r[i]-0.615)) - 5.883 + 6
        cons.append(0.15 - abs(c[i] - C_limit))
        cons.append(np.radians(10) - abs(beta[i] - np.radians(beta_limit)))
    cons.append(Q_max - Q) # torque constraint on rotor
    # cons.append(Q) # make Q greater than 0
    cons.append(T_max - T) # thrust constraint on rotor
    # print("Constraints:", cons)
    return np.array(cons)


def objective_function(x, r, V_inf, Omega, rho, B, R, dr):
    x_reshaped = x.reshape(-1, 2)  # Reshape to (num_radii, 2)
    c = x_reshaped[:, 0]
    beta = x_reshaped[:, 1]
    # total_cost = 0.0
    Q = 0
    for i in range(len(r)):
        dQ, _, _ = bem_solver(c[i], beta[i], r[i], V_inf, Omega, rho, B, R, dr)
        Q += dQ
    P = Omega * Q
    # print(f"Q_opt = {Q}")
    # print(f"Total cost {total_cost}")
    objective = -P #* 8700 * np.exp(-(V_inf/60)**2)  # 8700 * np.exp(-(V_inf / 80 )**2)
    return objective


def run_optimization():
    # Parameters
    V_inf = 10.0    # Wind speed [m/s]
    Omega = 1.57    # Angular velocity [rad/s]
    rho = 1.225     # Air density [kg/m³]
    B = 3           # Number of blades
    R = 1.6     # Blade radius [m]
    Q_max = 500.0  # Maximum torque [Nm]
    T_max = 500.0  # Maximum thrust [N]

    # Radial positions
    r = np.linspace(0.2, R, 10)
    dr = r[1] - r[0]

    num_radii = len(r)
    c0 = 0.178 * np.exp(-3.083*(r-0.109)) + 0.049
    beta0 = np.radians(3.412 * np.exp(-3.514*(r-0.615)) - 5.883 + 6)
    # c0 = np.zeros(r.shape[0]) + 0.01
    # print(c0.shape)
    # beta0 = np.zeros(r.shape[0]) + np.pi/15
    x0 = np.column_stack((c0, beta0)).flatten()  # Flatten into 1D

    result = minimize(
        objective_function,
        x0,
        args=(r, V_inf, Omega, rho, B, R, dr),
        method='trust-constr',
        bounds=[(1e-3, 0.4), (np.radians(0), np.radians(20))] * num_radii,
        constraints=[
            {'type': 'ineq', 'fun': lambda x: constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr)}
        ]
    )
    return result, r, R


opt_result, r, R = run_optimization()
x_opt = opt_result.x

print(x_opt.shape)
print(opt_result.success)
print(opt_result.message)
print(opt_result.fun)

x_reshaped = x_opt.reshape(-1, 2)  # Reshape to (num_radii, 2)
c = x_reshaped[:, 0]
# change to deg
beta = np.degrees(x_reshaped[:, 1])

# Reference curves
r_ref = np.linspace(0.2, R, 20)
C_limit = 0.178 * np.exp(-3.083*(r_ref-0.109)) + 0.049
beta_limit = 3.412 * np.exp(-3.514*(r_ref-0.615)) - 5.883 + 6

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(r, c, 'b-o', label='Optimized')
plt.plot(r_ref, C_limit, 'r--', label='C_limit')
plt.xlabel('Radius (m)')
plt.ylabel('Chord (m)')
plt.title('Chord Distribution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(r, beta, 'b-o', label='Optimized')
plt.plot(r_ref, beta_limit, 'r--', label='β_limit')
plt.xlabel('Radius (m)')
plt.ylabel('Twist Angle (degrees)')
plt.title('Twist Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()