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
    max_iterations = 200
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
    return dQ, dT, V_rel

# def constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr):
#     c, beta = x
#     # print(f"Beta: {round(beta, 6)}")
#     dQ, dT, _ = bem_solver(c, beta, r, V_inf, Omega, rho, B, R, dr)
#     C_limit = 0.178 * np.exp(-3.083*(r-0.109)) + 0.049
#     beta_limit = 3.412 * np.exp(-3.514*(r-0.615)) - 5.883
    
#     return [
#         0.1 - abs(c - C_limit),        # Increased margin from limit curve
#         np.radians(7.5) - abs(beta - np.radians(beta_limit)),  # Wider angle range
#         # 0.24 - c,                       # Maximum chord -> added in bounds below
#         # c,                      # Minimum chord -> also added below in bounds
#         # np.radians(20) - beta, -> added in bounds below
#         # beta, 
#         dQ - 0.1 * Q_max,             # Minimum torque
#         T_max - dT,                    # Maximum thrust
#         dT - 0.1 * T_max              # Minimum thrust
#     ]

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
        beta_limit = 3.412 * np.exp(-3.514*(r[i]-0.615)) - 5.883
        cons.append(0.2 - abs(c[i] - C_limit))
        cons.append(np.radians(7.5) - abs(beta[i] - np.radians(beta_limit)))
    cons.append(Q_max - Q) # torque constraint on rotor
    cons.append(T_max - T) # thrust constraint on rotor
    return np.array(cons)


# def objective_function(x, r, V_inf, Omega, rho, B, R, dr):
#     c, beta = x
#     dQ, _, _ = bem_solver(c, beta, r, V_inf, Omega, rho, B, R, dr)
#     # print(f"C = {c}, beta = {beta}")
    
#     # #Riemann summ to integrate
#     # dr = r[1] - r[0]  # Radial step size
#     # Q = np.sum(dQ) * dr  # Rectangular integration
    
#     P = Omega * dQ
#     cost = 5 * c + beta
    
#     return cost / (8700 * P * np.exp(-V_inf**2))

def objective_function(x, r, V_inf, Omega, rho, B, R, dr):
    x_reshaped = x.reshape(-1, 2)  # Reshape to (num_radii, 2)
    c = x_reshaped[:, 0]
    beta = x_reshaped[:, 1]
    total_cost = 0.0
    Q = 0
    for i in range(len(r)):
        dQ, _, _ = bem_solver(c[i], beta[i], r[i], V_inf, Omega, rho, B, R, dr)
        Q += dQ
        cost = 5 * c[i] + beta[i]
        total_cost += cost
    P = Omega * Q
    objective = total_cost / (8700 * P * np.exp(-V_inf**2))
    return objective

# def optimize_blade_element(r, V_inf, Omega, rho, B, R, Q_max, T_max, dr):
#     # Initial guess based on limit functions
#     # CL_design = 1
#     # c0 = (8 * np.pi * r) / (B * CL_design)

#     # phi = np.arctan2(V_inf, Omega * r)  # Flow angle
#     # alpha_opt = np.radians(6)  # Optimal angle of attack
#     # beta0 = phi - alpha_opt

#     # c0 = 0.1
#     # beta0 = 0
#     c0 = 0.178 * np.exp(-3.083*(r-0.109)) + 0.049
#     beta0 = np.radians(3.412 * np.exp(-3.514*(r-0.615)) - 5.883)
#     x0 = [c0, beta0]
    
#     # Bounds
#     bounds = [(1e-6, 0.4), (np.radians(-20), np.radians(20))]
    
#     # Constraints
#     cons = [{'type': 'ineq', 'fun': lambda x: constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr)[i]} 
#             for i in range(5)]
    
#     result = minimize(objective_function, 
#                      x0,
#                      args=(r, V_inf, Omega, rho, B, R, dr),
#                      method='SLSQP',
#                      bounds=bounds,
#                      constraints=cons,
#                      )
    
#     return result.x


def run_optimization():
    # Parameters
    V_inf = 5.0    # Wind speed [m/s]
    Omega = 30.0    # Angular velocity [rad/s]
    rho = 1.225     # Air density [kg/m³]
    B = 3           # Number of blades
    R = 1.4       # Blade radius [m]
    Q_max = 1000.0  # Maximum torque [Nm]
    T_max = 2000.0  # Maximum thrust [N]

    # Radial positions
    r = np.linspace(0.2, R, 20)
    dr = r[1] - r[0]

    num_radii = len(r)
    c0 = 0.178 * np.exp(-3.083*(r-0.109)) + 0.049
    beta0 = np.radians(3.412 * np.exp(-3.514*(r-0.615)) - 5.883)
    x0 = np.column_stack((c0, beta0)).flatten()  # Flatten into 1D

    result = minimize(
        objective_function,
        x0,
        args=(r, V_inf, Omega, rho, B, R, dr),
        method='SLSQP',
        bounds=[(1e-6, 0.4), (np.radians(-20), np.radians(20))] * num_radii,
        constraints=[
            {'type': 'ineq', 'fun': lambda x: constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr)}
        ]
    )
    return result.x, r, R

# Initialize interpolation functions
# cl_interp, cd_interp = load_airfoil_data()

# Parameters
# V_inf = 15.0    # Wind speed [m/s]
# Omega = 30.0    # Angular velocity [rad/s]
# rho = 1.225     # Air density [kg/m³]
# B = 3           # Number of blades
# R = 1.4       # Blade radius [m]
# Q_max = 1000.0  # Maximum torque [Nm]
# T_max = 2000.0  # Maximum thrust [N]

# # Radial positions
# r = np.linspace(0.2, R, 20)
# dr = r[1] - r[0]

# # Optimize for each radius
# results = []
# for r_i in r:
#     result = optimize_blade_element(r_i, V_inf, Omega, rho, B, R, Q_max, T_max, dr)
#     results.append(result)

x_opt, r, R = run_optimization()

print(x_opt.shape)

x_reshaped = x_opt.reshape(-1, 2)  # Reshape to (num_radii, 2)
c = x_reshaped[:, 0]
beta = x_reshaped[:, 1]

# Plot results
# results = np.array(results)
# chords = results[:, 0]
# twists = np.degrees(results[:, 1])

# print(results)

# Reference curves
r_ref = np.linspace(0.2, R, 20)
C_limit = 0.178 * np.exp(-3.083*(r_ref-0.109)) + 0.049
beta_limit = 3.412 * np.exp(-3.514*(r_ref-0.615)) - 5.883

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