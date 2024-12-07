import numpy as np
from scipy.optimize import minimize
# from scipy.interpolate import interp1d
# from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt

# def load_airfoil_data():
#     df = pd.read_csv('airfoil_lookup.csv')
#     cl_interp = interp1d(df['Alpha'], df['Cl'], bounds_error=False, 
#                         fill_value=(df['Cl'].iloc[0], df['Cl'].iloc[-1]))
#     cd_interp = interp1d(df['Alpha'], df['Cd'], bounds_error=False, 
#                         fill_value=(df['Cd'].iloc[0], df['Cd'].iloc[-1]))
#     return cl_interp, cd_interp

df = pd.read_csv('airfoil_lookup.csv')

def load_airfoil_data(alpha):
    """Retrieve Cl and Cd from airfoil lookup table for a given angle of attack."""
    nearest_idx = (df['Alpha'] - alpha).abs().idxmin()
    cl = df.loc[nearest_idx, 'Cl']
    cd = df.loc[nearest_idx, 'Cd']
    return cl, cd

def bem_solver(c, beta, r, V_inf, Omega, rho, B, R):
    # Initialize induction factors
    a = 0.3
    a_prime = 0.0
    max_iterations = 100
    tolerance = 1e-20  
    for _ in range(max_iterations):
        # Flow angle
        phi = np.arctan2(V_inf * (1 - a), Omega * r * (1 + a_prime))
        # print(phi)
        print(a_prime)
        # Relative velocity
        V_rel = np.sqrt(V_inf**2 * (1 - a)**2 + Omega**2 * r**2 * (1 + a_prime)**2)
        
        # Angle of attack
        alpha = np.degrees(phi - beta)
        
        # Get coefficients
        # print(alpha)
        C_L, C_D = load_airfoil_data(alpha)
        # Local solidity
        sigma = (B * c) / (2 * np.pi * r)
        
        eps = 1e-6
        # Prandtl tip loss factor
        f = ((-B * (R - r + eps))/ (2 * r * np.sin(phi) + eps))
        exp_f = np.exp(f)

        exp_f = min(exp_f, 1.0)
        F = (2/np.pi) * np.arccos(exp_f)

        # print(((C_L * np.cos(phi) + C_D * np.sin(phi))))
        k_a = (4 * F * np.sin(phi)) / (sigma * (C_L * np.cos(phi) + C_D * np.sin(phi)))
        a_new = 1 / (k_a + 1)
        k_a_prime = 4 * F * np.sin(phi) * np.cos(phi) / (sigma * (C_L * np.sin(phi) - C_D * np.cos(phi)))
        a_prime_new = 1 / (k_a_prime - 1)
        
        if abs(a - a_new) < tolerance and abs(a_prime - a_prime_new) < tolerance:
            break

        a = a_new
        a_prime = a_prime_new
    # Calculate torque and thrust
    dQ = 0.5 * rho * V_rel**2 * c * B * (C_L * np.sin(phi) - C_D * np.cos(phi)) * r
    dT = 0.5 * rho * V_rel**2 * c * B * (C_L * np.cos(phi) + C_D * np.sin(phi)) * c
    
    return dQ, dT, V_rel

def constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max):
    c, beta = x
    dQ, dT, _ = bem_solver(c, beta, r, V_inf, Omega, rho, B, R)
    
    C_limit = 0.178 * np.exp(-3.083*(r-0.109)) + 0.049
    beta_limit = 3.412 * np.exp(-3.514*(r-0.615)) - 5.883
    
    return [
        abs(c - C_limit),        # Increased margin from limit curve
        abs(beta - np.radians(beta_limit)),  # Wider angle range
        0.24 - c,                       # Maximum chord
        c,                      # Minimum chord
        np.radians(20) - beta,
        beta, 
        dQ - 0.1 * Q_max,             # Minimum torque
        T_max - dT,                    # Maximum thrust
        dT - 0.1 * T_max              # Minimum thrust
    ]

def objective_function(x, r, V_inf, Omega, rho, B, R):
    c, beta = x
    dQ, _, _ = bem_solver(c, beta, r, V_inf, Omega, rho, B, R)
    
    # #Riemann summ to integrate
    # dr = r[1] - r[0]  # Radial step size
    # Q = np.sum(dQ) * dr  # Rectangular integration
    
    P = Omega * dQ
    cost = 5 * c + beta
    
    return cost / (8700 * P * np.exp(-V_inf**2))

def optimize_blade_element(r, V_inf, Omega, rho, B, R, Q_max, T_max):
    # Initial guess based on limit functions
    CL_design = 1
    c0 = (8 * np.pi * r) / (B * CL_design)

    phi = np.arctan2(V_inf, Omega * r)  # Flow angle
    alpha_opt = np.radians(6)  # Optimal angle of attack
    beta0 = phi - alpha_opt

    c0 = 0.1
    beta0 = 0
    x0 = [c0, beta0]
    
    # Bounds
    bounds = [(0.0, 0.4), (np.radians(-20), np.radians(20))]
    
    # Constraints
    cons = [{'type': 'ineq', 'fun': lambda x: constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max)[i]} 
            for i in range(6)]
    
    result = minimize(objective_function, 
                     x0,
                     args=(r, V_inf, Omega, rho, B, R),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=cons,
                     )
    
    return result.x

# Initialize interpolation functions
# cl_interp, cd_interp = load_airfoil_data()

# Parameters
V_inf = 15.0    # Wind speed [m/s]
Omega = 30.0    # Angular velocity [rad/s]
rho = 1.225     # Air density [kg/m³]
B = 3           # Number of blades
R = 2       # Blade radius [m]
Q_max = 1000.0  # Maximum torque [Nm]
T_max = 2000.0  # Maximum thrust [N]

# Radial positions
r = np.linspace(0.2, R, 20)

# Optimize for each radius
results = []
for r_i in r:
    result = optimize_blade_element(r_i, V_inf, Omega, rho, B, R, Q_max, T_max)
    results.append(result)

# Plot results
results = np.array(results)
chords = results[:, 0]
twists = np.degrees(results[:, 1])

# print(results)

# Reference curves
r_ref = np.linspace(0.2, R, 20)
C_limit = 0.178 * np.exp(-3.083*(r_ref-0.109)) + 0.049
beta_limit = 3.412 * np.exp(-3.514*(r_ref-0.615)) - 5.883

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(r, chords, 'b-o', label='Optimized')
plt.plot(r_ref, C_limit, 'r--', label='C_limit')
plt.xlabel('Radius (m)')
plt.ylabel('Chord (m)')
plt.title('Chord Distribution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(r, twists, 'b-o', label='Optimized')
plt.plot(r_ref, beta_limit, 'r--', label='β_limit')
plt.xlabel('Radius (m)')
plt.ylabel('Twist Angle (degrees)')
plt.title('Twist Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()