import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('xf-n0012-il-200000.csv')

# Retrieve Cl and Cd from airfoil lookup table for a given angle of attack.
def load_airfoil_data(alpha):
    nearest_idx = (df['Alpha'] - alpha).abs().idxmin()
    cl = df.loc[nearest_idx, 'Cl']
    cd = df.loc[nearest_idx, 'Cd']
    return cl, cd

# Solve for different variables using the BEM methodology
def bem_solver(c, beta, r, V_inf, Omega, rho, B, R, dr):
    # Initialize induction factors
    a = 0.0
    a_prime = 0.0
    max_iterations = 1000
    tolerance = 1e-3

    def compute_a_and_a_prime(a, a_prime, c, beta, r, V_inf, Omega, B, R):
        # Flow angle
        phi = np.arctan2(V_inf * (1 - a), Omega * r * (1 + a_prime))

        # Relative velocity
        V_rel = np.sqrt(V_inf**2 * (1 - a)**2 + Omega**2 * r**2 * (1 + a_prime)**2)
        
        # Angle of attack
        alpha = np.degrees(phi - beta)
        
        # Get coefficients
        C_L, C_D = load_airfoil_data(alpha)

        # Local solidity
        sigma = (B * c) / (2 * np.pi * r)
        
        # Additive epsilon to prevent zero divisions and sin/cosine errors
        eps = 1e-6
        
        f = ((-B * (R - r + eps))/ (2 * r * np.sin(phi) + eps))
        exp_f = np.exp(f)

        # Take the minimum to prevent arccos of numbers greater than 1
        exp_f = min(exp_f, 1.0 - eps)
        F = (2/np.pi) * np.arccos(exp_f)

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

# Define and introduce the optimization constraints
def constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr):
    x_reshaped = x.reshape(-1, 2)  # Reshape to (num_radii, 2)
    c = x_reshaped[:, 0]
    beta = x_reshaped[:, 1]
    cons = []
    Q = 0
    T = 0
    # Set the constraints across each interval of the blades radius
    for i in range(len(r)):
        dQ, dT, _ = bem_solver(c[i], beta[i], r[i], V_inf, Omega, rho, B, R, dr)
        Q += dQ
        T += dT
        C_limit = 0.178 * np.exp(-3.083*(r[i]-0.109)) + 0.049
        beta_limit = 3.412 * np.exp(-3.514*(r[i]-0.615)) - 5.883 + 6
        cons.append(0.15 - abs(c[i] - C_limit))
        cons.append(np.radians(10) - abs(beta[i] - np.radians(beta_limit)))
    cons.append(Q_max - Q) # torque constraint on rotor
    cons.append(T_max - T) # thrust constraint on rotor

    return np.array(cons)


def objective_function(x, r, V_inf, Omega, rho, B, R, dr):
    x_reshaped = x.reshape(-1, 2)  # Reshape to (num_radii, 2)
    c = x_reshaped[:, 0]
    beta = x_reshaped[:, 1]
    Q = 0
    for i in range(len(r)):
        dQ, _, _ = bem_solver(c[i], beta[i], r[i], V_inf, Omega, rho, B, R, dr)
        Q += dQ
    P = Omega * Q
    objective = -P   # Negative to maximize the power
    return objective

# Runs optimization on desired method_type
def run_optimization(method_type):
    # Parameters
    V_inf = 4    # Wind speed [m/s]
    Omega = 1.57    # Angular velocity [rad/s]
    rho = 1.225     # Air density [kg/m³]
    B = 3           # Number of blades
    R = 1.6         # Blade radius [m]
    Q_max = 500.0  # Maximum torque [Nm]
    T_max = 500.0  # Maximum thrust [N]

    # Radial positions
    r = np.linspace(0.2, R, 10)
    dr = r[1] - r[0]

    num_radii = len(r)
    c0 = 0.178 * np.exp(-3.083*(r-0.109)) + 0.049
    beta0 = np.radians(3.412 * np.exp(-3.514*(r-0.615)) - 5.883 + 6)
    x0 = np.column_stack((c0, beta0)).flatten()  # Flatten into 1D

    result = minimize(
        objective_function,
        x0,
        args=(r, V_inf, Omega, rho, B, R, dr),
        method= method_type,
        bounds=[(1e-3, 0.4), (np.radians(0), np.radians(20))] * num_radii,
        constraints=[
            {'type': 'ineq', 'fun': lambda x: constraint_functions(x, r, V_inf, Omega, rho, B, R, Q_max, T_max, dr)}
        ]
    )
    return result, r, R


# Run both optimization methods
optimization_result_tc, r, R = run_optimization("trust-constr")
optimization_result_slsqp, r, R = run_optimization("SLSQP")

# Print optimization results
print("Trust-constr results:")
print(optimization_result_tc.success)
print(optimization_result_tc.message)
print("\nSLSQP results:")
print(optimization_result_slsqp.success)
print(optimization_result_slsqp.message)

# Process trust-constr results
x_opt_tc = optimization_result_tc.x
x_reshaped_tc = x_opt_tc.reshape(-1, 2)
c_tc = x_reshaped_tc[:, 0]
beta_tc = np.degrees(x_reshaped_tc[:, 1])

# Process SLSQP results
x_opt_slsqp = optimization_result_slsqp.x
x_reshaped_slsqp = x_opt_slsqp.reshape(-1, 2)
c_slsqp = x_reshaped_slsqp[:, 0]
beta_slsqp = np.degrees(x_reshaped_slsqp[:, 1])

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Trust-constr plots
ax1.plot(r, c_tc, 'b-o', label='Optimized')
ax1.set_xlabel('Radius (m)')
ax1.set_ylabel('Chord (m)')
ax1.set_title('Chord Distribution - trust-constr method')
ax1.legend()
ax1.grid(True)

ax2.plot(r, beta_tc, 'b-o', label='Optimized')
ax2.set_xlabel('Radius (m)')
ax2.set_ylabel('Twist Angle (degrees)')
ax2.set_title('Twist Distribution - trust-constr method')
ax2.legend()
ax2.grid(True)

# SLSQP plots
ax3.plot(r, c_slsqp, 'g-o', label='Optimized')
ax3.set_xlabel('Radius (m)')
ax3.set_ylabel('Chord (m)')
ax3.set_title('Chord Distribution - SLSQP method')
ax3.legend()
ax3.grid(True)

ax4.plot(r, beta_slsqp, 'g-o', label='Optimized')
ax4.set_xlabel('Radius (m)')
ax4.set_ylabel('Twist Angle (degrees)')
ax4.set_title('Twist Distribution - SLSQP method')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()