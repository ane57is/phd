"""
The main script that sets up the environment, runs the model, and invokes analysis functions.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Define the differential equations
def seir_model(y, t, betaC, betaP, sigma, gamma, muC, muP):
    SC, EC, IC, RC, DC, SP, EP, IQP, RP, DP = y

    # Christians
    dSC_dt = -betaC * SC * (IC + IQP)
    dEC_dt = betaC * SC * (IC + IQP) - sigma * EC
    dIC_dt = sigma * EC - gamma * IC - muC * IC
    dRC_dt = gamma * IC
    dDC_dt = muC * IC

    # Pagans (only get infected by infectious Christians)
    dSP_dt = -betaP * SP * IC
    dEP_dt = betaP * SP * IC - sigma * EP
    dIQP_dt = sigma * EP - gamma * IQP - muP * IQP
    dRP_dt = gamma * IQP
    dDP_dt = muP * IQP

    return dSC_dt, dEC_dt, dIC_dt, dRC_dt, dDC_dt, dSP_dt, dEP_dt, dIQP_dt, dRP_dt, dDP_dt


# Initial conditions
SC0 = 100  # Initial susceptible Christians
EC0 = 0  # Initial exposed Christians
IC0 = 1  # Initial infectious Christians
RC0 = 0  # Initial recovered Christians
DC0 = 0  # Initial dead Christians

SP0 = 1000  # Initial susceptible Pagans
EP0 = 0  # Initial exposed Pagans
IQP0 = 0  # Initial infectious quarantined Pagans
RP0 = 0  # Initial recovered Pagans
DP0 = 0  # Initial dead Pagans

# Parameters
betaC = 0.3  # Infection rate among Christians
betaP = 0.1  # Infection rate from Christians to Pagans
sigma = 1 / 5  # Rate of incubation
gamma = 1 / 7  # Rate of recovery
muC = 0.01  # Death rate for Christians
muP = 0.01  # Death rate for Pagans

# Time vector
t = np.linspace(0, 160, 160)

# Solve the differential equations
init_conditions = SC0, EC0, IC0, RC0, DC0, SP0, EP0, IQP0, RP0, DP0
sol = odeint(seir_model, init_conditions, t, args=(betaC, betaP, sigma, gamma, muC, muP))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='Susceptible Christians')
plt.plot(t, sol[:, 2], label='Infectious Christians')
plt.plot(t, sol[:, 4], label='Dead Christians')
plt.plot(t, sol[:, 5], label='Susceptible Pagans')
plt.plot(t, sol[:, 7], label='Infectious Quarantined Pagans')
plt.plot(t, sol[:, 9], label='Dead Pagans')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIR Model for Christians and Pagans')
plt.show()