import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from src.models.type_11_models.seir import direct_transmission_over_two_connected_subpopulations_seird_model
from src.parameters.params import default_seir_params
from src.parameters.params import initial_christian_population
from src.parameters.params import initial_pagan_population


# Initial conditions
y0 = [initial_christian_population - 1, 0, 1, 0, 0, 0, initial_pagan_population - 1, 0, 1, 0, 0, 0]

# Time vector
t = np.linspace(165, 189, (189 - 165 + 1) * 365)


# Solve the ODE
solution = odeint(direct_transmission_over_two_connected_subpopulations_seird_model, y0, t, args=(default_seir_params,))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t, solution[:, 0], label='Susceptible Christians')
ax.plot(t, solution[:, 1], label='Exposed Christians')
ax.plot(t, solution[:, 2], label='Infected Christians')
ax.plot(t, solution[:, 3], label='Recovered Christians')
ax.plot(t, solution[:, 4], label='Deceased Christians')

ax.plot(t, solution[:, 6], label='Susceptible Pagans')
ax.plot(t, solution[:, 7], label='Exposed Pagans')
ax.plot(t, solution[:, 8], label='Infected Pagans')
ax.plot(t, solution[:, 9], label='Recovered Pagans')
ax.plot(t, solution[:, 10], label='Deceased Pagans')

ax.set_xlabel('Time (years)')
ax.set_ylabel('Number of Individuals')
ax.set_title('SEIRD Model Over Time for Two Connected Subpopulations')
ax.legend()

plt.tight_layout()
plt.show()