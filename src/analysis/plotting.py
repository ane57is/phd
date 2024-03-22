import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.models.type_11_models.seir import (
    direct_transmission_over_two_connected_subpopulations_seird_model,
    direct_transmission_over_one_population_as_in_plos_paper,
    direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model,
    simple_demographic_model,
)
from src.parameters.params import default_seir_params, default_two_cfrs_params
from src.parameters.params import initial_christian_population
from src.parameters.params import initial_pagan_population


def plot_seir_model(
    solution,
    t,
    start_year,
    end_year,
    compartment_indices,
    compartment_labels,
    every_nth_year=5,  # TODO: rename to something more label-indicating
    y_tick_interval=100_000,
    display_y_label_every_n_ticks=1,
    x_label="Time (years)",
    y_label="Number of Individuals",
    plot_title="SEIR Model Over Time",
):
    """
    Plots the compartments of an SEIR model given the solution object, time array,
    and the start and end year for x-axis labeling.

    Parameters:
    - solution: The solution object returned by scipy.integrate.solve_ivp.
    - t: The time points at which the solution was evaluated.
    - start_year: The starting year of the simulation for labeling the x-axis.
    - end_year: The ending year of the simulation for labeling the x-axis.
    - compartment_indices: A list of indices for the compartments to be plotted.
    - compartment_labels: A list of labels corresponding to the compartments.
    - every_nth_year: Interval for labeling years on the x-axis, defaults to 5.
    - y_tick_interval: Interval between y-axis ticks.
    - display_y_label_every_n_ticks: Controls which y-axis labels are displayed.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - plot_title: Title of the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the specified compartments
    for index, label in zip(compartment_indices, compartment_labels):
        ax.plot(t, solution.y[index, :], label=label)

    # Set up the x-axis with year ticks and labels
    total_days = (end_year - start_year + 1) * 365
    year_tick_positions = np.arange(0, total_days, 365)
    year_tick_labels = np.arange(start_year, end_year + 1)
    ax.set_xticks(year_tick_positions)
    ax.set_xticklabels(year_tick_labels, rotation=45)

    # Optionally show every nth year label to avoid clutter
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % every_nth_year != 0:
            label.set_visible(False)

    # Set up the y-axis with custom ticks based on the maximum value across specified compartments
    y_max = np.max([np.max(solution.y[i, :]) for i in compartment_indices])
    y_ticks = np.arange(0, y_max + y_tick_interval, y_tick_interval)
    y_labels = [
        str(int(y)) if index % display_y_label_every_n_ticks == 0 else ""
        for index, y in enumerate(y_ticks)
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Set axis' labels and plot title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend()

    plt.tight_layout()
    plt.show()


def proof_of_concept_solve_and_plot_ap_as_smallpox_in_rome(
    start_year=165, end_year=189
):
    """
    Solve and plot the case of my PLOS ONE article as a proof of concept (showcase this approach works).
    """
    # Initial conditions (just the population of the city of Rome and 1 infected - no other people,
    # and not distinguishing Christians and Pagans.
    y0 = [0, 0, 0, 0, 0, 0, 923405, 0, 1, 0, 0, 0]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_one_population_as_in_plos_paper(
            y, t, default_seir_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant only to the Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
    )


def proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_in_empire(
    start_year=165,
    end_year=189,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_seird_model(
            y, t, default_seir_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=1_000_000,
        display_y_label_every_n_ticks=10,
        plot_title="Antonine Plague as smallpox over two subpopulations in the whole Empire",
    )


def proof_of_concept_solve_and_plot_basic_demographic_development_after_ap(
    solution, start_year=190, end_year=248, parameters=default_seir_params
):
    """
    Continues the simulation with a simple demographic model
    using the final state from a previous model as the initial condition.

    Parameters:
    - solution: The solution object from the previous model.
    - start_year: The starting year for the demographic model simulation.
    - end_year: The ending year for the demographic model simulation.
    - parameters: The parameters object for the demographic model.
    """
    # Initial conditions for the demographic model are the final state from the previous solution
    # y0 = solution.y[:, -1]
    compartments = solution.y[:, -1]
    y0 = [max(0, compartment) for compartment in compartments]

    # Timeframe of simulation for the demographic model
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model(y, t, parameters)

    solution = solve_ivp(
        wrapper_for_solve_ivp_demographic, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    # compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_indices = [0, 6]
    compartment_labels = [
        "Susceptible Christians",
        # 'Exposed Christians',
        # 'Infected Christians',
        # 'Recovered Christians',
        # 'Deceased Christians',
        "Susceptible Pagans",
        # 'Exposed Pagans',
        # 'Infected Pagans',
        # 'Recovered Pagans',
        # 'Deceased Pagans'
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="Demographic development after the Antonine Plague over two subpopulations in the whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_in_empire_and_continue_with_demographic_dev(
    start_year=165,
    end_year=189,
    demographic_end_year=248,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_seird_model(
            y, t, default_seir_params
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    proof_of_concept_solve_and_plot_basic_demographic_development_after_ap(
        solution_ap, end_year + 1, demographic_end_year, default_seir_params
    )


def proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_with_two_cfrs_in_empire(
    start_year=165,
    end_year=189,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, default_two_cfrs_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=1_000_000,
        display_y_label_every_n_ticks=10,
        plot_title="Antonine Plague as smallpox with two CFRs over two subpopulations in the whole Empire",
    )


proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_with_two_cfrs_in_empire()
