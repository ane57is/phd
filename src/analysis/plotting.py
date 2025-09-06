from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator
import os
from scipy.integrate import solve_ivp
from tabulate import tabulate
import time
from src.models.type_11_models.seir import (
    direct_transmission_over_two_connected_subpopulations_seird_model,
    direct_transmission_over_one_population_as_in_plos_paper,
    direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model,
    simple_demographic_model,
    simple_demographic_model_with_conversion,
    direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model,
    direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model,
    simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones,
    direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model,
    direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model,
    simple_demographic_model_subpopulation_pairs_with_conversion_to_dense_zones,
    direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model
)
from src.parameters.params import (
    default_seir_params,
    default_two_cfrs_params,
    measles_seir_params,
    measles_seir_params_with_lower_cfr_for_c_and_starks_conversion,
    smallpox_param_sets,
    measles_param_sets,
    cchf_param_sets,
    evd_param_sets,
    lassa_param_sets
)
from src.parameters.params import (
    initial_christian_population,
    initial_pagan_population,
    initial_populations_in_zones,
    zeleners_initial_delta_1,
    zeleners_initial_delta_2,
    zeleners_initial_delta_3,
    zeleners_initial_delta_4,
)


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

    for index, label in zip(compartment_indices, compartment_labels):
        ax.plot(solution.t, solution.y[index, :], label=label)

    total_days = (end_year - start_year + 1) * 365
    year_tick_positions = np.arange(0, total_days, 365)
    year_tick_labels = np.arange(start_year, end_year + 1)
    ax.set_xticks(year_tick_positions)
    ax.set_xticklabels(year_tick_labels, rotation=45)

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % every_nth_year != 0:
            label.set_visible(False)

    y_max = min(np.max([np.max(solution.y[i, :]) for i in compartment_indices]), 1e8)
    print(f"y_max: {y_max}")
    y_ticks = np.arange(0, y_max + y_tick_interval, y_tick_interval)
    y_labels = [
        str(int(y)) if index % display_y_label_every_n_ticks == 0 else ""
        for index, y in enumerate(y_ticks)
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

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

    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_one_population_as_in_plos_paper(
            y, t, default_seir_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

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

    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_seird_model(
            y, t, default_seir_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )


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

    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, default_two_cfrs_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )


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


def proof_of_concept_solve_and_plot_ap_demography_and_cp_with_two_subpopulations_and_smaller_cfrs_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # The Antonine Plague part:
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0_ap = [
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
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.arange(0, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, default_two_cfrs_params
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="BDF", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:, -1]
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.arange(0, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model(y, t, default_seir_params)

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="BDF",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]

    # Move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Reshuffling for Christians
    susceptible_christians_sum = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum
    y0_cp[1] = 0  # Exposed Christians
    y0_cp[2] = 1  # Infected Christians
    y0_cp[3] = 0  # Recovered Christians
    # y0_cp[4] = 0  # Deceased Christians (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum
    y0_cp[7] = 0  # Exposed Pagans
    y0_cp[8] = 1  # Infected Pagans
    y0_cp[9] = 0  # Recovered Pagans
    # y0_cp[10] = 0  # Deceased Pagans (set to 0 to see what CP does)

    # Timeframe of simulation for the demographic model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.arange(0, total_days_cp)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, measles_seir_params
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="BDF",
        t_eval=t_cp,
    )


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
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP as measles with two CFRs over two subpopulations in the whole Empire (after AP and demo)",
    )


def proof_of_concept_solve_and_plot_ap_only_demographic_development_with_conversion_in_empire(
    start_year=165,
    end_year=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population,
        0,
        0,
        0,
        0,
        0,
        initial_pagan_population,
        0,
        0,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp(t, y):
        return simple_demographic_model_with_conversion(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )


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
        plot_title="Antonine Plague as no disease but instead conversion from P to C in the whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_with_conversion_and_smaller_cfr_for_christians(
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

    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )


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
        plot_title="Antonine Plague as smallpox with smaller CFR for Christians and conversion in whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # The Antonine Plague part:
    y0_ap = [
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
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.arange(0, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="BDF", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:, -1]
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.arange(0, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_with_conversion(y, t, smallpox_seir_params_with_starks_conversion)

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="BDF",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]

    # Move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Reshuffling for Christians
    susceptible_christians_sum = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum
    y0_cp[1] = 0  # Exposed Christians
    y0_cp[2] = 1  # Infected Christians
    y0_cp[3] = 0  # Recovered Christians
    # y0_cp[4] = 0  # Deceased Christians (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum
    y0_cp[7] = 0  # Exposed Pagans
    y0_cp[8] = 1  # Infected Pagans
    y0_cp[9] = 0  # Recovered Pagans
    # y0_cp[10] = 0  # Deceased Pagans (set to 0 to see what CP does)

    # Timeframe of simulation for the demographic model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.arange(0, total_days_cp)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="BDF",
        t_eval=t_cp,
    )


    # compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        # "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        # "Deceased Pagans",
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    s_c = compartments_after_conversion[0]
    e_c = compartments_after_conversion[1]
    i_c = compartments_after_conversion[2]
    r_c = compartments_after_conversion[3]
    d_c = compartments_after_conversion[4]
    a_c = compartments_after_conversion[5]
    s_p = compartments_after_conversion[6]
    e_p = compartments_after_conversion[7]
    i_p = compartments_after_conversion[8]
    r_p = compartments_after_conversion[9]
    d_p = compartments_after_conversion[10]
    a_p = compartments_after_conversion[11]
    alive_c = s_c + e_c + i_c + r_c
    alive_p = s_c + e_p + i_p + r_p

    print(
        f"s_c = {s_c}\n"
        f"e_c = {e_c}\n"
        f"i_c = {i_c}\n"
        f"r_c = {r_c}\n"
        f"d_c = {d_c}\n"
        f"a_c = {a_c}\n"
        f"s_p = {s_p}\n"
        f"e_p = {e_p}\n"
        f"i_p = {i_p}\n"
        f"r_p = {r_p}\n"
        f"d_p = {d_p}\n"
        f"a_p = {a_p}\n"
        f"alive_c = {alive_c}\n"
        f"alive_p = {alive_p}"
    )


def proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_in_four_separate_zones(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # The Antonine Plague part:
    y0_ap = [
        initial_populations_in_zones["initial_christian_population_zone_1"] - 1,  # s1_c
        0,  # e1_c
        1,  # i1_c
        0,  # r1_c
        0,  # d1_c
        0,  # a1_c
        initial_populations_in_zones["initial_pagan_population_zone_1"] - 1,  # s1_p
        0,  # e1_p
        1,  # i1_p
        0,  # r1_p
        0,  # d1_p
        0,  # a1_p
        initial_populations_in_zones["initial_christian_population_zone_2"] - 1,  # s2_c
        0,  # e2_c
        1,  # i2_c
        0,  # r2_c
        0,  # d2_c
        0,  # a2_c
        initial_populations_in_zones["initial_pagan_population_zone_2"] - 1,  # s2_p
        0,  # e2_p
        1,  # i2_p
        0,  # r2_p
        0,  # d2_p
        0,  # a2_p
        initial_populations_in_zones["initial_christian_population_zone_3"] - 1,  # s3_c
        0,  # e3_c
        1,  # i3_c
        0,  # r3_c
        0,  # d3_c
        0,  # a3_c
        initial_populations_in_zones["initial_pagan_population_zone_3"] - 1,  # s3_p
        0,  # e3_p
        1,  # i3_p
        0,  # r3_p
        0,  # d3_p
        0,  # a3_p
        initial_populations_in_zones["initial_christian_population_zone_4"] - 1,  # s4_c
        0,  # e4_c
        1,  # i4_c
        0,  # r4_c
        0,  # d4_c
        0,  # a4_c
        initial_populations_in_zones["initial_pagan_population_zone_4"] - 1,  # s4_p
        0,  # e4_p
        1,  # i4_p
        0,  # r4_p
        0,  # d4_p
        0,  # a4_p
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.arange(0, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="BDF", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:, -1]
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.arange(0, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="BDF",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Timeframe of simulation for the demographic model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.arange(0, total_days_cp)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="BDF",
        t_eval=t_cp,
    )


    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone§ 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
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

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}"
    )


def proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_with_four_dynamic_deltas(
    start_year_ap=165,  # 165
    end_year_ap=171,  # 189
    # end_year_ap=189,  # 189
    end_year_demographic=177,  # 248
    # end_year_demographic=248,  # 248
    end_year_cp=183,  # 270
    # end_year_cp=270,  # 270
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # The Antonine Plague part:
    y0_ap = [
        initial_populations_in_zones["initial_christian_population_zone_1"] - 1,  # s1_c
        0,  # e1_c
        1,  # i1_c
        0,  # r1_c
        0,  # d1_c
        0,  # a1_c
        initial_populations_in_zones["initial_pagan_population_zone_1"] - 1,  # s1_p
        0,  # e1_p
        1,  # i1_p
        0,  # r1_p
        0,  # d1_p
        0,  # a1_p
        initial_populations_in_zones["initial_christian_population_zone_2"] - 1,  # s2_c
        0,  # e2_c
        1,  # i2_c
        0,  # r2_c
        0,  # d2_c
        0,  # a2_c
        initial_populations_in_zones["initial_pagan_population_zone_2"] - 1,  # s2_p
        0,  # e2_p
        1,  # i2_p
        0,  # r2_p
        0,  # d2_p
        0,  # a2_p
        initial_populations_in_zones["initial_christian_population_zone_3"] - 1,  # s3_c
        0,  # e3_c
        1,  # i3_c
        0,  # r3_c
        0,  # d3_c
        0,  # a3_c
        initial_populations_in_zones["initial_pagan_population_zone_3"] - 1,  # s3_p
        0,  # e3_p
        1,  # i3_p
        0,  # r3_p
        0,  # d3_p
        0,  # a3_p
        initial_populations_in_zones["initial_christian_population_zone_4"] - 1,  # s4_c
        0,  # e4_c
        1,  # i4_c
        0,  # r4_c
        0,  # d4_c
        0,  # a4_c
        initial_populations_in_zones["initial_pagan_population_zone_4"] - 1,  # s4_p
        0,  # e4_p
        1,  # i4_p
        0,  # r4_p
        0,  # d4_p
        0,  # a4_p,
        zeleners_initial_delta_1,  # delta_1_c
        zeleners_initial_delta_1,  # delta_1_p
        zeleners_initial_delta_2,  # delta_2_c
        zeleners_initial_delta_2,  # delta_2_p
        zeleners_initial_delta_3,  # delta_3_c
        zeleners_initial_delta_3,  # delta_3_p
        zeleners_initial_delta_4,  # delta_4_c
        zeleners_initial_delta_4,  # delta_4_p
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    # t_ap = np.arange(0, total_days_ap)
    t_ap = np.linspace(0, total_days_ap, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap,
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    # t_demographic = np.arange(0, total_days_demographic)
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]


    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p


    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    # t_cp = np.arange(0, total_days_cp)
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp,
        atol=1e-8,
        rtol=1e-6,
    )



    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
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

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}"
    )


def new_poc_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_with_four_deltas(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270
):
    # The Antonine Plague part:
    y0_ap = [
        initial_populations_in_zones["initial_christian_population_zone_1"] - 1,  # s1_c
        0,  # e1_c
        1,  # i1_c
        0,  # r1_c
        0,  # d1_c
        0,  # a1_c
        initial_populations_in_zones["initial_pagan_population_zone_1"] - 1,  # s1_p
        0,  # e1_p
        1,  # i1_p
        0,  # r1_p
        0,  # d1_p
        0,  # a1_p
        initial_populations_in_zones["initial_christian_population_zone_2"] - 1,  # s2_c
        0,  # e2_c
        1,  # i2_c
        0,  # r2_c
        0,  # d2_c
        0,  # a2_c
        initial_populations_in_zones["initial_pagan_population_zone_2"] - 1,  # s2_p
        0,  # e2_p
        1,  # i2_p
        0,  # r2_p
        0,  # d2_p
        0,  # a2_p
        initial_populations_in_zones["initial_christian_population_zone_3"] - 1,  # s3_c
        0,  # e3_c
        1,  # i3_c
        0,  # r3_c
        0,  # d3_c
        0,  # a3_c
        initial_populations_in_zones["initial_pagan_population_zone_3"] - 1,  # s3_p
        0,  # e3_p
        1,  # i3_p
        0,  # r3_p
        0,  # d3_p
        0,  # a3_p
        initial_populations_in_zones["initial_christian_population_zone_4"] - 1,  # s4_c
        0,  # e4_c
        1,  # i4_c
        0,  # r4_c
        0,  # d4_c
        0,  # a4_c
        initial_populations_in_zones["initial_pagan_population_zone_4"] - 1,  # s4_p
        0,  # e4_p
        1,  # i4_p
        0,  # r4_p
        0,  # d4_p
        0,  # a4_p
        zeleners_initial_delta_1,  # delta_1_c
        zeleners_initial_delta_1,  # delta_1_p
        zeleners_initial_delta_2,  # delta_2_c
        zeleners_initial_delta_2,  # delta_2_p
        zeleners_initial_delta_3,  # delta_3_c
        zeleners_initial_delta_3,  # delta_3_p
        zeleners_initial_delta_4,  # delta_4_c
        zeleners_initial_delta_4,  # delta_4_p
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.linspace(0, total_days_ap, total_days_ap + 1)

    # Solve the ODE for the Antonine Plague
    # TODO: implement the seir function with some delta usage - a new one, tho!
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    print(y0_ap)
    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    # compartments_demographic = solution_ap.y[:, -1]
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic + 1)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    print(y0_demographic)
    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    # TODO: check if this extension is not a mistake
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p

    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    print(y0_cp)
    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp
    )


    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
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

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    alive_total = alive1_c + alive1_p + alive2_c + alive2_p + alive3_c + alive3_p + alive4_c + alive4_p

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}\n"
        f"alive_total = {alive_total}\n"
    )

    print("Length of t_cp:", len(t_cp))
    print("Length of solution_cp.t:", len(solution_cp.t))
    print("solution_cp.t[-1]:", solution_cp.t[-1])

    for i, label in enumerate(compartment_labels):
        max_val = np.max(solution_cp.y[compartment_indices[i], :])
        print(f"{label}: max = {max_val}")

    print("Solver success:", solution_cp.success)
    print("Solver message:", solution_cp.message)

    print(
        "final deltas:\n"
        f"delta_1_c = {y0_cp[48]}\n"
        f"delta_1_p = {y0_cp[49]}\n"
        f"delta_2_c = {y0_cp[50]}\n"
        f"delta_2_p = {y0_cp[51]}\n"
        f"delta_3_c = {y0_cp[52]}\n"
        f"delta_3_p = {y0_cp[53]}\n"
        f"delta_4_c = {y0_cp[54]}\n"
        f"delta_4_p = {y0_cp[55]}\n"
    )


def poc_solve_and_plot_ap_demo_cp_with_converstion_to_denser_of_four_zones_and_smaller_cfr_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270
):
    # The Antonine Plague part:
    y0_ap = [
        initial_populations_in_zones["initial_christian_population_zone_1"] - 1,  # s1_c
        0,  # e1_c
        1,  # i1_c
        0,  # r1_c
        0,  # d1_c
        0,  # a1_c
        initial_populations_in_zones["initial_pagan_population_zone_1"] - 1,  # s1_p
        0,  # e1_p
        1,  # i1_p
        0,  # r1_p
        0,  # d1_p
        0,  # a1_p
        initial_populations_in_zones["initial_christian_population_zone_2"] - 1,  # s2_c
        0,  # e2_c
        1,  # i2_c
        0,  # r2_c
        0,  # d2_c
        0,  # a2_c
        initial_populations_in_zones["initial_pagan_population_zone_2"] - 1,  # s2_p
        0,  # e2_p
        1,  # i2_p
        0,  # r2_p
        0,  # d2_p
        0,  # a2_p
        initial_populations_in_zones["initial_christian_population_zone_3"] - 1,  # s3_c
        0,  # e3_c
        1,  # i3_c
        0,  # r3_c
        0,  # d3_c
        0,  # a3_c
        initial_populations_in_zones["initial_pagan_population_zone_3"] - 1,  # s3_p
        0,  # e3_p
        1,  # i3_p
        0,  # r3_p
        0,  # d3_p
        0,  # a3_p
        initial_populations_in_zones["initial_christian_population_zone_4"] - 1,  # s4_c
        0,  # e4_c
        1,  # i4_c
        0,  # r4_c
        0,  # d4_c
        0,  # a4_c
        initial_populations_in_zones["initial_pagan_population_zone_4"] - 1,  # s4_p
        0,  # e4_p
        1,  # i4_p
        0,  # r4_p
        0,  # d4_p
        0,  # a4_p
        zeleners_initial_delta_1,  # delta_1_c
        zeleners_initial_delta_1,  # delta_1_p
        zeleners_initial_delta_2,  # delta_2_c
        zeleners_initial_delta_2,  # delta_2_p
        zeleners_initial_delta_3,  # delta_3_c
        zeleners_initial_delta_3,  # delta_3_p
        zeleners_initial_delta_4,  # delta_4_c
        zeleners_initial_delta_4,  # delta_4_p
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.linspace(0, total_days_ap, total_days_ap + 1)

    # Solve the ODE for the Antonine Plague
    # TODO: implement the seir function with some delta usage - a new one, tho!
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    print(y0_ap)
    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    # compartments_demographic = solution_ap.y[:, -1]
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic + 1)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_to_dense_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    print(y0_demographic)
    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    # TODO: check if this extension is not a mistake
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p

    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    print(y0_cp)
    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp
    )


    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
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

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    alive_total = alive1_c + alive1_p + alive2_c + alive2_p + alive3_c + alive3_p + alive4_c + alive4_p
    alive_c = alive1_c + alive2_c + alive3_c + alive4_c
    alive_p = alive1_p + alive2_p + alive3_p + alive4_p
    alive_c_percentage_of_total = (alive_c / alive_total) * 100

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}\n"
        f"alive_total = {alive_total}\n"
        f"alive_c = {alive_c}\n"
        f"alive_p = {alive_p}\n"
        f"alive_c_percentage_of_total = {alive_c_percentage_of_total} %\n"
    )

    print("Length of t_cp:", len(t_cp))
    print("Length of solution_cp.t:", len(solution_cp.t))
    print("solution_cp.t[-1]:", solution_cp.t[-1])

    for i, label in enumerate(compartment_labels):
        max_val = np.max(solution_cp.y[compartment_indices[i], :])
        print(f"{label}: max = {max_val}")

    print("Solver success:", solution_cp.success)
    print("Solver message:", solution_cp.message)

    print(
        "final deltas:\n"
        f"delta_1_c = {y0_cp[48]}\n"
        f"delta_1_p = {y0_cp[49]}\n"
        f"delta_2_c = {y0_cp[50]}\n"
        f"delta_2_p = {y0_cp[51]}\n"
        f"delta_3_c = {y0_cp[52]}\n"
        f"delta_3_p = {y0_cp[53]}\n"
        f"delta_4_c = {y0_cp[54]}\n"
        f"delta_4_p = {y0_cp[55]}\n"
    )


def poc_solve_and_plot_ap_demo_cp_with_or_without_converstion_in_four_zones_and_smaller_cfr_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    ap_params=smallpox_param_sets["with_conversion_literature_cfr"],
    demo_params=smallpox_param_sets["with_conversion_literature_cfr"],
    cp_params=measles_seir_params_with_lower_cfr_for_c_and_starks_conversion,
    plot=False,
    plot_alive_christians=False,
    conversion=None
):
    # The Antonine Plague part:
    y0_ap = [
        initial_populations_in_zones["initial_christian_population_zone_1"] - 1,  # s1_c
        0,  # e1_c
        1,  # i1_c
        0,  # r1_c
        0,  # d1_c
        0,  # a1_c
        initial_populations_in_zones["initial_pagan_population_zone_1"] - 1,  # s1_p
        0,  # e1_p
        1,  # i1_p
        0,  # r1_p
        0,  # d1_p
        0,  # a1_p
        initial_populations_in_zones["initial_christian_population_zone_2"] - 1,  # s2_c
        0,  # e2_c
        1,  # i2_c
        0,  # r2_c
        0,  # d2_c
        0,  # a2_c
        initial_populations_in_zones["initial_pagan_population_zone_2"] - 1,  # s2_p
        0,  # e2_p
        1,  # i2_p
        0,  # r2_p
        0,  # d2_p
        0,  # a2_p
        initial_populations_in_zones["initial_christian_population_zone_3"] - 1,  # s3_c
        0,  # e3_c
        1,  # i3_c
        0,  # r3_c
        0,  # d3_c
        0,  # a3_c
        initial_populations_in_zones["initial_pagan_population_zone_3"] - 1,  # s3_p
        0,  # e3_p
        1,  # i3_p
        0,  # r3_p
        0,  # d3_p
        0,  # a3_p
        initial_populations_in_zones["initial_christian_population_zone_4"] - 1,  # s4_c
        0,  # e4_c
        1,  # i4_c
        0,  # r4_c
        0,  # d4_c
        0,  # a4_c
        initial_populations_in_zones["initial_pagan_population_zone_4"] - 1,  # s4_p
        0,  # e4_p
        1,  # i4_p
        0,  # r4_p
        0,  # d4_p
        0,  # a4_p
        zeleners_initial_delta_1,  # delta_1_c
        zeleners_initial_delta_1,  # delta_1_p
        zeleners_initial_delta_2,  # delta_2_c
        zeleners_initial_delta_2,  # delta_2_p
        zeleners_initial_delta_3,  # delta_3_c
        zeleners_initial_delta_3,  # delta_3_p
        zeleners_initial_delta_4,  # delta_4_c
        zeleners_initial_delta_4,  # delta_4_p
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.linspace(0, total_days_ap, total_days_ap + 1)

    # Solve the ODE for the Antonine Plague
    # TODO: implement the seir function with some delta usage - a new one, tho!
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(
                y, t, ap_params
            )
        )

    print(y0_ap)
    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic + 1)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_to_dense_zones(
            y, t, demo_params
        )

    print(y0_demographic)
    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    # TODO: check if this extension is not a mistake
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p

    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(
            y, t, cp_params
        )

    print(y0_cp)
    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp
    )


    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    if plot:
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=compartment_indices,
            compartment_labels=compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
        )

        # Function to plot the totals of compartments
        def plot_total_compartments():
            total_compartment_labels = [
                "Susceptible Christians",
                "Exposed Christians",
                "Infected Christians",
                "Recovered Christians",
                # "Deceased Christians",
                # "Susceptible Pagans",
                # "Exposed Pagans",
                # "Infected Pagans",
                # "Recovered Pagans",
                # "Deceased Pagans",
            ]

            # Calculate the total compartments for each group
            susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
            exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
            infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
            recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
            deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

            susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
            exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
            infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
            recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
            deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

            total_compartments = [
                susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
                susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
            ]

            # Plot the totals
            plot_seir_model(
                solution_cp,
                t_cp,
                start_year=end_year_demographic,
                end_year=end_year_cp,
                compartment_indices=range(len(total_compartments)),
                compartment_labels=total_compartment_labels,
                every_nth_year=5,
                y_tick_interval=100_000,
                display_y_label_every_n_ticks=10,
                plot_title="Total compartments across all zones",
            )

        # Call the function to plot totals
        plot_total_compartments()

    if plot_alive_christians:
        ap_disease = ap_params.disease_name_full
        cp_disease = cp_params.disease_name_full
        plot_alive_christians_full_timeline_clean(
            solutions=[solution_ap, solution_demographic, solution_cp],
            time_segments=[t_ap, t_demographic, t_cp],
            scenario_labels=["Antonine Plague", "Post-plague growth", "Cyprianic Plague"],
            plot_title=f"Alive Christian Population (AP: {ap_disease}, CP: {cp_disease},\nconversion: {'Enabled' if conversion else 'Disabled'}, CFR: Stark, Starting total population: 150M)"
        )

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    alive_total = alive1_c + alive1_p + alive2_c + alive2_p + alive3_c + alive3_p + alive4_c + alive4_p
    alive_c = alive1_c + alive2_c + alive3_c + alive4_c
    alive_p = alive1_p + alive2_p + alive3_p + alive4_p
    alive_c_percentage_of_total = (alive_c / alive_total) * 100

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}\n"
        f"\n"
        f"alive_total = {alive_total}\n"
        f"alive_c = {alive_c}\n"
        f"alive_p = {alive_p}\n"
        f"alive_c_percentage_of_total = {alive_c_percentage_of_total} %\n"
    )

    print("Length of t_cp:", len(t_cp))
    print("Length of solution_cp.t:", len(solution_cp.t))
    print("solution_cp.t[-1]:", solution_cp.t[-1])

    for i, label in enumerate(compartment_labels):
        max_val = np.max(solution_cp.y[compartment_indices[i], :])
        print(f"{label}: max = {max_val}")

    print("Solver success:", solution_cp.success)
    print("Solver message:", solution_cp.message)

    print(
        "final deltas:\n"
        f"delta_1_c = {y0_cp[48]}\n"
        f"delta_1_p = {y0_cp[49]}\n"
        f"delta_2_c = {y0_cp[50]}\n"
        f"delta_2_p = {y0_cp[51]}\n"
        f"delta_3_c = {y0_cp[52]}\n"
        f"delta_3_p = {y0_cp[53]}\n"
        f"delta_4_c = {y0_cp[54]}\n"
        f"delta_4_p = {y0_cp[55]}\n"
    )
    result = {
        "alive_total": alive_total,
        "alive_c": alive_c,
        "alive_p": alive_p,
        "alive_c_percentage_of_total": alive_c_percentage_of_total,
    }
    return result


def table_2356_runs(output_path="table_2_results", with_timestamp=False):
    """
    Run simulations for:
    Tables 2, 3, 5, and 6 Summary of model output for Antonine Plague as smallpox and a different disease for Cyprianic Plague.
    """
    if with_timestamp:
        timestamp = time.time()
        output_path = f"{output_path}_{timestamp}.csv"
    else:
        output_path = f"{output_path}.csv"

    ap_params = {
        "Smallpox": smallpox_param_sets,
        "Measles": measles_param_sets
    }

    diseases = {
        "Smallpox": smallpox_param_sets,
        "Measles": measles_param_sets,
        "Crimean–Congo hemorrhagic fever": cchf_param_sets,
        "Ebola virus disease": evd_param_sets,
        "Lassa fever": lassa_param_sets
    }

    results = []

    for ap_name, ap_param_set in ap_params.items():
        for disease_name, cp_param_set in diseases.items():
            # WITH conversion
            try:
                cp_with = deepcopy(cp_param_set["with_conversion_literature_cfr"])
                demo_with = deepcopy(ap_param_set["with_conversion_literature_cfr"])
                ap_with = deepcopy(ap_param_set["with_conversion_literature_cfr"])

                result_with = poc_solve_and_plot_ap_demo_cp_with_or_without_converstion_in_four_zones_and_smaller_cfr_for_christians(
                        ap_params=ap_with,
                        demo_params=demo_with,
                        cp_params=cp_with,
                        conversion=True,
                        # plot=True,
                        plot_alive_christians=True
                    )
                alive_total_with = result_with["alive_total"]
                alive_c_with = result_with["alive_c"]
                alive_p_with = result_with["alive_p"]
                percent_with = result_with["alive_c_percentage_of_total"]
            except Exception:
                alive_c_with = percent_with = None

            # WITHOUT conversion
            try:
                cp_without = deepcopy(cp_param_set["without_conversion_literature_cfr"])
                demo_without = deepcopy(ap_param_set["without_conversion_literature_cfr"])
                ap_without = deepcopy(ap_param_set["without_conversion_literature_cfr"])

                result_without = poc_solve_and_plot_ap_demo_cp_with_or_without_converstion_in_four_zones_and_smaller_cfr_for_christians(
                    ap_params=ap_without,
                    demo_params=demo_without,
                    cp_params=cp_without,
                    conversion=False,
                    # plot=False,
                    plot_alive_christians=True
                )

                alive_total_without = result_without["alive_total"]
                alive_c_without = result_without["alive_c"]
                alive_p_without = result_without["alive_p"]
                percent_without = result_without["alive_c_percentage_of_total"]

            except Exception:
                alive_c_without = percent_without = None

            # Compute differences
            try:
                ratio_increase = percent_with - percent_without
            except:
                ratio_increase = 0.00

            try:
                if alive_c_with and alive_c_without:
                    pop_increase = (1 - (alive_c_without / alive_c_with)) * 100
                else:
                    pop_increase = "#DIV/0!"
            except:
                pop_increase = "#DIV/0!"

            results.append([
                ap_name,
                disease_name,
                int(alive_c_with) if alive_c_with is not None else "",
                f"{percent_with:}" if percent_with is not None else "",
                int(alive_c_without) if alive_c_without is not None else "",
                f"{percent_without}" if percent_without is not None else "",
                f"{ratio_increase}",
                f"{pop_increase}"
            ])

    columns = [
        "Antonine Plague model",
        "Cyprianic Plague model",
        "Size of Christian subpopulation (individuals)",
        "Total Christians (compared to total population; percent)",
        "Size of Chistian subpopulation without conversion (individuals)",
        "Total Christians without conversion (compared to total population; percent)",
        "Christian subpopulation ratio increase (compared to modeled baseline without conversion; percent points)",
        "Christian subpopulation increase (compared to modeled baseline without conversion; percent)"
    ]

    df = pd.DataFrame(results, columns=columns)

    print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False) + "\n")
    df.to_csv(output_path, index=False)
    print(f"[Saved] CSV output written to: {os.path.abspath(output_path)}")

    return df


def table_47_runs(output_path="table_4_results", with_timestamp=False):
    """
    Run simulations for:
    Tables 4 and 7 Summary of model output for Antonine Plague as smallpox or measles and a different disease for Cyprianic Plague.
    """
    if with_timestamp:
        timestamp = time.time()
        output_path = f"{output_path}_{timestamp}.csv"
    else:
        output_path = f"{output_path}.csv"

    ap_params = {
        "Smallpox": smallpox_param_sets,
        "Measles": measles_param_sets
    }
    diseases = {
        "Smallpox": smallpox_param_sets,
        "Measles": measles_param_sets,
        "Crimean–Congo hemorrhagic fever": cchf_param_sets,
        "Ebola virus disease": evd_param_sets,
        "Lassa fever": lassa_param_sets
    }

    results = []

    for ap_name, ap_param_set in ap_params.items():
        for disease_name, cp_param_set in diseases.items():
            # WITH conversion
            try:
                cp_with = deepcopy(cp_param_set["with_conversion_hardcoded_cfr"])
                demo_with = deepcopy(ap_param_set["with_conversion_hardcoded_cfr"])
                ap_with = deepcopy(ap_param_set["with_conversion_hardcoded_cfr"])

                result_with = poc_solve_and_plot_ap_demo_cp_with_or_without_converstion_in_four_zones_and_smaller_cfr_for_christians(
                        ap_params=ap_with,
                        demo_params=demo_with,
                        cp_params=cp_with,
                        conversion=True,
                        # plot=False,
                        plot_alive_christians=True
                    )
                alive_total_with = result_with["alive_total"]
                alive_c_with = result_with["alive_c"]
                alive_p_with = result_with["alive_p"]
                percent_with = result_with["alive_c_percentage_of_total"]
            except Exception:
                alive_c_with = percent_with = None

            # WITHOUT conversion
            try:
                cp_without = deepcopy(cp_param_set["without_conversion_hardcoded_cfr"])
                demo_without = deepcopy(ap_param_set["without_conversion_hardcoded_cfr"])
                ap_without = deepcopy(ap_param_set["without_conversion_hardcoded_cfr"])

                result_without = poc_solve_and_plot_ap_demo_cp_with_or_without_converstion_in_four_zones_and_smaller_cfr_for_christians(
                    ap_params=ap_without,
                    demo_params=demo_without,
                    cp_params=cp_without,
                    conversion=False,
                    # plot=False,
                    plot_alive_christians=True
                )

                alive_total_without = result_without["alive_total"]
                alive_c_without = result_without["alive_c"]
                alive_p_without = result_without["alive_p"]
                percent_without = result_without["alive_c_percentage_of_total"]

            except Exception:
                alive_c_without = percent_without = None

            # Compute differences
            try:
                ratio_increase = percent_with - percent_without
            except:
                ratio_increase = 0.00

            try:
                if alive_c_with and alive_c_without:
                    pop_increase = (1 - (alive_c_without / alive_c_with)) * 100
                else:
                    pop_increase = "#DIV/0!"
            except:
                pop_increase = "#DIV/0!"

            results.append([
                ap_name,
                disease_name,
                int(alive_c_with) if alive_c_with is not None else "",
                f"{percent_with:}" if percent_with is not None else "",
                int(alive_c_without) if alive_c_without is not None else "",
                f"{percent_without}" if percent_without is not None else "",
                f"{ratio_increase}",
                f"{pop_increase}"
            ])

    columns = [
        "Antonine Plague model",
        "Cyprianic Plague model",
        "Size of Christian subpopulation (individuals)",
        "Total Christians (compared to total population; percent)",
        "Size of Chistian subpopulation without conversion (individuals)",
        "Total Christians without conversion (compared to total population; percent)",
        "Christian subpopulation ratio increase (compared to modeled baseline without conversion; percent points)",
        "Christian subpopulation increase (compared to modeled baseline without conversion; percent)"
    ]

    df = pd.DataFrame(results, columns=columns)

    print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False) + "\n")
    df.to_csv(output_path, index=False)
    print(f"[Saved] CSV output written to: {os.path.abspath(output_path)}")

    return df


def plot_alive_christians_full_timeline_clean(solutions, time_segments, scenario_labels=None,
                                        start_year=165, end_year=270,
                                        x_label="Year CE",
                                        y_label="Alive Christians",
                                        plot_title="Alive Christian Population (Antonine to Cyprianic Plagues)"):

    print("\n\nGot into the plot_alive_christians_full_timeline_clean function\n\n")

    plt.figure(figsize=(14, 6))
    total_years = []
    total_population = []

    for i, (sol, t) in enumerate(zip(solutions, time_segments)):
        alive_christians = (
            sol.y[0, :] + sol.y[1, :] + sol.y[2, :] + sol.y[3, :] +
            sol.y[12, :] + sol.y[13, :] + sol.y[14, :] + sol.y[15, :] +
            sol.y[24, :] + sol.y[25, :] + sol.y[26, :] + sol.y[27, :] +
            sol.y[36, :] + sol.y[37, :] + sol.y[38, :] + sol.y[39, :]
        )

        t_norm = t - t[0]
        if i == 0:
            years = start_year + t_norm / 365.0  # Convert daily to yearly offset
        else:
            prev_end = total_years[-1]
            years = prev_end + t_norm / 365.0

        total_years.extend(years)
        total_population.extend(alive_christians)

        label = scenario_labels[i] if scenario_labels and i < len(scenario_labels) else None
        plt.plot(years, alive_christians, linewidth=2, label=label)

    if total_population:
        ax = plt.gca()
        ax.set_title(plot_title, fontsize=14, weight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.6, which='major', axis='both')

        ax.set_xlim(start_year, end_year)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)

        ymax = max(total_population)

        # Aim for 8–12 ticks
        raw_step = ymax / 10
        # Round step to nearest 1000 or 5000
        if raw_step <= 1000:
            step = 1000
        elif raw_step <= 5000:
            step = 5000
        else:
            step = int(round(raw_step / 1000.0)) * 1000

        # Round ymax up to nearest multiple of step
        ymax_rounded = ((int(ymax) // step) + 1) * step

        ax.set_ylim(0, ymax_rounded)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))

        ax.grid(which='major', linestyle='--', alpha=0.6, axis='y')

        if scenario_labels:
            ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No data to plot: solutions or time arrays may be empty.")


if __name__ == "__main__":
    table_2356_runs(output_path="tables_5-6_150m_results", with_timestamp=True)
    table_47_runs(output_path="table_7_150m_results", with_timestamp=True)
