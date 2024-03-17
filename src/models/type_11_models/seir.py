def direct_transmission_over_two_connected_subpopulations_seird_model(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate = parameters.fatality_rate
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations and whole population
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p
    n = n_c + n_p

    # Christian compartments:
    ds_c = b * n_c - beta * s_c * (i_c + i_p) / n - d * s_c
    de_c = beta * s_c * (i_c + i_p) / n - sigma * e_c - d * e_c
    di_c = sigma * e_c - gamma * i_c - d * i_c
    dr_c = (1 - fatality_rate) * gamma * i_c - d * r_c
    dd_c = fatality_rate * gamma * i_c
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * (i_c + i_p) / n - d * s_p
    de_p = beta * s_p * (i_c + i_p) / n - sigma * e_p - d * e_p
    di_p = sigma * e_p - gamma * i_p - d * i_p
    dr_p = (1 - fatality_rate) * gamma * i_p - d * r_p
    dd_p = fatality_rate * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_one_population_as_in_plos_paper(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate = parameters.fatality_rate
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations and whole population
    n_p = s_p + e_p + i_p + r_p

    # Christian compartments:
    ds_c = 0
    de_c = 0
    di_c = 0
    dr_c = 0
    dd_c = 0
    da_c = 0

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * i_p / n_p - d * s_p
    de_p = beta * s_p * i_p / n_p - sigma * e_p - d * e_p
    di_p = sigma * e_p - gamma * i_p - d * i_p
    dr_p = (1 - fatality_rate) * gamma * i_p - d * r_p
    dd_p = fatality_rate * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]
