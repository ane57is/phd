from src.parameters.params import (
    zeleners_initial_delta_1,
    zeleners_initial_delta_2,
    zeleners_initial_delta_3,
    zeleners_initial_delta_4,
    adjust_delta
)

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


def simple_demographic_model(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p

    # Christian compartments:
    ds_c = b * n_c - d * s_c
    de_c = -d * e_c
    di_c = -d * i_c
    dr_c = -d * r_c
    dd_c = 0
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - d * s_p
    de_p = -d * e_p
    di_p = -d * i_p
    dr_p = -d * r_p
    dd_p = 0
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_two_connected_subpopulations_seird_model with two CFRs.
    """

    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
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
    dr_c = (1 - fatality_rate_c) * gamma * i_c - d * r_c
    dd_c = fatality_rate_c * gamma * i_c
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * (i_c + i_p) / n - d * s_p
    de_p = beta * s_p * (i_c + i_p) / n - sigma * e_p - d * e_p
    di_p = sigma * e_p - gamma * i_p - d * i_p
    dr_p = (1 - fatality_rate_p) * gamma * i_p - d * r_p
    dd_p = fatality_rate_p * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def simple_demographic_model_with_conversion(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p

    # Pagans converted to Christianity for all living compartments:
    converted_s = conversion_rate * s_c
    converted_e = conversion_rate * e_c
    converted_i = conversion_rate * i_c
    converted_r = conversion_rate * r_c

    # Christian compartments:
    ds_c = b * n_c - d * s_c + converted_s
    de_c = -d * e_c + converted_e
    di_c = -d * i_c + converted_i
    dr_c = -d * r_c + converted_r
    dd_c = 0
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - d * s_p - converted_s
    de_p = -d * e_p - converted_e
    di_p = -d * i_p - converted_i
    dr_p = -d * r_p - converted_r
    dd_p = 0
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model with conversion.
    """

    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations and whole population
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p
    n = n_c + n_p

    # Pagans converted to Christianity for all living compartments:
    converted_s = conversion_rate * s_c
    converted_e = conversion_rate * e_c
    converted_i = conversion_rate * i_c
    converted_r = conversion_rate * r_c

    # Christian compartments:
    ds_c = b * n_c - beta * s_c * (i_c + i_p) / n - d * s_c + converted_s
    de_c = beta * s_c * (i_c + i_p) / n - sigma * e_c - d * e_c + converted_e
    di_c = sigma * e_c - gamma * i_c - d * i_c + converted_i
    dr_c = (1 - fatality_rate_c) * gamma * i_c - d * r_c + converted_r
    dd_c = fatality_rate_c * gamma * i_c
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * (i_c + i_p) / n - d * s_p - converted_s
    de_p = beta * s_p * (i_c + i_p) / n - sigma * e_p - d * e_p - converted_e
    di_p = sigma * e_p - gamma * i_p - d * i_p - converted_i
    dr_p = (1 - fatality_rate_p) * gamma * i_p - d * r_p - converted_r
    dd_p = fatality_rate_p * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model by
    four zones. Each zone has a pair of Christian and Pagan subpopulations. Each pair of subpopulations has a conversion
    rate mechanism. Within pairs of connected subpopulations (inside of four separate zones) a dynamic interaction rate
    modifies the impact of beta (the transmission rate) by the sum of these subpopulations (the total population of
    a zone).
    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p
    ) = y

    beta = parameters.beta
    delta = None  # TODO: implement calculation of delta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations in each zone and whole population
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p
    n1 = n1_c + n1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p
    n2 = n2_c + n2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p
    n3 = n3_c + n3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p
    n4 = n4_c + n4_p

    # TODO: not sure if there is a need for this n variable as long as there is no interaction between zones 🤷🤔
    n = n1 + n2 + n3 + n4

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = conversion_rate * s1_p
    converted_e1 = conversion_rate * e1_p
    converted_i1 = conversion_rate * i1_p
    converted_r1 = conversion_rate * r1_p

    converted_s2 = conversion_rate * s2_p
    converted_e2 = conversion_rate * e2_p
    converted_i2 = conversion_rate * i2_p
    converted_r2 = conversion_rate * r2_p

    converted_s3 = conversion_rate * s3_p
    converted_e3 = conversion_rate * e3_p
    converted_i3 = conversion_rate * i3_p
    converted_r3 = conversion_rate * r3_p

    converted_s4 = conversion_rate * s4_p
    converted_e4 = conversion_rate * e4_p
    converted_i4 = conversion_rate * i4_p
    converted_r4 = conversion_rate * r4_p

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = b * n1_c - beta * s1_c * (i1_c + i1_p) / n1 - d * s1_c + converted_s1
    de1_c = beta * s1_c * (i1_c + i1_p) / n1 - sigma * e1_c - d * e1_c + converted_e1
    di1_c = sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1
    dr1_c = (1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1
    dd1_c = fatality_rate_c * gamma * i1_c
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - beta * s1_p * (i1_c + i1_p) / n1 - d * s1_p - converted_s1
    de1_p = beta * s1_p * (i1_c + i1_p) / n1 - sigma * e1_p - d * e1_p - converted_e1
    di1_p = sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1
    dr1_p = (1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1
    dd1_p = fatality_rate_p * gamma * i1_p
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - beta * s2_c * (i2_c + i2_p) / n2 - d * s2_c + converted_s2
    de2_c = beta * s2_c * (i2_c + i2_p) / n2 - sigma * e2_c - d * e2_c + converted_e2
    di2_c = sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2
    dr2_c = (1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r2
    dd2_c = fatality_rate_c * gamma * i2_c
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - beta * s2_p * (i2_c + i2_p) / n2 - d * s2_p - converted_s2
    de2_p = beta * s2_p * (i2_c + i2_p) / n2 - sigma * e2_p - d * e2_p - converted_e2
    di2_p = sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2
    dr2_p = (1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2
    dd2_p = fatality_rate_p * gamma * i2_p
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - beta * s3_c * (i3_c + i3_p) / n3 - d * s3_c + converted_s3
    de3_c = beta * s3_c * (i3_c + i3_p) / n3 - sigma * e3_c - d * e3_c + converted_e3
    di3_c = sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3
    dr3_c = (1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r3
    dd3_c = fatality_rate_c * gamma * i3_c
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - beta * s3_p * (i3_c + i3_p) / n3 - d * s3_p - converted_s3
    de3_p = beta * s3_p * (i3_c + i3_p) / n3 - sigma * e3_p - d * e3_p - converted_e3
    di3_p = sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3
    dr3_p = (1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3
    dd3_p = fatality_rate_p * gamma * i3_p
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - beta * s4_c * (i4_c + i4_p) / n4 - d * s4_c + converted_s4
    de4_c = beta * s4_c * (i4_c + i4_p) / n4 - sigma * e4_c - d * e4_c + converted_e4
    di4_c = sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4
    dr4_c = (1 - fatality_rate_c) * gamma * i4_c - d * r4_c + converted_r4
    dd4_c = fatality_rate_c * gamma * i4_c
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - beta * s4_p * (i4_c + i4_p) / n4 - d * s4_p - converted_s4
    de4_p = beta * s4_p * (i4_c + i4_p) / n4 - sigma * e4_p - d * e4_p - converted_e4
    di4_p = sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4
    dr4_p = (1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4
    dd4_p = fatality_rate_p * gamma * i4_p
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]


def simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(y, t, parameters):
    """
    Extension of def simple_demographic_model_with_conversion by four zones.
    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p
    ) = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations in each zone
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    def non_negative(value):
        # return max(0, value)
        return value

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = non_negative(b * n1_c - d * s1_c + converted_s1)
    de1_c = non_negative(-d * e1_c + converted_e1)
    di1_c = non_negative(-d * i1_c + converted_i1)
    dr1_c = non_negative(-d * r1_c + converted_r1)
    dd1_c = 0
    da1_c = non_negative(d * n1_c)

    # Pagan compartments in Zone 1
    ds1_p = non_negative(b * n1_p - d * s1_p - converted_s1)
    de1_p = non_negative(-d * e1_p - converted_e1)
    di1_p = non_negative(-d * i1_p - converted_i1)
    dr1_p = non_negative(-d * r1_p - converted_r1)
    dd1_p = 0
    da1_p = non_negative(d * n1_p)

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = non_negative(b * n2_c - d * s2_c + converted_s2)
    de2_c = non_negative(-d * e2_c + converted_e2)
    di2_c = non_negative(-d * i2_c + converted_i2)
    dr2_c = non_negative(-d * r2_c + converted_r2)
    dd2_c = 0
    da2_c = non_negative(d * n2_c)

    # Pagan compartments in Zone 2
    ds2_p = non_negative(b * n2_p - d * s2_p - converted_s2)
    de2_p = non_negative(-d * e2_p - converted_e2)
    di2_p = non_negative(-d * i2_p - converted_i2)
    dr2_p = non_negative(-d * r2_p - converted_r2)
    dd2_p = 0
    da2_p = non_negative(d * n2_p)

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = non_negative(b * n3_c - d * s3_c + converted_s3)
    de3_c = non_negative(-d * e3_c + converted_e3)
    di3_c = non_negative(-d * i3_c + converted_i3)
    dr3_c = non_negative(-d * r3_c + converted_r3)
    dd3_c = 0
    da3_c = non_negative(d * n3_c)

    # Pagan compartments in Zone 3
    ds3_p = non_negative(b * n3_p - d * s3_p - converted_s3)
    de3_p = non_negative(-d * e3_p - converted_e3)
    di3_p = non_negative(-d * i3_p - converted_i3)
    dr3_p = non_negative(-d * r3_p - converted_r3)
    dd3_p = 0
    da3_p = non_negative(d * n3_p)

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = non_negative(b * n4_c - d * s4_c + converted_s4)
    de4_c = non_negative(-d * e4_c + converted_e4)
    di4_c = non_negative(-d * i4_c + converted_i4)
    dr4_c = non_negative(-d * r4_c + converted_r4)
    dd4_c = 0
    da4_c = non_negative(d * n4_c)

    # Pagan compartments in Zone 4
    ds4_p = non_negative(b * n4_p - d * s4_p - converted_s4)
    de4_p = non_negative(-d * e4_p - converted_e4)
    di4_p = non_negative(-d * i4_p - converted_i4)
    dr4_p = non_negative(-d * r4_p - converted_r4)
    dd4_p = 0
    da4_p = non_negative(d * n4_p)

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]


def direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model
    by a dynamic delta (interaction rate). The current delta values are stored in the returned list of derivatives.

    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p,
        delta_1_c, delta_1_p, delta_2_c, delta_2_p, delta_3_c, delta_3_p, delta_4_c, delta_4_p
    ) = y
    if t % 365 == 0:  # Example: print yearly updates
        print(f"Time {t}: delta_1_c={delta_1_c}, delta_1_p={delta_1_p}")

    # if t % (365 * 10) == 0:  # Log every 10 years
    #     print(f"Time {t}: delta_1_c={delta_1_c}, delta_1_p={delta_1_p}")

    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    # unified_deltas = parameters.unified_deltas
    unified_deltas = False
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily
    max_delta = parameters.max_delta

    if unified_deltas:
        updated_delta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c + s1_p, i_n=i1_c + i1_p, r_n=r1_c + r1_p
        )
        updated_delta_1_p = updated_delta_1_c

        updated_delta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c + s2_p, i_n=i2_c + i2_p, r_n=r2_c + r2_p
        )
        updated_delta_2_p = updated_delta_2_c

        updated_delta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c + s3_p, i_n=i3_c + i3_p, r_n=r3_c + r3_p
        )
        updated_delta_3_p = updated_delta_3_c

        updated_delta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c + s4_p, i_n=i4_c + i4_p, r_n=r4_c + r4_p
        )
        updated_delta_4_p = updated_delta_4_c

    else:
        updated_delta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c, i_n=i1_c, r_n=r1_c
        )
        updated_delta_1_p = adjust_delta(
            delta_prev=delta_1_p, delta_0=zeleners_initial_delta_1, s_n=s1_p, i_n=i1_p, r_n=r1_p
        )
        updated_delta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c, i_n=i2_c, r_n=r2_c
        )
        updated_delta_2_p = adjust_delta(
            delta_prev=delta_2_p, delta_0=zeleners_initial_delta_2, s_n=s2_p, i_n=i2_p, r_n=r2_p
        )
        updated_delta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c, i_n=i3_c, r_n=r3_c
        )
        updated_delta_3_p = adjust_delta(
            delta_prev=delta_3_p, delta_0=zeleners_initial_delta_3, s_n=s3_p, i_n=i3_p, r_n=r3_p
        )
        updated_delta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c, i_n=i4_c, r_n=r4_c
        )
        updated_delta_4_p = adjust_delta(
            delta_prev=delta_4_p, delta_0=zeleners_initial_delta_4, s_n=s4_p, i_n=i4_p, r_n=r4_p
        )

    # Totals of subpopulations in each zone and whole population
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p
    # n1 = n1_c + n1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p
    # n2 = n2_c + n2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p
    # n3 = n3_c + n3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p
    # n4 = n4_c + n4_p

    # TODO: not sure if there is a need for this n variable as long as there is no interaction between zones 🤷🤔
    # n = n1 + n2 + n3 + n4

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    def non_negative(value):
        return max(0, value)
        # return value
        # return max(0, min(value, 1e10))  # Cap at 10 billion
        # return max(0, min(value, 10_000_000))  # Cap at 10 billion

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = non_negative(b * n1_c - beta * updated_delta_1_c * (i1_c + i1_p) / max_delta - d * s1_c + converted_s1)
    de1_c = non_negative(beta * updated_delta_1_c * (i1_c + i1_p) / max_delta - sigma * e1_c - d * e1_c + converted_e1)
    di1_c = non_negative(sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1)
    dr1_c = non_negative((1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1)
    dd1_c = non_negative(fatality_rate_c * gamma * i1_c)
    da1_c = non_negative(d * n1_c)

    # Pagan compartments in Zone 1
    ds1_p = non_negative(b * n1_p - beta * updated_delta_1_p * (i1_c + i1_p) / max_delta - d * s1_p - converted_s1)
    de1_p = non_negative(beta * updated_delta_1_p * (i1_c + i1_p) / max_delta - sigma * e1_p - d * e1_p - converted_e1)
    di1_p = non_negative(sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1)
    dr1_p = non_negative((1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1)
    dd1_p = non_negative(fatality_rate_p * gamma * i1_p)
    da1_p = non_negative(d * n1_p)

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = non_negative(b * n2_c - beta * updated_delta_2_c * (i2_c + i2_p) / max_delta - d * s2_c + converted_s2)
    de2_c = non_negative(beta * updated_delta_2_c * (i2_c + i2_p) / max_delta - sigma * e2_c - d * e2_c + converted_e2)
    di2_c = non_negative(sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2)
    dr2_c = non_negative((1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r2)
    dd2_c = non_negative(fatality_rate_c * gamma * i2_c)
    da2_c = non_negative(d * n2_c)

    # Pagan compartments in Zone 2
    ds2_p = non_negative(b * n2_p - beta * updated_delta_2_p * (i2_c + i2_p) / max_delta - d * s2_p - converted_s2)
    de2_p = non_negative(beta * updated_delta_2_p * (i2_c + i2_p) / max_delta - sigma * e2_p - d * e2_p - converted_e2)
    di2_p = non_negative(sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2)
    dr2_p = non_negative((1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2)
    dd2_p = non_negative(fatality_rate_p * gamma * i2_p)
    da2_p = non_negative(d * n2_p)

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = non_negative(b * n3_c - beta * updated_delta_3_c * (i3_c + i3_p) / max_delta - d * s3_c + converted_s3)
    de3_c = non_negative(beta * updated_delta_3_c * (i3_c + i3_p) / max_delta - sigma * e3_c - d * e3_c + converted_e3)
    di3_c = non_negative(sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3)
    dr3_c = non_negative((1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r3)
    dd3_c = non_negative(fatality_rate_c * gamma * i3_c)
    da3_c = non_negative(d * n3_c)

    # Pagan compartments in Zone 3
    ds3_p = non_negative(b * n3_p - beta * updated_delta_3_p * (i3_c + i3_p) / max_delta - d * s3_p - converted_s3)
    de3_p = non_negative(beta * updated_delta_3_p * (i3_c + i3_p) / max_delta - sigma * e3_p - d * e3_p - converted_e3)
    di3_p = non_negative(sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3)
    dr3_p = non_negative((1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3)
    dd3_p = non_negative(fatality_rate_p * gamma * i3_p)
    da3_p = non_negative(d * n3_p)

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = non_negative(b * n4_c - beta * updated_delta_4_c * (i4_c + i4_p) / max_delta - d * s4_c + converted_s4)
    de4_c = non_negative(beta * updated_delta_4_c * (i4_c + i4_p) / max_delta - sigma * e4_c - d * e4_c + converted_e4)
    di4_c = non_negative(sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4)
    dr4_c = non_negative((1 - fatality_rate_c) * gamma * i4_c - d * r4_c + converted_r4)
    dd4_c = non_negative(fatality_rate_c * gamma * i4_c)
    da4_c = non_negative(d * n4_c)

    # Pagan compartments in Zone 4
    ds4_p = non_negative(b * n4_p - beta * updated_delta_4_p * (i4_c + i4_p) / max_delta - d * s4_p - converted_s4)
    de4_p = non_negative(beta * updated_delta_4_p * (i4_c + i4_p) / max_delta - sigma * e4_p - d * e4_p - converted_e4)
    di4_p = non_negative(sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4)
    dr4_p = non_negative((1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4)
    dd4_p = non_negative(fatality_rate_p * gamma * i4_p)
    da4_p = non_negative(d * n4_p)

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p,
        updated_delta_1_c, updated_delta_1_p, updated_delta_2_c, updated_delta_2_p,
        updated_delta_3_c, updated_delta_3_p, updated_delta_4_c, updated_delta_4_p
    ]


def direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(y, t, parameters):
    """
    New take on a seir model with interaction rate delta in 4 zones.

    Basing it on
    def direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model
    """
    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p,
        delta_1_c, delta_1_p, delta_2_c, delta_2_p, delta_3_c, delta_3_p, delta_4_c, delta_4_p
    ) = y

    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    unified_deltas = parameters.unified_deltas
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily
    max_delta = parameters.max_delta

    epsilon = 1e-4  # Small value to prevent division by zero

    if unified_deltas:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c + s1_p, i_n=i1_c + i1_p, r_n=r1_c + r1_p
        )
        ddelta_1_p = ddelta_1_c

        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c + s2_p, i_n=i2_c + i2_p, r_n=r2_c + r2_p
        )
        ddelta_2_p = ddelta_2_c

        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c + s3_p, i_n=i3_c + i3_p, r_n=r3_c + r3_p
        )
        ddelta_3_p = ddelta_3_c

        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c + s4_p, i_n=i4_c + i4_p, r_n=r4_c + r4_p
        )
        ddelta_4_p = ddelta_4_c

    else:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c, i_n=i1_c, r_n=r1_c
        )
        ddelta_1_p = adjust_delta(
            delta_prev=delta_1_p, delta_0=zeleners_initial_delta_1, s_n=s1_p, i_n=i1_p, r_n=r1_p
        )
        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c, i_n=i2_c, r_n=r2_c
        )
        ddelta_2_p = adjust_delta(
            delta_prev=delta_2_p, delta_0=zeleners_initial_delta_2, s_n=s2_p, i_n=i2_p, r_n=r2_p
        )
        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c, i_n=i3_c, r_n=r3_c
        )
        ddelta_3_p = adjust_delta(
            delta_prev=delta_3_p, delta_0=zeleners_initial_delta_3, s_n=s3_p, i_n=i3_p, r_n=r3_p
        )
        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c, i_n=i4_c, r_n=r4_c
        )
        ddelta_4_p = adjust_delta(
            delta_prev=delta_4_p, delta_0=zeleners_initial_delta_4, s_n=s4_p, i_n=i4_p, r_n=r4_p
        )

    # Totals of subpopulations in each zone and whole population
    n1_c = max(s1_c + e1_c + i1_c + r1_c, epsilon)
    n1_p = max(s1_p + e1_p + i1_p + r1_p, epsilon)
    n1 = max(n1_c + n1_p, epsilon)

    n2_c = max(s2_c + e2_c + i2_c + r2_c, epsilon)
    n2_p = max(s2_p + e2_p + i2_p + r2_p, epsilon)
    n2 = max(n2_c + n2_p, epsilon)

    n3_c = max(s3_c + e3_c + i3_c + r3_c, epsilon)
    n3_p = max(s3_p + e3_p + i3_p + r3_p, epsilon)
    n3 = max(n3_c + n3_p, epsilon)

    n4_c = max(s4_c + e4_c + i4_c + r4_c, epsilon)
    n4_p = max(s4_p + e4_p + i4_p + r4_p, epsilon)
    n4 = max(n4_c + n4_p, epsilon)

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_c, s1_p, 0)
    converted_e1 = min(conversion_rate * e1_c, e1_p, 0)
    converted_i1 = min(conversion_rate * i1_c, i1_p, 0)
    converted_r1 = min(conversion_rate * r1_c, r1_p, 0)

    converted_s2 = min(conversion_rate * s2_c, s2_p, 0)
    converted_e2 = min(conversion_rate * e2_c, e2_p, 0)
    converted_i2 = min(conversion_rate * i2_c, i2_p, 0)
    converted_r2 = min(conversion_rate * r2_c, r2_p, 0)

    converted_s3 = min(conversion_rate * s3_c, s3_p, 0)
    converted_e3 = min(conversion_rate * e3_c, e3_p, 0)
    converted_i3 = min(conversion_rate * i3_c, i3_p, 0)
    converted_r3 = min(conversion_rate * r3_c, r3_p, 0)

    converted_s4 = min(conversion_rate * s4_c, s4_p, 0)
    converted_e4 = min(conversion_rate * e4_c, e4_p, 0)
    converted_i4 = min(conversion_rate * i4_c, i4_p, 0)
    converted_r4 = min(conversion_rate * r4_c, r4_p, 0)

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = b * n1_c - (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - d * s1_c + converted_s1
    de1_c = (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - sigma * e1_c - d * e1_c + converted_e1
    di1_c = sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1
    dr1_c = (1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1
    dd1_c = fatality_rate_c * gamma * i1_c
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - d * s1_p - converted_s1
    de1_p = (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - sigma * e1_p - d * e1_p - converted_e1
    di1_p = sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1
    dr1_p = (1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1
    dd1_p = fatality_rate_p * gamma * i1_p
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - d * s2_c + converted_s2
    de2_c = (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - sigma * e2_c - d * e2_c + converted_e2
    di2_c = sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2
    dr2_c = (1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r2
    dd2_c = fatality_rate_c * gamma * i2_c
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - d * s2_p - converted_s2
    de2_p = (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - sigma * e2_p - d * e2_p - converted_e2
    di2_p = sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2
    dr2_p = (1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2
    dd2_p = fatality_rate_p * gamma * i2_p
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - d * s3_c + converted_s3
    de3_c = (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - sigma * e3_c - d * e3_c + converted_e3
    di3_c = sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3
    dr3_c = (1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r3
    dd3_c = fatality_rate_c * gamma * i3_c
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - d * s3_p - converted_s3
    de3_p = (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - sigma * e3_p - d * e3_p - converted_e3
    di3_p = sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3
    dr3_p = (1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3
    dd3_p = fatality_rate_p * gamma * i3_p
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - d * s4_c + converted_s4
    de4_c = (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - sigma * e4_c - d * e4_c + converted_e4
    di4_c = sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4
    dr4_c = (1 - fatality_rate_c) * gamma * i4_c - d * r4_c + converted_r4
    dd4_c = fatality_rate_c * gamma * i4_c
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - d * s4_p - converted_s4
    de4_p = (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - sigma * e4_p - d * e4_p - converted_e4
    di4_p = sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4
    dr4_p = (1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4
    dd4_p = fatality_rate_p * gamma * i4_p
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p,
        ddelta_1_c, ddelta_1_p, ddelta_2_c, ddelta_2_p, ddelta_3_c, ddelta_3_p, ddelta_4_c, ddelta_4_p
    ]


def simple_demographic_model_subpopulation_pairs_with_conversion_to_dense_zones(y, t, parameters):
    """
    Extension of def simple_demographic_model_with_conversion by four zones.
    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p
    ) = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations in each zone
    n1_c = max(s1_c + e1_c + i1_c + r1_c, 0)
    n1_p = max(s1_p + e1_p + i1_p + r1_p, 0)

    n2_c = max(s2_c + e2_c + i2_c + r2_c, 0)
    n2_p = max(s2_p + e2_p + i2_p + r2_p, 0)

    n3_c = max(s3_c + e3_c + i3_c + r3_c, 0)
    n3_p = max(s3_p + e3_p + i3_p + r3_p, 0)

    n4_c = max(s4_c + e4_c + i4_c + r4_c, 0)
    n4_p = max(s4_p + e4_p + i4_p + r4_p, 0)

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_c, s1_p, 0)
    converted_e1 = min(conversion_rate * e1_c, e1_p, 0)
    converted_i1 = min(conversion_rate * i1_c, i1_p, 0)
    converted_r1 = min(conversion_rate * r1_c, r1_p, 0)

    converted_s2 = min(conversion_rate * s2_c, s2_p, 0)
    converted_e2 = min(conversion_rate * e2_c, e2_p, 0)
    converted_i2 = min(conversion_rate * i2_c, i2_p, 0)
    converted_r2 = min(conversion_rate * r2_c, r2_p, 0)

    converted_s3 = min(conversion_rate * s3_c, s3_p, 0)
    converted_e3 = min(conversion_rate * e3_c, e3_p, 0)
    converted_i3 = min(conversion_rate * i3_c, i3_p, 0)
    converted_r3 = min(conversion_rate * r3_c, r3_p, 0)

    converted_s4 = min(conversion_rate * s4_c, s4_p, 0)
    converted_e4 = min(conversion_rate * e4_c, e4_p, 0)
    converted_i4 = min(conversion_rate * i4_c, i4_p, 0)
    converted_r4 = min(conversion_rate * r4_c, r4_p, 0)

    def non_negative(value):
        # return max(0, value)
        return value

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = non_negative(b * n1_c - d * s1_c + converted_s1 + converted_s2)
    de1_c = non_negative(-d * e1_c + converted_e1)
    di1_c = non_negative(-d * i1_c + converted_i1)
    dr1_c = non_negative(-d * r1_c + converted_r1 + converted_s2)
    dd1_c = 0
    da1_c = non_negative(d * n1_c)

    # Pagan compartments in Zone 1
    ds1_p = non_negative(b * n1_p - d * s1_p - converted_s1)
    de1_p = non_negative(-d * e1_p - converted_e1)
    di1_p = non_negative(-d * i1_p - converted_i1)
    dr1_p = non_negative(-d * r1_p - converted_r1)
    dd1_p = 0
    da1_p = non_negative(d * n1_p)

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = non_negative(b * n2_c - d * s2_c + converted_s3)
    de2_c = non_negative(-d * e2_c + converted_e2)
    di2_c = non_negative(-d * i2_c + converted_i2)
    dr2_c = non_negative(-d * r2_c + converted_r3)
    dd2_c = 0
    da2_c = non_negative(d * n2_c)

    # Pagan compartments in Zone 2
    ds2_p = non_negative(b * n2_p - d * s2_p - converted_s2)
    de2_p = non_negative(-d * e2_p - converted_e2)
    di2_p = non_negative(-d * i2_p - converted_i2)
    dr2_p = non_negative(-d * r2_p - converted_r2)
    dd2_p = 0
    da2_p = non_negative(d * n2_p)

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = non_negative(b * n3_c - d * s3_c + converted_s4)
    de3_c = non_negative(-d * e3_c + converted_e3)
    di3_c = non_negative(-d * i3_c + converted_i3)
    dr3_c = non_negative(-d * r3_c + converted_r4)
    dd3_c = 0
    da3_c = non_negative(d * n3_c)

    # Pagan compartments in Zone 3
    ds3_p = non_negative(b * n3_p - d * s3_p - converted_s3)
    de3_p = non_negative(-d * e3_p - converted_e3)
    di3_p = non_negative(-d * i3_p - converted_i3)
    dr3_p = non_negative(-d * r3_p - converted_r3)
    dd3_p = 0
    da3_p = non_negative(d * n3_p)

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = non_negative(b * n4_c - d * s4_c + converted_s4)
    de4_c = non_negative(-d * e4_c + converted_e4)
    di4_c = non_negative(-d * i4_c + converted_i4)
    dr4_c = non_negative(-d * r4_c + converted_r4)
    dd4_c = 0
    da4_c = non_negative(d * n4_c)

    # Pagan compartments in Zone 4
    ds4_p = non_negative(b * n4_p - d * s4_p - converted_s4)
    de4_p = non_negative(-d * e4_p - converted_e4)
    di4_p = non_negative(-d * i4_p - converted_i4)
    dr4_p = non_negative(-d * r4_p - converted_r4)
    dd4_p = 0
    da4_p = non_negative(d * n4_p)

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]


def direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(y, t, parameters):
    """
    An extenstion of the following model:
    direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model

    In this model the Christians and Pagans still form subpopulations in 4 zones, but converstion
    if Pagans to Christians now happens from the lower density zones to the higher density zones.
    This means that individuals convert from zone 4 to 3, 3 to 2, 2 to 1, and 1 to 1, and in doing so
    represent the idea that Christianity was an urban phenomenon, happening towards areas with more
    towns and cities.
    """
    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p,
        delta_1_c, delta_1_p, delta_2_c, delta_2_p, delta_3_c, delta_3_p, delta_4_c, delta_4_p
    ) = y

    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    unified_deltas = parameters.unified_deltas
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily
    max_delta = parameters.max_delta

    epsilon = 1e-4  # Small value to prevent division by zero

    if unified_deltas:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c + s1_p, i_n=i1_c + i1_p, r_n=r1_c + r1_p
        )
        ddelta_1_p = ddelta_1_c

        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c + s2_p, i_n=i2_c + i2_p, r_n=r2_c + r2_p
        )
        ddelta_2_p = ddelta_2_c

        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c + s3_p, i_n=i3_c + i3_p, r_n=r3_c + r3_p
        )
        ddelta_3_p = ddelta_3_c

        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c + s4_p, i_n=i4_c + i4_p, r_n=r4_c + r4_p
        )
        ddelta_4_p = ddelta_4_c

    else:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c, i_n=i1_c, r_n=r1_c
        )
        ddelta_1_p = adjust_delta(
            delta_prev=delta_1_p, delta_0=zeleners_initial_delta_1, s_n=s1_p, i_n=i1_p, r_n=r1_p
        )
        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c, i_n=i2_c, r_n=r2_c
        )
        ddelta_2_p = adjust_delta(
            delta_prev=delta_2_p, delta_0=zeleners_initial_delta_2, s_n=s2_p, i_n=i2_p, r_n=r2_p
        )
        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c, i_n=i3_c, r_n=r3_c
        )
        ddelta_3_p = adjust_delta(
            delta_prev=delta_3_p, delta_0=zeleners_initial_delta_3, s_n=s3_p, i_n=i3_p, r_n=r3_p
        )
        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c, i_n=i4_c, r_n=r4_c
        )
        ddelta_4_p = adjust_delta(
            delta_prev=delta_4_p, delta_0=zeleners_initial_delta_4, s_n=s4_p, i_n=i4_p, r_n=r4_p
        )

    # Totals of subpopulations in each zone and whole population
    n1_c = max(s1_c + e1_c + i1_c + r1_c, 0)
    n1_p = max(s1_p + e1_p + i1_p + r1_p, 0)
    n1 = max(n1_c + n1_p, 0)

    n2_c = max(s2_c + e2_c + i2_c + r2_c, 0)
    n2_p = max(s2_p + e2_p + i2_p + r2_p, 0)
    n2 = max(n2_c + n2_p, 0)

    n3_c = max(s3_c + e3_c + i3_c + r3_c, 0)
    n3_p = max(s3_p + e3_p + i3_p + r3_p, 0)
    n3 = max(n3_c + n3_p, 0)

    n4_c = max(s4_c + e4_c + i4_c + r4_c, 0)
    n4_p = max(s4_p + e4_p + i4_p + r4_p, 0)
    n4 = max(n4_c + n4_p, 0)

    # Pagans converted to Christianity for all living compartments in each zone
    # converted_s1 = min(conversion_rate * s1_p, s1_p)
    # converted_e1 = min(conversion_rate * e1_p, e1_p)
    # converted_i1 = min(conversion_rate * i1_p, i1_p)
    # converted_r1 = min(conversion_rate * r1_p, r1_p)
    #
    # converted_s2 = min(conversion_rate * s2_p, s2_p)
    # converted_e2 = min(conversion_rate * e2_p, e2_p)
    # converted_i2 = min(conversion_rate * i2_p, i2_p)
    # converted_r2 = min(conversion_rate * r2_p, r2_p)
    #
    # converted_s3 = min(conversion_rate * s3_p, s3_p)
    # converted_e3 = min(conversion_rate * e3_p, e3_p)
    # converted_i3 = min(conversion_rate * i3_p, i3_p)
    # converted_r3 = min(conversion_rate * r3_p, r3_p)
    #
    # converted_s4 = min(conversion_rate * s4_p, s4_p)
    # converted_e4 = min(conversion_rate * e4_p, e4_p)
    # converted_i4 = min(conversion_rate * i4_p, i4_p)
    # converted_r4 = min(conversion_rate * r4_p, r4_p)

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_c, s1_p, 0)
    converted_e1 = min(conversion_rate * e1_c, e1_p, 0)
    converted_i1 = min(conversion_rate * i1_c, i1_p, 0)
    converted_r1 = min(conversion_rate * r1_c, r1_p, 0)

    converted_s2 = min(conversion_rate * s2_c, s2_p, 0)
    converted_e2 = min(conversion_rate * e2_c, e2_p, 0)
    converted_i2 = min(conversion_rate * i2_c, i2_p, 0)
    converted_r2 = min(conversion_rate * r2_c, r2_p, 0)

    converted_s3 = min(conversion_rate * s3_c, s3_p, 0)
    converted_e3 = min(conversion_rate * e3_c, e3_p, 0)
    converted_i3 = min(conversion_rate * i3_c, i3_p, 0)
    converted_r3 = min(conversion_rate * r3_c, r3_p, 0)

    converted_s4 = min(conversion_rate * s4_c, s4_p, 0)
    converted_e4 = min(conversion_rate * e4_c, e4_p, 0)
    converted_i4 = min(conversion_rate * i4_c, i4_p, 0)
    converted_r4 = min(conversion_rate * r4_c, r4_p, 0)

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = b * n1_c - (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - d * s1_c + converted_s1 + converted_s2
    de1_c = (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - sigma * e1_c - d * e1_c + converted_e1
    di1_c = sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1
    dr1_c = (1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1 + converted_r2
    dd1_c = fatality_rate_c * gamma * i1_c
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - d * s1_p - converted_s1
    de1_p = (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - sigma * e1_p - d * e1_p - converted_e1
    di1_p = sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1
    dr1_p = (1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1
    dd1_p = fatality_rate_p * gamma * i1_p
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - d * s2_c + converted_s3
    de2_c = (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - sigma * e2_c - d * e2_c + converted_e2
    di2_c = sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2
    dr2_c = (1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r3
    dd2_c = fatality_rate_c * gamma * i2_c
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - d * s2_p - converted_s2
    de2_p = (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - sigma * e2_p - d * e2_p - converted_e2
    di2_p = sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2
    dr2_p = (1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2
    dd2_p = fatality_rate_p * gamma * i2_p
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - d * s3_c + converted_s4
    de3_c = (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - sigma * e3_c - d * e3_c + converted_e3
    di3_c = sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3
    dr3_c = (1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r4
    dd3_c = fatality_rate_c * gamma * i3_c
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - d * s3_p - converted_s3
    de3_p = (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - sigma * e3_p - d * e3_p - converted_e3
    di3_p = sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3
    dr3_p = (1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3
    dd3_p = fatality_rate_p * gamma * i3_p
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - d * s4_c
    de4_c = (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - sigma * e4_c - d * e4_c + converted_e4
    di4_c = sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4
    dr4_c = (1 - fatality_rate_c) * gamma * i4_c - d * r4_c
    dd4_c = fatality_rate_c * gamma * i4_c
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - d * s4_p - converted_s4
    de4_p = (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - sigma * e4_p - d * e4_p - converted_e4
    di4_p = sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4
    dr4_p = (1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4
    dd4_p = fatality_rate_p * gamma * i4_p
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p,
        ddelta_1_c, ddelta_1_p, ddelta_2_c, ddelta_2_p, ddelta_3_c, ddelta_3_p, ddelta_4_c, ddelta_4_p
    ]
