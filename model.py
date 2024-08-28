import numpy as np
import matplotlib.pyplot as plt
from cbsyst import Csys # this is for carbon chemistry calculations

################################################################################################
# initial variables and constants and boxes - the model is set to equilibrium here.

# global variables
V_ocean = 1.34e18  # volume of the ocean in m3
SA_ocean = 358e12  # surface area of the ocean in m2
fSA_hilat = 0.15  # fraction of ocean surface area in 'high latitude' box

# variables used to calculate Q (essentially various flux coefficients)
Q_alpha = 1e-4
Q_beta = 7e-4
Q_k = 8.3e17

# salinity balance - the total amount of salt added or removed to the surface boxes
Fw = 0.1  # low latitude evaporation - precipitation in units of m yr-1
Sref = 35  # reference salinity in units of g kg-1
E = Fw * SA_ocean * (1 - fSA_hilat) * Sref  # amount of salt removed from the low latitude box,  g kg-1 yr-1, ~ kg m-3 yr-1

init_hilat = {
    'name': 'hilat',
    'depth': 200,  # box depth, m
    'SA': SA_ocean * fSA_hilat,  # box surface area, m2
    'T': 3.897678,  # initial water temperature, Celcius
    'S': 34.37786,  # initial salinity
    'T_atmos': 0.,  # air temperature, Celcius
    'tau_M': 100.,  # timescale of surface-deep mixing, yr
    'tau_T': 2.,  # timescale of temperature exchange with atmosphere, yr
    'E': -E,  # salt added due to evaporation - precipitation, kg m-3 yr-1
    'tau_CO2': 2.,  # timescale of CO2 exchange, yr
    'DIC': 2.02837,  # Dissolved Inorganic Carbon concentration, mol m-3
    'TA': 2.22116,  # Total Alkalinity, mol m-3
    'tau_PO4': 3.,  # phosphate half life, yr at initial f_CaCO3
    'PO4': 8.8995e-5,  # Phosphate conc, mol m-3
    'f_CaCO3': 0.2,  # fraction of organic matter export that produces CaCO3 at starting [CO3]
}
init_hilat['V'] = init_hilat['SA'] *  init_hilat['depth']  # box volume, m3

init_lolat = {
    'name': 'lolat',
    'depth': 100,  # box depth, m
    'SA': SA_ocean * (1 - fSA_hilat),  # box surface area, m2
    'T': 23.60040,  # initial water temperature, Celcius
    'S': 35.37898,  # initial salinity
    'T_atmos': 25.,  # air temperature, Celcius
    'tau_M': 250.,  # timescale of surface-deep mixing, yr
    'tau_T': 2.,  # timescale of temperature exchange with atmosphere, yr
    'E': E,  # salinity balance, PSU m3 yr-1
    'tau_CO2': 2.,  # timescale of CO2 exchange, yr
    'DIC': 1.99405,  # Dissolved Inorganic Carbon concentration, mol m-3
    'TA': 2.21883,  # Total Alkalinity, mol m-3
    'tau_PO4': 2.,  # phosphate half life, yr at initial f_CaCO3
    'PO4': 1.6541e-4,  # Phosphate conc, mol m-3
    'f_CaCO3': 0.3,  # fraction of organic matter export that produces CaCO3 at starting [CO3]
}
init_lolat['V'] = init_lolat['SA'] *  init_lolat['depth']  # box volume, m3

init_deep = {
    'name': 'deep',
    'V': V_ocean - init_lolat['V'] - init_hilat['V'],  # box volume, m3
    'T': 5.483637,  # initial water temperature, Celcius
    'S': 34.47283,  # initial salinity
    'DIC': 2.32712,  # Dissolved Inorganic Carbon concentration, mol m-3
    'TA': 2.31640,  # Total Alkalinity, mol m-3
    'PO4': 2.30515e-3,  # Phosphate conc, mol m-3
}

init_atmos = {
    'name': 'atmos',
    'mass': 5.132e18,  # kg
    'moles_air': 1.736e20,  # moles
    'moles_CO2': 872e15 / 12,  # moles
    'GtC_emissions': 0.0  # annual emissions of CO2 into the atmosphere, GtC
}
init_atmos['pCO2'] = init_atmos['moles_CO2'] / init_atmos['moles_air'] * 1e6

################################################################################################
# the actual model, which uses Euler's method to solve for carbon in each reservoir

def ocean_model(lolat, hilat, deep, atmos, tmax, dt):

    # create the time scale for the model
    time = np.arange(0, tmax + dt, dt)

    # identify which variables will change with time
    model_vars = ['T', 'S', 'DIC', 'TA', 'PO4']
    atmos_model_vars = ['moles_CO2', 'pCO2']

    # create copies of the input dictionaries so we don't modify the originals
    lolat = lolat.copy()
    hilat = hilat.copy()
    deep = deep.copy()
    atmos = atmos.copy()

    # turn all time-evolving variables into arrays containing the start values
    for box in [lolat, hilat, deep]:
        for k in model_vars:
            box[k] = np.full(time.shape, box[k])
    for k in atmos_model_vars:
        atmos[k] = np.full(time.shape, atmos[k])
    if isinstance(atmos['GtC_emissions'], (int, float)):
        atmos['GtC_emissions'] = np.full(time.shape, atmos['GtC_emissions'])

    # calculate initial surface carbon chemistry in the surface boxes using Csys, and store a few key variables - CO2, pH, pCO2 and K0
    for box in [lolat, hilat]:
        csys = Csys(
            TA=box['TA'],
            DIC=box['DIC'],
            T_in=box['T'], S_in=box['S'],
            unit='mmol'
            )
        box['CO2'] = csys.CO2
        box['pH'] = csys.pHtot
        box['pCO2'] = csys.pCO2
        box['K0'] = csys.Ks.K0

    # Create a dictionary to keep track of the fluxes calculated at each step
    fluxes = {}

    for i in range(1, time.size):
        last = i - 1  # index of last model step

        # calculate circulation flux, Q
        dT = lolat['T'][last] - hilat['T'][last]
        dS = lolat['S'][last] - hilat['S'][last]
        Q = Q_k * (Q_alpha * dT - Q_beta * dS)

        # calculate mixing fluxes for model variables
        for var in model_vars:
            fluxes[f'Q_{var}_deep'] = Q * (hilat[var][last] - deep[var][last]) * dt  # mol dt-1
            fluxes[f'Q_{var}_hilat'] = Q * (lolat[var][last] - hilat[var][last]) * dt  # mol dt-1
            fluxes[f'Q_{var}_lolat'] = Q * (deep[var][last] - lolat[var][last]) * dt  # mol dt-1

            fluxes[f'vmix_{var}_hilat'] = hilat['V'] / hilat['tau_M'] * (hilat[var][last] - deep[var][last]) * dt  # mol dt-1
            fluxes[f'vmix_{var}_lolat'] = lolat['V'] / lolat['tau_M'] * (lolat[var][last] - deep[var][last]) * dt  # mol dt-1

        # calculate surface-specific fluxes
        for box in [hilat, lolat]:
            boxname = box['name']
            # temperature exchange with atmosphere
            fluxes[f'dT_{boxname}'] = box['V'] / box['tau_T'] * (box['T_atmos'] - box['T'][last]) * dt  # mol dt-1
            # CO2 exchange with atmosphere
            fluxes[f'dCO2_{boxname}'] = box['V'] / box['tau_CO2'] * (box['CO2'][last] - 1e-3 * atmos['pCO2'][last] * box['K0'][last]) * dt  # mol dt-1
            # organic matter production
            fluxes[f'export_PO4_{boxname}'] = box['PO4'][last] * box['V'] / box['tau_PO4'] * dt  # mol PO4 dt-1
            # DIC export by productivity :                                  redfield + calcification
            fluxes[f'export_DIC_{boxname}'] = fluxes[f'export_PO4_{boxname}'] * (106 + 106 * box['f_CaCO3'])  # mol DIC dt-1
            # TA export by productivity :                                  redfield + calcification
            fluxes[f'export_TA_{boxname}'] = fluxes[f'export_PO4_{boxname}'] * (-18 + 2 * 106 * box['f_CaCO3'])  # mol TA dt-1

        fluxes['dCO2_emissions'] = atmos['GtC_emissions'][last] * 1e15 / 12 * dt  # mol dt-1

        # update deep box
        for var in model_vars:
            if var in ['T', 'S']:
                deep[var][i] = deep[var][last] + (
                    fluxes[f'Q_{var}_deep'] + fluxes[f'vmix_{var}_hilat'] + fluxes[f'vmix_{var}_lolat']
                ) / deep['V']
            else:
                deep[var][i] = deep[var][last] + (
                    fluxes[f'Q_{var}_deep'] + fluxes[f'vmix_{var}_hilat'] + fluxes[f'vmix_{var}_lolat'] + fluxes[f'export_{var}_hilat'] + fluxes[f'export_{var}_lolat']
                ) / deep['V']

        # update surface boxes
        for box in [hilat, lolat]:
            boxname = box['name']
            box['S'][i] = box['S'][last] + (fluxes[f'Q_S_{boxname}'] - fluxes[f'vmix_S_{boxname}'] + box['E'] * dt) / box['V']
            box['T'][i] = box['T'][last] + (fluxes[f'Q_T_{boxname}'] - fluxes[f'vmix_T_{boxname}'] + fluxes[f'dT_{boxname}']) / box['V']

            box['DIC'][i] = box['DIC'][last] + (fluxes[f'Q_DIC_{boxname}'] - fluxes[f'vmix_DIC_{boxname}'] - fluxes[f'dCO2_{boxname}'] - fluxes[f'export_DIC_{boxname}']) / box['V']
            box['TA'][i] = box['TA'][last] + (fluxes[f'Q_TA_{boxname}'] - fluxes[f'vmix_TA_{boxname}'] - fluxes[f'export_TA_{boxname}']) / box['V']
            box['PO4'][i] = box['PO4'][last] + (fluxes[f'Q_PO4_{boxname}'] - fluxes[f'vmix_PO4_{boxname}'] - fluxes[f'export_PO4_{boxname}']) / box['V']

            # update carbon speciation
            csys = Csys(
                TA=box['TA'][i],
                DIC=box['DIC'][i],
                T_in=box['T'][i], S_in=box['S'][i],
                unit='mmol'
                )
            box['CO2'][i] = csys.CO2[0]
            box['pH'][i] = csys.pHtot[0]
            box['pCO2'][i] = csys.pCO2[0]
            box['K0'][i] = csys.Ks.K0

        # update atmosphere
        atmos['moles_CO2'][i] = atmos['moles_CO2'][last] + fluxes['dCO2_hilat'] + fluxes['dCO2_lolat'] + fluxes['dCO2_emissions']
        atmos['pCO2'][i] = 1e6 * atmos['moles_CO2'][i] / atmos['moles_air']

    return time, lolat, hilat, deep, atmos

################################################################################################
# setting up the model for a particular problem: here we consider the effect of anthropogenic emissions
# on equilibrium carbon concentrations in three different situations.

# a) ocean acidification:

init_lolat_1 = init_lolat.copy()
init_hilat_1 = init_hilat.copy()

init_lolat_1['f_CaCO3']=(init_lolat['f_CaCO3']/2)
init_hilat_1['f_CaCO3']=(init_hilat['f_CaCO3']/2)

# b) ocean acidification with ballasting feedback:

init_lolat_2 = init_lolat.copy()
init_hilat_2 = init_hilat.copy()

init_lolat_2['f_CaCO3']=(init_lolat['f_CaCO3']/2)
init_hilat_2['f_CaCO3']=(init_hilat['f_CaCO3']/2)

init_lolat_2['tau_PO4']=(init_lolat['tau_PO4']*2)
init_hilat_2['tau_PO4']=(init_hilat['tau_PO4']*2)

# finding equilibrium for these three models, so we can analyse peturbation from equilibrium later

time, lolat, hilat, deep, atmos = ocean_model(init_lolat, init_hilat, init_deep, init_atmos, 1000, 0.5)
time_1, lolat_1, hilat_1, deep_1, atmos_1 = ocean_model(init_lolat_1, init_hilat_1, init_deep, init_atmos, 1000, 0.5)
time_2, lolat_2, hilat_2, deep_2, atmos_2 = ocean_model(init_lolat_2, init_hilat_2, init_deep, init_atmos, 1000, 0.5)

equilibrium = {
    'lolat': lolat,
    'hilat': hilat,
    'deep': deep,
    'atmos': atmos
}

equilibrium_1 = {
    'lolat': lolat_1,
    'hilat': hilat_1,
    'deep': deep_1,
    'atmos': atmos_1
}

equilibrium_2 = {
    'lolat': lolat_2,
    'hilat': hilat_2,
    'deep': deep_2,
    'atmos': atmos_2
}

# make copies of the original input dictionaries
initial_lolat = init_lolat.copy()
initial_hilat = init_hilat.copy()
initial_deep = init_deep.copy()
initial_atmos = init_atmos.copy()

initial_lolat_1 = init_lolat_1.copy()
initial_hilat_1 = init_hilat_1.copy()
initial_deep_1 = init_deep.copy()
initial_atmos_1 = init_atmos.copy()

initial_lolat_2 = init_lolat_2.copy()
initial_hilat_2 = init_hilat_2.copy()
initial_deep_2 = init_deep.copy()
initial_atmos_2 = init_atmos.copy()

for variable in ['T', 'S', 'DIC', 'pCO2', 'TA', 'moles_CO2', 'PO4']:
    for dictionary_name, dictionary in [('lolat', initial_lolat), ('hilat', initial_hilat), ('deep', initial_deep), ('atmos', initial_atmos)]:
        if variable in dictionary:
            dictionary[variable] = equilibrium[dictionary_name][variable][-1]
        else:
            continue

for variable in ['T', 'S', 'DIC', 'pCO2', 'TA', 'moles_CO2', 'PO4']:
    for dictionary_name, dictionary in [('lolat', initial_lolat_1), ('hilat', initial_hilat_1), ('deep', initial_deep_1), ('atmos', initial_atmos_1)]:
        if variable in dictionary:
            dictionary[variable] = equilibrium_1[dictionary_name][variable][-1]
        else:
            continue

for variable in ['T', 'S', 'DIC', 'pCO2', 'TA', 'moles_CO2', 'PO4']:
    for dictionary_name, dictionary in [('lolat', initial_lolat_2), ('hilat', initial_hilat_2), ('deep', initial_deep_2), ('atmos', initial_atmos_2)]:
        if variable in dictionary:
            dictionary[variable] = equilibrium_2[dictionary_name][variable][-1]
        else:
            continue

# create a new time axis for the model containing 2000 years with a 0.5 year time step
tmax = 2000  # how many years to simulate (yr)
dt = 0.5  # the time step of the simulation (yr)
time = np.arange(0, tmax + dt, dt)  # the time axis for the model

# create an array containing GtC_emissions that contains zeros except between 800-1000 years, where 8 GtC are emitted each year.
initial_atmos['GtC_emissions'] = np.zeros(time.shape)  
initial_atmos['GtC_emissions'][(time > 200) & (time <= 400)] = 8.0  

initial_atmos_1['GtC_emissions'] = np.zeros(time.shape)  
initial_atmos_1['GtC_emissions'][(time > 200) & (time <= 400)] = 8.0  

initial_atmos_2['GtC_emissions'] = np.zeros(time.shape)  
initial_atmos_2['GtC_emissions'][(time > 200) & (time <= 400)] = 8.0 

################################################################################################
# run the models and plot suitable graphs

time_a, lolat_a, hilat_a, deep_a, atmos_a = ocean_model(initial_lolat, initial_hilat, initial_deep, initial_atmos, 2000, 0.5)
time_b, lolat_b, hilat_b, deep_b, atmos_b = ocean_model(initial_lolat_1, initial_hilat_1, initial_deep_1, initial_atmos_1, 2000, 0.5)
time_c, lolat_c, hilat_c, deep_c, atmos_c = ocean_model(initial_lolat_2, initial_hilat_2, initial_deep_2, initial_atmos_2, 2000, 0.5)

fig, axs = plt.subplots(5, 1, sharex=True, figsize=(5, 5))

axs[0].plot(time_a, atmos_a['GtC_emissions'], color = "black", alpha = 0.8)
axs[0].set_ylabel('CO$_2$ (Gt)')
axs[0].set_xlim(0,2000)
axs[0].text(1750, 3, 'carbon dioxide emissions', ha='center')
axs[0].axvspan(200, 400, color='gray', alpha=0.15)

axs[1].plot(time_a, atmos_a['pCO2'])
axs[1].plot(time_b, atmos_b['pCO2'])
axs[1].plot(time_c, atmos_c['pCO2'])
axs[1].set_ylabel('pCO$_2$ (ppm)')
axs[1].text(1750, 800, 'atmosphere', ha='center')
axs[1].axvspan(200, 400, color='gray', alpha=0.15)

axs[2].plot(time_a, lolat_a['pCO2'], label = "control")
axs[2].plot(time_b, lolat_b['pCO2'], label = "ocean acidification")
axs[2].plot(time_c, lolat_c['pCO2'], label = "ocean acidification and ballasting feedback")
axs[2].legend(loc='top', prop={'size': 10})
axs[2].set_ylabel('pCO$_2$ (ppm)')
axs[2].text(1750, 550, 'low latitude surface ocean', ha='center')
axs[2].axvspan(200, 400, color='gray', alpha=0.15)

axs[3].plot(time_a, hilat_a['pCO2'])
axs[3].plot(time_b, hilat_b['pCO2'])
axs[3].plot(time_c, hilat_c['pCO2'])
axs[3].set_ylabel('pCO$_2$ (ppm)')
axs[3].text(1750, 550, 'high latitude surface ocean', ha='center')
axs[3].axvspan(200, 400, color='gray', alpha=0.15)

axs[4].plot(time_a, hilat_a['DIC'])
axs[4].plot(time_b, hilat_b['DIC'])
axs[4].plot(time_c, hilat_c['DIC'])
axs[4].set_xlabel('Time (years)')
axs[4].set_ylabel('DIC (mol/m$^3$)')
axs[4].text(1750, 2.05, 'deep ocean', ha='center')
axs[4].axvspan(200, 400, color='gray', alpha=0.15)

plt.show()














