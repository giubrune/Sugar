This repository contain the HYDRUS models used to perform the analysis for both irrigation scenarios.
INSTRUCTIONS

    The project folder and the .h1d file must share the same path.
    Files .h1d can be open in HYDRUS-1D to explore model definition and boundary conditions, and eventually modify soil-related parameters.
    The project folder contains the file SUGAR.IN, which lists all parameters related to root water uptake (Couvreur model) and fruit biophysical processes (SUGAR model).
    The file SUGAR.IN contains also the time series of air temperature and relative humidity, and the Root Area Index (RAI).
    The file FIT.IN contains the measured volumetric water contents at z=-15 cm (Code=2), the measured fruit water mass (Code=40), the measured dry mass (Code=41), the measured mass of soluble sugars (Code=42), and the measured starch mass (Code=43).
    The project folder contains also the executable H1D_Sugar.exe, which executes the coupled HYDRUS-SUGAR model described in the paper.
    To execute it, the user must set the project path in the file LEVEL_01.DIR, open the command prompt in the project folder, and type .\H1D_Sugar.exe.
    After executing the model (besides classical hydrus output files), two further output files are produced:
        SUGAR.OUT: Contains the time series of the stem water potential, fruit water mass, fruit dry mass, soluble sugars mass, starch mass, and the mass of ther structural compounds.
        SUGAR_flux.out: Contains the time series of the xylem water flow, phloem water flow, fruit water transpiration, active uptake of sugars, convective sugar flow, passive diffusion fluxes of sugars, total sugar uptake fluxes, and sugar respiration fluxes
