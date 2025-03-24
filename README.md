# eco-simulator

A package to simulate ecological systems by numerical integration of ordinary differential equations. We use the generalized MacArthur consumer-resource model to simulate the dynamics of a community of species. We want to analyze the stability of ecological systems and therefore some randomness is forced on the system parameters.

## Experiments

Parameters used for simulations are stored in the 'experiments' directory as individual subdirectories. Each experiment uses its own specific parameters, but can be used to simulate different initial conditions. The parameters are stored in the data subdirectory inside each experiment directory. Community class has load_data and create_data methods to easily use this data in reproducible manner.
