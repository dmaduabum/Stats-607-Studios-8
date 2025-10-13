all code is stored in the src folder. the simulation file stores csv's of the simulated data in data/. the regression files read this data and store true and estimated beta in the artifacts folder. these are read by the files that calculate the MSE and produce visuals stored in the figures/ folder.

simulation results should follow this format: each row contains `method`, `n`, `p`, `df`, `rho`, `SNR`, `rep`, `mse` columns for evaluation and plotting.
