# halo_galaxy_GNNs
Using graph neural networks (GNNs) to map dark matter halo 3D positions and velocities to omega matter.
After training the model, we use symbolic regression (PYSR) to approximate the learned model and obtain analytic expressions that can be evaluated on halo and galaxy fields from various N-body and hydrodynamic simulations
"GNN_train" contains the code for training the GNN model on N-body halos
"GNN_test" contains the code for testing the GNN model on N-body (or hydrodynamic) simulations
"symbolic_regression" contains the code for training and testing the symbolic regressor, using PYSR
Paper: [Link text Here](https://arxiv.org/abs/2302.14591)