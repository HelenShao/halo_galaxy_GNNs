# A Universal Equation to Predict $\Omega_{m}$ from Halo and Galaxy Catalogues
Using graph neural networks (GNNs) to map dark matter halo 3D positions and velocities to omega matter. 

After training the model, we use symbolic regression (PYSR) to approximate the learned model and obtain analytic expressions that can be evaluated on halo and galaxy fields from various N-body and hydrodynamic simulations. 

1. "GNN_train" contains the code for training the GNN model on N-body halos
2. "GNN_test" contains the code for testing the GNN model on N-body (or hydrodynamic) simulations
3. "symbolic_regression" contains the code for training and testing the symbolic regressor, using PYSR

Paper: [https://arxiv.org/abs/2302.14591](https://arxiv.org/abs/2302.14591)
