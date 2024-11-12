# GAN and VAE Implementation

This assignment implements Generative Adversarial Networks (GAN) and Variational Autoencoders (VAE) for generating and visualizing images.

## Requirements

Make sure to install the required packages before running the project. The code is tested on a custom environment on the Kelvin server.

## Usage Instructions

1. Ensure all the necessary files are in the working directory:
   - `GAN_main.ipynb`: Notebook for implementing and visualizing GANs.
   - `autoencoder_main.ipynb`: Notebook for implementing and visualizing VAEs.
   - `utils.py`: Contains utility functions.
   - `models.py`: Contains all Models.
   - `requirements.txt`: Contains all dependencies for the project.

2. **Running GAN and VAE Notebooks**:
   - Open and run `GAN_main.ipynb` to train the GAN model.
   - Open and run `autoencoder_main.ipynb` to train and visualize the VAE model, including anomaly detection and latent space visualization.

3. **Visualizing Generation with TensorBoard**:
   - To visualize image generations after each epoch, launch TensorBoard from `utils.py`.