# latent_space_norm
The repository contains a pytorch implementation of a VAE used for speaker-normalizing vowel sounds, together with code for calculating LPC spectra, and searching latent space for formant targets.

Suggested normalizing method: 
1. Generate a training set of order 16 LPC vowel spectra (e.g. from TIMIT) using 100 frequencies in 0Hz->4000Hz equally space on mel scale.
2. Train VAE with latent dimension 6 using hidden layer sizes [100,100,50] (encoder) [50,100,100] (decoder) and $\sigma^2=0.1$.
3. Recover latent features for each spectrum and separately standardise each speaker's set of features, as in $\tilde{z_i}=(z_i-\mu_i)/\sigma_i$.
4. Decode the spectra using the standardised features.
5. Calculate formant peaks using e.g. `scipy.signal.find_peaks`.


To search latent space for formant targets:
1. Determine spectral peak locations for each LPC spectrum using e.g. `scipy.signal.find_peaks`.
2. Train sklearn models (e.g. `KNeighboursRegressor` with Gaussian kernel having width 0.5) to predict first two spectral peaks (F1 and F2) from latent features.
3. Place models in dictionary, and pass to `normalize` function along with initial latent point, and F1, F2 targets. 
