# latent_space_norm
The repository contains a pytorch implementation of a VAE used for speaker-normalizing vowel sounds, together with code for calculating LPC spectra.

Suggested normalizing method: 
1. Generate a training set of order 16 LPC vowel spectra (e.g. from TIMIT) using 100 frequencies in 0Hz->4000Hz equally space on mel scale
2. Train VAE with latend dimension 6 using hidden layer sizes [100,100,50] (encoder) [50,10,10] (decoder) and $\sigma^2=0.1$
3. Recover latent features for each spectrum and separately standardise each speaker's set of features
4. Decode the spectra using the standardised features
5. Formant peaks using e.g. scipy.signal.find_peaks
