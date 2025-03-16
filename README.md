# Hyperspectral-Images
Repository for hyperspectral image analysis, focused on the classification of pulp-seed in fruits and crown-stem in vegetables, using image processing and machine learning with Giessen data. The objective is to develop a classification model based on hyperspectral images to accurately differentiate regions of interest in fruits and vegetables, such as pulp-seed and crown-stem, using image processing and machine learning techniques.

## How to use:
1. Download the Giessen dataset: https://zenodo.org/records/1186372#.XOoch8hKjIU
2. Install dependencies: Make sure all necessary libraries are installed.
3. Configure directories and files: Set the paths to the HDR and RAW files in the code according to your directory structure.
4. Run the code: Run the script to perform the classification and display the results.

## Project Features:
• Hyperspectral image reading and processing: Use of HDR and RAW files to load and process hyperspectral images using the spectral library.
• Extraction of spectra: Selection of specific pixels corresponding to seeds and pulp to extract their characteristic spectra with respect to wavelength.
• Calculation of Statistical Moments: Calculation of the mean, standard deviation, skewness and kurtosis of the spectra to use them as features in classification.
• Support Vector Machine (SVM) Classification: Implementation of an SVM classifier to differentiate between seed and pulp pixels based on statistical moments.
• K-Nearest Neighbors (KNN) classification: Implementation of a KNN classifier to differentiate between seed and pulp pixels based on statistical moments.
• Decision Boundary Visualization: Plots showing how the KNN classifier separates data based on different pairs of statistical features.

### Requirements
- Python 3.x
- Libraries: spectral, numpy, matplotlib, scipy, sklearn
