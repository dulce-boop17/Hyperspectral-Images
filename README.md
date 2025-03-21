# Hyperspectral Image Processing using Machine Learning Algorithms
Repository for hyperspectral image analysis, the objective is to develop a classification model based on hyperspectral images to accurately differentiate regions of interest in fruits and vegetables, such as pulp-seed and crown-stem, using image processing and machine learning techniques with Giessen data

## How to use:
1. Download the Giessen dataset: https://zenodo.org/records/1186372#.XOoch8hKjIU
2. Install the dependencies: Make sure that all necessary libraries are installed.
3. Set up directories and files: Set up the HDR and RAW file paths in the code according to your directory structure.
4. Pixels: Make sure to choose the pixel coordinates according to the corresponding fruit or vegetable.
5. Run the code: Run the script to perform the sorting and display the results.

## Project Features:
• **Hyperspectral image reading and processing:** Use of HDR and RAW files to load and process hyperspectral images using the spectral library.  
• **Extraction of spectra:** Selection of specific pixels corresponding to seeds and pulp to extract their characteristic spectra with respect to wavelength.  
• **Calculation of statistical moments:** Calculation of the mean, standard deviation, skewness and kurtosis of the spectra to use them as features in classification.  
• **Support Vector Machine (SVM) classification:** Implementation of an SVM classifier to differentiate between seed and pulp pixels based on statistical moments.  
• **K-Nearest Neighbors (KNN) classification:** Implementation of a KNN classifier to differentiate between seed and pulp pixels based on statistical moments.  
• **Decision boundary visualization:** Plots showing how the KNN classifier separates data based on different pairs of statistical features.  

### Requirements
- Python 3.x
- Libraries: spectral, numpy, matplotlib, scipy, sklearn
- Previously downloaded dataset
