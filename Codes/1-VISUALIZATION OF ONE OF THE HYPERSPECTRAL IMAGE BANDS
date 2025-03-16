# Libraries 
import spectral.io.envi as envi 
import matplotlib.pyplot as plt 
import numpy as np 

# Input directory and base name of the file 
input_folder = "C:/Users/Hp/Documents/Internals" 
base_name = "internals_mushroom" 

# HDR and RAW files 
hdr_file = f"{input_folder}/{base_name}.hdr" 
raw_file = f"{input_folder}/{base_name}.raw" 

# Read metadata 
metadata = envi.read_envi_header(hdr_file) 

# Print metadata 
print("HDR file metadata:") 
print(metadata) 

# Image dimensions 
rows = metadata['lines'] 
cols = metadata['samples'] 
bands = metadata['bands'] 
dtype = metadata['data type'] 

print("\nImage dimensions:") 
print("Rows:", rows) 
print("Columns:", cols) 
print("Number of bands:", bands) 
print("Data type:", dtype) 

# Load hyperspectral image 
img =  envi.open(hdr_file, raw_file) 

#Band number 50 of 400 bands is displayed 
plt.figure() 
plt.imshow(np.array(img[:,:,50])) 

