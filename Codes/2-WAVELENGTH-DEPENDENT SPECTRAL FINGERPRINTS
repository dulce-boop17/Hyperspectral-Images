# Libraries 
import spectral.io.envi as envi 
import matplotlib.pyplot as plt 
import numpy as np 

# Input directory and base name of the file  
input_folder = "C:/Users/Hp/Documents/Internals" 
base_name = "internals_apple" 

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
img = envi.open(hdr_file, raw_file) 

# Verify the dimensions of the loaded image 
print("Dimensions of the loaded image:", img.shape) 

# Select pixels seeds 
pixel_coords =  [ 
(552,462), (557,467), (557,462), (560,462), (562,462), 
(565,459), (565,464), (565,455), (572,472), (577,472), 
(577,467), (577,459), (572,459), (572,455), (570,450), 
(570,445), (570,440), (570,435), (570,445), (577,452), 
(577,456), (589,447), (589,455), (594,467), (599,459), 
(599,442), (606,435), (614,430), (619,425), (609,477), 
(604,464), (611,459), (611,457), (614,467), (616,462), 
(609,462), (611,469), (611,472), (611,479), (606,479), 
(599,479), (601,477), (604,469), (604,467), (611,462), 
(611,455), (619,450), (619,440), (619,443), (611,435), 
(624,455), (626,459), (631,469), (636,479), (636,489), 
(636,496), (628,496), (628,491), (633,491), (633,486), 
(633,485), (633,482), (633,477), (633,472), (638,469), 
(636,462), (643,459), (638,459), (646,455), (638,455), 
(643,455), (643,452), (643,450), (644,442), (638,442), 
(638,447), (636,455), (633,455), (633,447), (633,435), 
(638,432), (633,430), (633,428), (633,420), (638,413), 
(631,413), (626,413), (626,420), (633,428), (626,423), 
(619,423), (624,428), (626,437), (626,445), (626,443), 
(633,447), (633,455), (628,462), (587,445), (576,442) 
] 

# Select pixels pulp
pixel_coords = [ 
(447,437), (440,526), (435,445), (442,545), (435,459), 
(459,518), (430,482), (462,482), (508,430), (469,459), 
(467,354), (484,391), (476,408), (479,445), (481,467), 
(484,499), (491,528), (484,555), (484,572), (452,580), 
(467,594), (508,619), (521,619), (511,594), (513,609), 
(523,616), (506,597), (530,580), (540,560), (540,543), 
(523,533), (530,496), (530,479), (555,437), (548,413), 
(550,364), (540,337), (523,310), (548,310), (562,305), 
(579,352), (589,410), (616,398), (621,379), (611,347), 
(611,322), (614,504), (601,496), (601,604), (616,585), 
(631,562), (638,518), (638,388), (655,359), (663,337), 
(675,307), (695,288), (695,322), (682,356), (736,303), 
(724,329), (736,361), (722,391), (739,415), (729,425), 
(739,462), (754,489), (741,508), (741,535), (729,572), 
(746,560), (756,553), (766,562), (780,548), (795,531), 
(810,511), (817,591), (822,477), (812,423), (798,388), 
(771,342), (692,445), (700,415), (712,516), (727,533), 
(658,300), (692,317), (820,450), (798,447), (751,452), 
(631,521), (636,545), (670,560), (670,540), (702,575), 
(692,288), (552,445), (535,450), (545,415), (533,366) 
] 

# Verify if the coordinates are within the image range 
for y, x in pixel_coords: 
  if y >= rows or x >= cols: 
    print(f" Coordinate {(y, x)} is outside the image range. Image dimensions: ({rows}, {cols})") 
    continue 

# Obtain wavelengths from metadata, if present 
if 'wavelength' in metadata: 
  wavelengths = np.array(metadata['wavelength'], dtype=float) 
  print("\nWavelength:") 
  print(wavelengths) 
else: 

# Use band indices as “wavelengths” if not present in the metadata 
  wavelengths = np.arange(bands) 
  print("\nWavelengths not available in the metadata. Using band indices.") 

# Plot spectra of selected pixels 
plt.figure() 
for idx, (y, x) in enumerate(pixel_coords): 
  spectrum = img[y, x, :].flatten() 
  plt.plot(wavelengths, spectrum, label=f'Pixel ({y}, {x})') 

plt.xlabel('Wavelength ' if 'wavelength' in metadata else 'Bands') 
plt.ylabel('Intensity') 
plt.title('Spectral fingerprinting of selected apple pixels (seeds)') 
plt.title('Spectral fingerprinting of selected apple pixels (pulp)') 
plt.legend() 
plt.show() 

