# Libraries  
import spectral.io.envi as envi 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
from scipy.stats import skew, kurtosis 

# Input directory and base name of the file 
input_folder = "C:/Users/Hp/Documents/Internals" 
base_name = "internals_mushroom" 

# HDR and RAW files 
hdr_file = f"{input_folder}/{base_name}.hdr" 
raw_file = f"{input_folder}/{base_name}.raw" 

# Read metadata  
metadata = envi.read_envi_header(hdr_file) 

# Image dimensions 
rows = int(metadata['lines']) 
cols = int(metadata['samples']) 
bands = int(metadata['bands']) 
dtype = metadata['data type'] 

print("\nImage dimensions:")  
print("Rows:", rows)  
print("Columns:", cols)  
print("Number of bands:", bands)  
print("Data type:", dtype) 

# Load hyperspectral image 
img = envi.open(hdr_file, raw_file).load() 

# Select pixels 
pulp_coords = [ 
(566,364), (571,364), (567,373), (566,376), (568,371), 
(576,364), (573,367), (578,366), (574,369), (574,372), 
(580,365), (578,370), (585,369), (582,372), (585,366), 
(583,369), (589,369), (591,380), (586,386), (570,398), 
(579,397), (574,407), (581,405), (576,414), (581,410), 
(583,404), (585,409), (578,422), (584,417), (590,415), 
(587,421), (596,427), (595,434), (588,440), (588,439), 
(590,446), (597,439), (597,446), (597,454), (603,457), 
(603,450), (607,456), (611,463), (610,458), (615,463), 
(621,466), (621,461), (624,466), (622,457), (620,446), 
(618,437), (616,428), (611,421), (606,401), (613,406), 
(608,402), (609,389), (610,383), (611,374), (613,368), 
(614,363), (624,364), (626,376), (629,390), (621,390), 
(629,365), (621,390), (629,365), (622,359), (626,413), 
(635,418), (634,418), (625,426), (436,627), (644,432), 
(650,441), (643,451), (651,450), (655,449), (658,448), 
(656,452), (654,456), (649,460), (647,457), (638,453), 
(642,458), (645,464), (642,471), (637,469), (638,474), 
(631,474), (626,471), (623,469), (614,468), (595,376), 
(612,410), (608,402), (600,393), (604,383), (631,411) 
] 
seed_coords =   [ 
(498,351), (497,359), (597,368), (489,375), (500,384), 
(501,392), (505,398), (504,391), (504,386), (503,377), 
(506,369), (507,361), (509,354), (509,346), (509,342), 
(509,336), (508,332), (510,351), (513,406), (515,402), 
(519,401), (525,397), (530,394), (528,386), (528,376), 
(530,373), (531,367), (532,362), (536,339), (538,323), 
(539,311), (542,305), (548,422), (548,415), (548,409), 
(546,403), (547,395), (551,384), (553,372), (553,365), 
(558,357), (569,425), (568,418), (566,410), (565,404), 
(565,398), (571,351), (575,338), (576,324), (577,311), 
(578,302), (585,302), (588,323), (618,353), (616,340), 
(613,332), (614,325), (618,318), (620,312), (618,304), 
(621,298), (625,300), (626,314), (628,327), (631,336), 
(635,348), (641,357), (637,363), (633,369), (632,376), 
(633,383), (640,382), (644,385), (648,387), (645,398), 
(644,412), (656,412), (666,403), (663,394), (666,384), 
(667,378), (660,372), (664,366), (664,360), (666,344), 
(663,336), (668,326), (674,321), (679,328), (688,340), 
(694,346), (698,354), (698,366), (696,381), (685,396), 
(685,396), (676,411), (672,406), (658,416), (648,370) 
] 

# Verify if the coordinates are within the image range 
valid_seed_coords = [(y, x) for y, x in seed_coords if y < rows and x < cols] 
valid_pulp_coords = [(y, x) for y, x in pulp_coords if y < rows and x < cols] 

# Labels : 0 = seeds, 1 = pulp  
labels_seed = [0] * len(valid_seed_coords) 
labels_pulp = [1] * len(valid_pulp_coords) 

# Linking of data and labels 
coords = valid_seed_coords + valid_pulp_coords 
labels = labels_seed + labels_pulp 

# Extract spectra of the selected pixels 
spectra = [img[y, x, :].flatten() for (y, x) in coords] 

# Calculate the statistical moments for each spectrum 
def compute_moments(spectrum): 
  return [ 
    np.mean(spectrum), 
    np.std(spectrum), 
    skew(spectrum), 
    kurtosis(spectrum) 
] 

# Apply the function to all spectra  
moments = [compute_moments(spectrum) for spectrum in spectra] 

# Training and test data (75 training, 25 test per class) 
X = np.array(moments) 
y = np.array(labels) 

# Select 75 training and 25 test data per class 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

# Create and train the KNN model 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) 

# Predictions in the test set 
y_pred = knn.predict(X_test) 

# Calculate and print the overall accuracy 
accuracy = accuracy_score(y_test, y_pred) 
print(f" Overall accuracy of the KNN model: {accuracy:.2f}") 

# Calculate and print the accuracy for each combination of statistical moments  
from itertools import combinations 
momentos_nombres = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis'] 
momentos_indices = list(range(len(momentos_nombres))) 
print("\nAccuracy by combination of statistical moments:") 

for comb in combinations(momentos_indices, 2):   
# Select the characteristics corresponding to the combination 
  X_train_comb = X_train[:, comb] 
  X_test_comb = X_test[:, comb] 

# Train a new model with the selected features 
  knn_comb = KNeighborsClassifier(n_neighbors=3) 
  knn_comb.fit(X_train_comb, y_train) 

# Predicting and calculating the accuracy for this combination 
  y_pred_comb = knn_comb.predict(X_test_comb) 
  accuracy_comb = accuracy_score(y_test, y_pred_comb) 

# Names of the moments in the combination 
  comb_nombres = f"{momentos_nombres[comb[0]]} vs {momentos_nombres[comb[1]]}" 
  print(f"{comb_nombres}: {accuracy_comb:.2f}") 

# Plot pairs of statistical moments 
plt.figure(figsize=(15, 12)) 

plt.subplot(2, 3, 1) 
plt.scatter([f[0] for f in X_test], [f[1] for f in X_test], c=y_pred, cmap='coolwarm', alpha=0.7) 
plt.title('Mean vs Standard Deviation') 
plt.xlabel('Mean') 
plt.ylabel('Standard Deviation') 
for i, txt in enumerate(y_pred): 
plt.annotate(txt, ([f[0] for f in X_test][i], [f[1] for f in X_test][i])) 

plt.subplot(2, 3, 2) 
plt.scatter([f[0] for f in X_test], [f[2] for f in X_test], c=y_pred, cmap='coolwarm', alpha=0.7) 
plt.title('Mean vs Skewness') 
plt.xlabel('Mean') 
plt.ylabel('Skewness') 
for i, txt in enumerate(y_pred): 
plt.annotate(txt, ([f[0] for f in X_test][i], [f[2] for f in X_test][i])) 

plt.subplot(2, 3, 3) 
plt.scatter([f[0] for f in X_test], [f[3] for f in X_test], c=y_pred, cmap='coolwarm', alpha=0.7) 
plt.title('Mean vs Kurtosis') 
plt.xlabel('Mean') 
plt.ylabel('Kurtosis') 
for i, txt in enumerate(y_pred): 
plt.annotate(txt, ([f[0] for f in X_test][i], [f[3] for f in X_test][i])) 

plt.subplot(2, 3, 4) 
plt.scatter([f[1] for f in X_test], [f[2] for f in X_test], c=y_pred, cmap='coolwarm', alpha=0.7) 
plt.title('Standard Deviation vs Skewness') 
plt.xlabel('Standard Deviation') 
plt.ylabel('Skewness') 
for i, txt in enumerate(y_pred): 
plt.annotate(txt, ([f[1] for f in X_test][i], [f[2] for f in X_test][i])) 

plt.subplot(2, 3, 5) 
plt.scatter([f[1] for f in X_test], [f[3] for f in X_test], c=y_pred, cmap='coolwarm', alpha=0.7) 
plt.title('Standard Deviation vs Kurtosis') 
plt.xlabel('Standard Deviation') 
plt.ylabel('Kurtosis') 
for i, txt in enumerate(y_pred): 
plt.annotate(txt, ([f[1] for f in X_test][i], [f[3] for f in X_test][i])) 

plt.subplot(2, 3, 6) 
plt.scatter([f[2] for f in X_test], [f[3] for f in X_test], c=y_pred, cmap='coolwarm', alpha=0.7) 
plt.title('Skewness vs Kurtosis') 
plt.xlabel('Skewness') 
plt.ylabel('Kurtosis') 
for i, txt in enumerate(y_pred): 
plt.annotate(txt, ([f[2] for f in X_test][i], [f[3] for f in X_test][i])) 
plt.tight_layout() 
plt.show()
