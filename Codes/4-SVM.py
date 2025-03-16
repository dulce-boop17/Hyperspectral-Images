# Libraries 
import spectral.io.envi as envi 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
from scipy.stats import skew, kurtosis 
from sklearn.svm import SVC 

# Input directory and base name of the file 
input_folder = "C:/Users/Hp/Documents/Internals" 
base_name = "internals_ apple" 

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

seed_coords = [ 
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

# Create and train the MSV model 
svm = SVC(kernel = 'rbf' , C = 1 , gamma = 0.125) 
svm.fit(X_train, y_train) 

# Predictions in the test set 
y_pred = svm.predict(X_test) 

# Calculate and print the overall accuracy 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Overall accuracy of the SVM model: {accuracy:.2f}") 

# Calculate and print the accuracy for each combination of statistical moments 
combinations = [ 
(0, 1, 'Mean vs Standard Deviation'), 
(0, 2, 'Mean vs Skewness'), 
(0, 3, 'Mean vs Kurtosis'), 
(1, 2, 'Standard Deviation vs Skewness'), 
(1, 3, 'Standard Deviation vs Kurtosis'), 
(2, 3, 'Skewness vs Kurtosis') 
] 

for i, j, label in combinations: 
# Create subset of characteristics for combination 
  X_train_subset = X_train[:, [i, j]] 
  X_test_subset = X_test[:, [i, j]] 

# Training and prediction with the model 
svm_subset = SVC(kernel='rbf', C=1, gamma=0.125) 
svm_subset.fit(X_train_subset, y_train) 
y_pred_subset = svm_subset.predict(X_test_subset) 

# Calculate accuracy 
subset_accuracy = accuracy_score(y_test, y_pred_subset) 
print(f" Accuracy for {label}: {subset_accuracy:.2f}") 

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
