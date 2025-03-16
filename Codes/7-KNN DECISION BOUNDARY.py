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
base_name = "internals_artichoke" 

# HDR and RAW files  
hdr_file = f"{input_folder}/{base_name}.hdr" 
raw_file = f"{input_folder}/{base_name}.raw" 

# Read metadata 
metadata = envi.read_envi_header(hdr_file) 

# Print metadata 
print("HDR file metadata:") 
print(metadata) 

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
stem_coords = [ 
    (141,279), (141,290), (154,282), (149,312), (151,325), 
    (154,334), (155,349), (163,361), (153,373), (150,382), 
    (155,392), (157,399), (158,407), (165,413), (172,407), 
    (172,395), (174,386), (180,378), (190,361), (194,348), 
    (195,339), (201,330), (202,316), (198,304), (203,295), 
    (214,288), (220,306), (220,321), (221,343), (225,352), 
    (229,368), (231,380), (231,400), (235,416), (239,426), 
    (251,429), (253,419), (256,398), (262,386), (266,363), 
    (271,353), (273,335), (280,320), (289,313), (304,314), 
    (306,336), (303,360), (306,378), (308,390), (312,403), 
    (296,416), (290,433), (297,441), (304,431), (313,417), 
    (323,402), (327,386), (333,371), (337,353), (337,338), 
    (345,319), (360,325), (352,340), (350,362), (350,378), 
    (352,402), (343,416), (337,435), (340,447), (335,454), 
    (356,459), (370,462), (378,453), (382,466), (379,449), 
    (379,436), (385,426), (389,408), (398,397), (407,382), 
    (413,367), (417,356), (426,349), (439,352), (436,364), 
    (436,392), (434,412), (424,427), (424,445), (425,461), 
    (410,468), (414,477), (425,483), (442,477), (457,477), 
    (453,456), (456,443), (472,435), (485,413), (497,401) 
] 
 
crown_coords =[ 
    (598,400), (611,383), (623,348), (623,391), (620,439), 
    (599,493), (613,523), (577,540), (542,580), (553,601), 
    (539,619), (554,625), (558,645), (597,659), (560,668), 
    (573,679), (584,702), (604,717), (605,660), (618,637), 
    (636,635), (651,597), (662,597), (677,540), (687,511), 
    (696,478), (703,443), (722,397), (731,358), (743,305), 
    (758,276), (772,222), (794,219), (818,219), (834,226), 
    (879,219), (868,220), (858,242), (825,255), (816,278), 
    (815,301), (818,326), (804,340), (809,361), (803,379), 
    (802,409), (794,432), (801,448), (798,466), (803,483), 
    (802,505), (804,538), (814,556), (806,585), (808,604), 
    (813,624), (814,653), (811,672), (798,687), (808,714), 
    (780,730), (755,753), (748,766), (833,761), (805,757), 
    (854,705), (885,647), (879,677), (878,659), (879,608), 
    (897,585), (893,551), (905,538), (900,505), (898,468), 
    (893,337), (881,300), (884,280), (875,217), (833,220), 
    (913,346), (941,348), (944,369), (953,388), (969,397), 
    (993,422), (1015,420), (1019,452), (1018,476), (1001,510), 
    (996,551), (1003,594), (963,658), (959,634), (975,598), 
    (952,524), (1037,425), (1037,426), (901,522), (899,467)
]
 
# Verify if the coordinates are within the image range 
valid_crown_coords = [(y, x) for y, x in crown_coords if y < rows and x < cols] 
valid_stem_coords = [(y, x) for y, x in stem_coords if y < rows and x < cols] 
 
# Labels : 0 = crown, 1 = stem 
labels_crown = [0] * len(valid_crown_coords) 
labels_stem = [1] * len(valid_stem_coords) 
 
# Linking of data and labels 
coords = valid_crown_coords + valid_stem_coords 
labels = labels_crown + labels_stem 
 
# Extraer los espectros de los píxeles seleccionados 
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

# Create a grid of points for plotting decision boundaries 
h = 0.1  

# Plot pairs of statistical moments 
plt.figure(figsize=(15, 12)) 

# Mean vs. Standard Deviation 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = knn.predict(np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 2])), np.full_like(xx.ravel(), 
np.mean(X[:, 3]))]) 
Z = Z.reshape(xx.shape) 
plt.subplot(2, 3, 1) 
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm') 
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k') 
plt.title('Mean vs Standard Deviation') 
plt.xlabel('Mean') 
plt.ylabel('Standard Deviation') 

# Mean vs. Skewness 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = knn.predict(np.c_[xx.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 1])), yy.ravel(), np.full_like(xx.ravel(), 
np.mean(X[:, 3]))]) 
Z = Z.reshape(xx.shape) 
plt.subplot(2, 3, 2) 
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm') 
plt.scatter(X_test[:, 0], X_test[:, 2], c=y_pred, cmap='coolwarm', edgecolors='k') 
plt.title('Mean vs Skewness') 
plt.xlabel('Mean') 
plt.ylabel('Skewness') 

# Mean vs Kurtosis 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = knn.predict(np.c_[xx.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 1])), np.full_like(xx.ravel(), np.mean(X[:, 
2])), yy.ravel()]) 
Z = Z.reshape(xx.shape) 
plt.subplot(2, 3, 3) 
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm') 
plt.scatter(X_test[:, 0], X_test[:, 3], c=y_pred, cmap='coolwarm', edgecolors='k') 
plt.title('Mean vs Kurtosis') 
plt.xlabel('Mean') 
plt.ylabel('Kurtosis') 

# Standard Deviation vs Skewness 
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = knn.predict(np.c_[np.full_like(xx.ravel(), np.mean(X[:, 0])), xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), 
np.mean(X[:, 3]))]) 
Z = Z.reshape(xx.shape) 
plt.subplot(2, 3, 4) 
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm') 
plt.scatter(X_test[:, 1], X_test[:, 2], c=y_pred, cmap='coolwarm', edgecolors='k') 
plt.title('Standard Deviation vs Skewness') 
plt.xlabel('Standard Deviation') 
plt.ylabel('Skewness') 

# Standard Deviation vs Kurtosis 
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = knn.predict(np.c_[np.full_like(xx.ravel(), np.mean(X[:, 0])), xx.ravel(), np.full_like(xx.ravel(), np.mean(X[:, 
2])), yy.ravel()]) 
Z = Z.reshape(xx.shape) 
plt.subplot(2, 3, 5) 
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm') 
plt.scatter(X_test[:, 1], X_test[:, 3], c=y_pred, cmap='coolwarm', edgecolors='k') 
plt.title('Standard Deviation vs Kurtosis') 
plt.xlabel('Standard Deviation') 
plt.ylabel('Kurtosis') 

# Skewness vs Kurtosis 
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1 
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = knn.predict(np.c_[np.full_like(xx.ravel(), np.mean(X[:, 0])), np.full_like(xx.ravel(), np.mean(X[:, 1])), 
xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape) 
plt.subplot(2, 3, 6) 
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm') 
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred, cmap='coolwarm', edgecolors='k') 
plt.title('Skewness vs Kurtosis') 
plt.xlabel('Skewness') 
plt.ylabel('Kurtosis') 

plt.tight_layout() 
plt.show()
