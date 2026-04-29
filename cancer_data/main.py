import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
print("Loading dataset...")
data = pd.read_csv("data.csv", header=None)

# Step 2: Add column names
columns = [
    "id","clump_thickness","cell_size","cell_shape","adhesion",
    "epithelial_size","bare_nuclei","chromatin","nucleoli",
    "mitoses","class"
]
data.columns = columns

# Step 3: Clean data
# Replace '?' with NaN and remove missing values
data.replace("?", np.nan, inplace=True)
data.dropna(inplace=True)

# Convert all columns to numeric
data = data.apply(pd.to_numeric)

# Step 4: Prepare input & output
X = data.drop(["id", "class"], axis=1)
y = data["class"]

# Convert class values: 2 → 0 (benign), 4 → 1 (malignant)
y = y.map({2: 0, 4: 1})

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Step 10: Visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Cancer Classification Confusion Matrix")
plt.show()