import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r"C:\Users\snigd\ML\student-mat.csv")

# Show basic info
print("Dataset shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())

# Plot G3 (final grade) distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='G3', data=data)
plt.title('Distribution of Final Grade (G3)', fontsize=18)
plt.xlabel('G3 Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.show()

# Calculate and print gender count
male_students = len(data[data['sex'] == 'M'])
female_students = len(data[data['sex'] == 'F'])
print(f"\nNumber of male students: {male_students}")
print(f"Number of female students: {female_students}")

# Add average grade column (optional)
data['GradeAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3

# Drop unnecessary columns
data_dum = data.drop(['school', 'age'], axis=1)

# Binary categorical mappings
binary_map = {'yes': 1, 'no': 0}
for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
    data_dum[col] = data_dum[col].map(binary_map)

# Gender mapping
data_dum['sex'] = data_dum['sex'].map({'F': 1, 'M': 0})

# Address and Pstatus mappings
data_dum['address'] = data_dum['address'].map({'U': 1, 'R': 0})
data_dum['Pstatus'] = data_dum['Pstatus'].map({'T': 1, 'A': 0})

# Job and reason mappings
job_map = {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}
data_dum['Mjob'] = data_dum['Mjob'].map(job_map)
data_dum['Fjob'] = data_dum['Fjob'].map(job_map)

reason_map = {'home': 0, 'reputation': 1, 'course': 2, 'other': 3}
data_dum['reason'] = data_dum['reason'].map(reason_map)

guardian_map = {'mother': 0, 'father': 1, 'other': 2}
data_dum['guardian'] = data_dum['guardian'].map(guardian_map)

# Check for any remaining non-numeric columns
non_numeric = data_dum.select_dtypes(include=['object']).columns
if len(non_numeric) > 0:
    print("Unprocessed non-numeric columns:", non_numeric.tolist())
else:
    print("\n✅ All categorical features have been encoded successfully!")

# Features and Target
X = data_dum.drop(columns=['G3', 'GradeAvg'])  # Using G3 as target
y = data_dum['G3']

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=18)

# Train Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Predict
ypred = model.predict(xtest)


