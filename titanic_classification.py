import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

url = "train.csv"
df = pd.read_csv(url)

df.head()

shape_01 = df.shape # rows and columns
info_01 = df.info() # column types and nulls
describe_01 = df.describe() # summary stats for numbers

# 2. Data Cleaning

# Check for missing values
missing_values = df.isnull().sum()

# Drop or fill missing values

# Remove rows with any missing values
df = df.dropna()

# Fill missing values with a default
df = df.fillna(0)

df = df.drop(["Name", "Cabin", "Ticket", "SibSp", "Parch"], axis=1)

df = df[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]]

# 3. Feature Engineering

# Encode categorical columns (e.g., sex, embarked, class)
# Use pd.get_dummies() or LabelEncoder
# Choose a target column: survived
# Choose features like: sex, pclass, age, fare, embarked, etc.
df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], drop_first=True)

dataset = df.head()
print(dataset)

# 4. Train-Test Split + Logistic Regression

# Split the data into train/test sets using train_test_split
# Train a Logistic Regression model from sklearn
# Make predictions

# Assume 'Survived' is your target variable
x = df.drop("Survived", axis=1)
y = df["Survived"]

# Step 1: Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Train shape: ", x_train.shape)
print("Test shape: ", x_test.shape)

# Step 2: Train a classifier
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Step 3: Make Predictions
y_pred = model.predict(x_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)

# Step 5: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Classification Report

# This gives you:
# Precision: out of predicted survivors, how many actually survived?
# Recall: out of all who survived, how many did we catch?
# F1-score: balance between precision and recall
print(classification_report(y_test, y_pred))

# Day 6 – Plotting & Explanation

# Plot feature distributions for survived vs not

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, palette="Set1")
plt.title("Age Distribution by Survival")
plt.show()

# Fare distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Fare", hue="Survived", bins=30, kde=True, palette="Set2")
plt.title("Fare Distribution by Survival")
plt.show()

# Sex distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex_male", hue="Survived", palette="Set3")
plt.title("Sex Distribution by Survival")
plt.xlabel("Male")
plt.show()
