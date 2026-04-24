# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and prepare the dataset (X and Y values).
2. Create and train the simple linear regression model using the dataset.
3. Accept input (hours studied) and predict the marks using the trained model.
4. Display the predicted result and visualize the data using a graph.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.

Developed by:Sakthipriya R 
RegisterNumber:212225220088

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df

X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
/*
  
*/
```

## Output:

<img width="1340" height="665" alt="Screenshot 2026-04-24 112138" src="https://github.com/user-attachments/assets/3f8965f6-73d4-4a9f-a24f-ebd110242b52" />
<img width="1082" height="783" alt="Screenshot 2026-04-24 112146" src="https://github.com/user-attachments/assets/cc5bbd92-5c32-4bdb-b8c9-038e89fe9d48" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
