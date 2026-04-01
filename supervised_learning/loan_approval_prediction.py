import kagglehub

# Download latest version
path = kagglehub.dataset_download("tanishaj225/loancsv")

print("Path to dataset files:", path)

!unzip archive.zip

import pandas as pd
df = pd.read_csv("loan.csv")
df.shape

df.head()

df.isnull().sum()

df.dropna(inplace=True)

df = pd.get_dummies(df)

X = df.drop(["Loan_Status_N", "Loan_Status_Y"], axis=1)
y = df["Loan_Status_Y"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

import joblib

joblib.dump(model, "loan_model.pkl")

