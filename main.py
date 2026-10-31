
import pandas as pd

data = pd.read_csv("telecom_data.csv")

print("Dataset loaded successfully!")
print("Dataset size:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nMissing values in each column:")
print(data.isnull().sum())

print("\nCustomer churn counts:")
print(data["Churn"].value_counts())

print("\nCustomer churn percentages:")
print(data["Churn"].value_counts(normalize=True).mul(100).round(2))

print("\nMissing values:")
print(data.isnull().sum().sum())


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = data.drop("Churn", axis=1)
y = data["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_columns = X_train.select_dtypes(include="number").columns
categorical_columns = X_train.select_dtypes(exclude="number").columns

numeric_steps = Pipeline([
    ("missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_steps = Pipeline([
    ("missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preparation = ColumnTransformer([
    ("numbers", numeric_steps, numeric_columns),
    ("categories", categorical_steps, categorical_columns)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

for name, algorithm in models.items():
    model = Pipeline([
        ("preparation", preparation),
        ("algorithm", algorithm)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name} Results")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    if name == "Random Forest":
        print("Random Forest Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))


