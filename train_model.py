import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
df = pd.read_csv("student-mat.csv", sep=';')
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
X = df[['studytime', 'failures', 'absences', 'health', 'freetime']]
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "student_model.pkl")
print("✅ Model saved successfully!")
