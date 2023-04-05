import mlflow
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/raw/diabetes.csv')

df_sub = df.loc[:, ['Pregnancies', 'Glucose', 'BloodPressure', 'Outcome']]
df_sub.head(2)

df_train, df_test = train_test_split(df_sub, test_size=0.3, random_state=123)
model = KNeighborsClassifier().fit(X = df_train.iloc[:, :3], y = df_train.Outcome)

eval_data = df_test

with mlflow.start_run() as run:
    # Log the baseline model to MLflow
    mlflow.sklearn.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")

    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets = "Outcome",
        model_type = "classifier",
        evaluators = ["default"],
    )
