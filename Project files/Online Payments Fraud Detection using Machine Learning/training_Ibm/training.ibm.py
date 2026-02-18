import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_path):
    # 1. Load Dataset
    print("Loading data...")
    df = pd.read_csv(data_path)

    # 2. Data Pre-processing (As per project flow)
    # Dropping columns that aren't useful for numerical ML models
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

    # Handling Categorical Values: Encode 'type' (CASH_OUT, TRANSFER, etc.)
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    
    # Save the encoder if you need to decode later
    # pickle.dump(le, open('label_encoder.pkl', 'wb'))

    # 3. Feature Selection
    # Standard features for the Online Payment Fraud dataset
    X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    y = df['isFraud']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Building & Hyperparameter Tuning
    print("Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save the model for IBM App Integration
    model_filename = 'payments.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model successfully saved as {model_filename}")

if __name__ == "__main__":
    # Ensure your dataset CSV is in the same folder or provide the full path
    train_model('online_payments_data.csv')