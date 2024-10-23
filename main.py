import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(C:\Users\Achuth Kaja\heart_disease_prediction):
    """Load dataset from a CSV file."""
    data = pd.read_csv(C:\Users\Achuth Kaja\heart_disease_prediction)
    return data

def preprocess_data(data):
    """Preprocess the data for modeling."""
    # Assuming the target variable is 'target' and all other columns are features
    X = data.drop('target', axis=1)  # Features
    y = data['target']                # Target variable
    return X, y

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved as '{file_path}'")

def main():
    # Load dataset
    data = load_data('data/dataset_heart.csv')

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    save_model(model, 'model/heart_disease_model.pkl')

if __name__ == '__main__':
    main()
