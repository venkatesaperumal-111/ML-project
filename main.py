import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Data Loading
    print("Loading California Housing dataset...")
    # The California Housing dataset contains features like median income, house age, average rooms, etc.
    # The target variable is the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['Price'] = california.target

    # 2. Data Preprocessing
    print("\n--- Data Preprocessing ---")
    print("Checking for null values:")
    print(df.isnull().sum()) # It's good practice to check for missing data.

    # Separate features (X) and target variable (y)
    X = df.drop('Price', axis=1)
    y = df['Price']

    # 3. Train-Test Split
    # We split the data so we can train on one portion and test the model on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nData split complete. Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # 4. Feature Scaling
    # Machine learning algorithms perform better when numerical input variables are scaled to a standard range.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Note: we only fit on train data, but transform both!

    # 5. Model Training and Evaluation - Linear Regression
    print("\n--- Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)

    # Evaluate Linear Regression
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)
    print(f"Linear Regression Mean Squared Error (MSE): {lr_mse:.4f}")
    print(f"Linear Regression R-squared (R2): {lr_r2:.4f}")

    # 6. Model Training and Evaluation - Random Forest Regressor
    print("\n--- Random Forest Regressor ---")
    # Random Forest is an ensemble method that usually yields higher accuracy for complex datasets.
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_predictions = rf_model.predict(X_test_scaled)

    # Evaluate Random Forest
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    print(f"Random Forest Mean Squared Error (MSE): {rf_mse:.4f}")
    print(f"Random Forest R-squared (R2): {rf_r2:.4f}")

    # 7. Simple Prediction Function
    print("\n--- Making a New Prediction ---")
    # Let's take the first row of our dataset to simulate a "new" house
    sample_features = X.iloc[0].values
    feature_names = X.columns.tolist()
    
    print("New House Features:")
    for name, value in zip(feature_names, sample_features):
        print(f"  {name}: {value}")

    predicted_price = predict_house_price(rf_model, scaler, sample_features)
    actual_price = y.iloc[0]
    
    print(f"\nPredicted Price (Random Forest): ${predicted_price * 100000:,.2f}")
    print(f"Actual Price: ${actual_price * 100000:,.2f}")


def predict_house_price(model, scaler, features):
    """
    Predict the price of a house given its features.
    
    Args:
        model: Trained scikit-learn model.
        scaler: Fitted StandardScaler used during training.
        features: List or array of house features in the same order as training data.
        
    Returns:
        Predicted price (scaled in 100,000s).
    """
    # 1. Convert features to a 2D numpy array (1 row, N columns)
    features_array = np.array(features).reshape(1, -1)
    
    # 2. Scale the features using the fitted scaler
    features_scaled = scaler.transform(features_array)
    
    # 3. Predict the price
    prediction = model.predict(features_scaled)
    return prediction[0]

if __name__ == "__main__":
    main()
