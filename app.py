#/Users/saikirananugam/Desktop/ai_challenge/data.csv
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache
def load_data():
    file_path = '/Users/saikirananugam/Desktop/ai_challenge/data.csv'
    data = pd.read_csv(file_path)
    return data

# Train Random Forest Model
def train_model(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_depth=None)
    rf.fit(X_train, y_train)
    return rf

# Main function to run the app
def main():
    st.title("Tensile Strength Prediction")

    data = load_data()
    st.write("Dataset loaded successfully!")
    st.write(data.head())

    features = list(data.columns[:-3])  # Exclude the last 3 columns (YS, TS, EL)

    selected_features = st.multiselect("Select chemical properties to use for training", features)

    if selected_features:
        st.write("Selected features: ", selected_features)

        percentages = {}
        total_percentage = 0

        st.write("Specify the percentage (value) for each selected feature (sum must be 100)")
        for feature in selected_features:
            percentages[feature] = st.slider(f"Percentage (value) for {feature}", 0, 100, 0)
            total_percentage += percentages[feature]

        if total_percentage != 100:
            st.warning(f"Total percentage of selected features must be 100. Currently: {total_percentage}")
        else:
            X = data[selected_features]
            y = data['TS']

            # Split the dataset into training (80%) and testing (20%) sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = train_model(X_train, y_train)

            # Convert percentages to DataFrame for prediction
            user_input_df = pd.DataFrame([percentages])

            # Predict and evaluate
            if st.button("Predict"):
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                user_pred = model.predict(user_input_df)

                mae_train = mean_absolute_error(y_train, y_pred_train)
                mse_train = mean_squared_error(y_train, y_pred_train)
                rmse_train = sqrt(mse_train)

                mae_test = mean_absolute_error(y_test, y_pred_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                rmse_test = sqrt(mse_test)

                # Display results
                st.write("Predicted Tensile Strength for provided input values:", user_pred[0])
                st.write("Training set evaluation:")
                st.write(f"Mean Absolute Error (MAE): {mae_train}")
                st.write(f"Mean Squared Error (MSE): {mse_train}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse_train}")

                st.write("Testing set evaluation:")
                st.write(f"Mean Absolute Error (MAE): {mae_test}")
                st.write(f"Mean Squared Error (MSE): {mse_test}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse_test}")

if __name__ == "__main__":
    main()