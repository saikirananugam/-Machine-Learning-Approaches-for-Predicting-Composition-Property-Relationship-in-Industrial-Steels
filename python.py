import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('/Users/saikirananugam/Desktop/ai_challenge/data.csv')  # Adjust the path as necessary

# Prepare the data (excluding target variables)
X = data.drop(columns=['TS', 'YS', 'EL'])
y = data['TS']

# Default values for the features
default_values = {
    'Fe': 99.0731, 'S': 0.0028, 'Cu': 0.2774, 'Ni': 0.1265, 'Cr': 0.4813,
    'Mo': 0.0032, 'V': 0.0022, 'Nb': 0.0005, 'Al': 0.0228, 'Ti': 0.0035,
    'B': 0.0002, 'Sn': 0.0024, 'As': 0.0026, 'Zr': 0.0006, 'Ca': 0.0007,
    'Pb': 0.0001, 'Sb': 0.0001, 'N': 0.0, 'O': 0.0, 'W': 0.0
}

# Set the title and display the team name
st.title('Tensile Strength Prediction')
st.markdown('### Team: Vekilov Bankers [G-0021]')

# Display the image with a specified width
st.image('/Users/saikirananugam/Desktop/ai_challenge/WhatsApp Image 2024-07-27 at 16.40.34.jpeg', caption='Steel Tensile Strength Analysis', width=400)

st.sidebar.header('Select Input Features')

# Allow user to select features (except Fe which is mandatory)
selected_features = st.sidebar.multiselect(
    'Select features for training (Fe is mandatory)', X.columns.tolist(), default=['Fe', 'Cr', 'Cu', 'Ni'])

# Ensure Fe is included in the selected features
if 'Fe' not in selected_features:
    selected_features.insert(0, 'Fe')

st.sidebar.header('Input Chemical Composition')

# Create input fields for each selected feature with default values
user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.number_input(
        f'{feature} (%)', min_value=0.0, max_value=100.0, value=default_values.get(feature, 0), step=0.1)

# Calculate the total percentage and round it to 2 decimal places
total_percentage = round(sum(user_input.values()), 2)

# Display a message if the total is not 100%
if total_percentage < 100.0:
    st.sidebar.info(
        f'Total percentage is {total_percentage:.2f}%. Add {100.0 - total_percentage:.2f}% more to reach 100%.')
elif total_percentage > 100.0:
    st.sidebar.warning(
        f'Total percentage is {total_percentage:.2f}%. Reduce by {total_percentage - 100.0:.2f}% to reach 100%.')
else:
    st.sidebar.success('Total percentage is exactly 100%!')

# Button to trigger training and prediction
if st.sidebar.button('Train and Predict'):
    if total_percentage != 100.0:
        st.error('Total percentage must be exactly 100%.')
    else:
        # Prepare the data with selected features
        X_selected = data[selected_features]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42)

        # Train the Random Forest Regressor with the selected features
        rf_model = RandomForestRegressor(
            n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_depth=None, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model with cross-validation for RMSE
        cv_scores_rmse = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores_rmse.mean())

        # Prepare the input data for prediction
        input_data = np.array([user_input[feature]
                              for feature in selected_features]).reshape(1, -1)

        # Make prediction
        y_pred = rf_model.predict(input_data)[0]

        # Evaluate the model on the test set
        y_pred_test = rf_model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        # Display the feature importances
        feature_importances = rf_model.feature_importances_
        feature_importances_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Display the results
        st.write(f'**Selected Features:** {selected_features}')
        st.write(f'**Predicted Tensile Strength (MPa):** {y_pred:.2f}')
        st.write(f'**Cross-Validation RMSE:** {cv_rmse:.2f}')
        st.write(f'**MAE (on test set):** {mae_test:.2f}')
        st.write(f'**MSE (on test set):** {mse_test:.2f}')
        st.write(f'**RMSE (on test set):** {rmse_test:.2f}')
        st.write(f'**Feature Importances:**')

        # Plot the feature importances
        st.bar_chart(feature_importances_df.set_index('Feature'))






# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load the dataset
# data = pd.read_csv('/Users/saikirananugam/Desktop/ai_challenge/data.csv')  # Adjust the path as necessary

# # Prepare the data (excluding target variables)
# X = data.drop(columns=['TS', 'YS', 'EL'])
# y = data['TS']

# # Default values for the features
# default_values = {
#     'Fe': 99.0731, 'S': 0.0028, 'Cu': 0.2774, 'Ni': 0.1265, 'Cr': 0.4813,
#     'Mo': 0.0032, 'V': 0.0022, 'Nb': 0.0005, 'Al': 0.0228, 'Ti': 0.0035,
#     'B': 0.0002, 'Sn': 0.0024, 'As': 0.0026, 'Zr': 0.0006, 'Ca': 0.0007,
#     'Pb': 0.0001, 'Sb': 0.0001, 'N': 0.0, 'O': 0.0, 'W': 0.0
# }

# # Set the title and display the team name
# st.title('Tensile Strength Prediction')
# st.markdown('### Team: Vekilov Bankers [G-0021]')

# # Display the image with a specified width
# st.image('/Users/saikirananugam/Desktop/ai_challenge/WhatsApp Image 2024-07-27 at 16.40.34.jpeg', caption='Steel Tensile Strength Analysis', width=400)

# st.sidebar.header('Select Input Features')

# # Allow user to select features (except Fe which is mandatory)
# selected_features = st.sidebar.multiselect(
#     'Select features for training (Fe is mandatory)', X.columns.tolist(), default=['Fe', 'Cr', 'Cu', 'Ni'])

# # Ensure Fe is included in the selected features
# if 'Fe' not in selected_features:
#     selected_features.insert(0, 'Fe')

# st.sidebar.header('Input Chemical Composition')

# # Create input fields for each selected feature with default values
# user_input = {}
# for feature in selected_features:
#     user_input[feature] = st.sidebar.number_input(
#         f'{feature} (%)', min_value=0.0, max_value=100.0, value=default_values.get(feature, 0), step=0.1)

# # Calculate the total percentage and round it to 2 decimal places
# total_percentage = round(sum(user_input.values()), 2)

# # Display a message if the total is not 100%
# if total_percentage < 100.0:
#     st.sidebar.info(
#         f'Total percentage is {total_percentage:.2f}%. Add {100.0 - total_percentage:.2f}% more to reach 100%.')
# elif total_percentage > 100.0:
#     st.sidebar.warning(
#         f'Total percentage is {total_percentage:.2f}%. Reduce by {total_percentage - 100.0:.2f}% to reach 100%.')
# else:
#     st.sidebar.success('Total percentage is exactly 100%!')

# # Button to trigger training and prediction
# if st.sidebar.button('Train and Predict'):
#     if total_percentage != 100.0:
#         st.error('Total percentage must be exactly 100%.')
#     else:
#         # Prepare the data with selected features
#         X_selected = data[selected_features]

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_selected, y, test_size=0.2, random_state=42)

#         # Train the Random Forest Regressor with the selected features
#         rf_model = RandomForestRegressor(
#             n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_depth=None, random_state=42)
#         rf_model.fit(X_train, y_train)

#         # Prepare the input data for prediction
#         input_data = np.array([user_input[feature]
#                               for feature in selected_features]).reshape(1, -1)

#         # Make prediction
#         y_pred = rf_model.predict(input_data)[0]

#         # Evaluate the model on the test set
#         y_pred_test = rf_model.predict(X_test)
#         mae_test = mean_absolute_error(y_test, y_pred_test)
#         mse_test = mean_squared_error(y_test, y_pred_test)
#         rmse_test = np.sqrt(mse_test)

#         # Display the results
#         st.write(f'**Selected Features:** {selected_features}')
#         st.write(f'**Predicted Tensile Strength (MPa):** {y_pred:.2f}')
#         st.write(f'**MAE (on test set):** {mae_test:.2f}')
#         st.write(f'**MSE (on test set):** {mse_test:.2f}')
#         st.write(f'**RMSE (on test set):** {rmse_test:.2f}')

