# Naan-mudhalvan-

https://vethaa007.github.io/Naan-mudhalvan-/


Air quality analysis and prediction in Tamilnadu 

Drive link- phase 1 presentation
https://drive.google.com/file/d/138lWO0hGQv66Fw0VO7qVJKIIi2TbMO_U/view?usp=drivesdk

Drive link -phase 2 presentation

https://drive.google.com/file/d/16J4TiwWIoDSujwUNq0wIO9uiYIX1N2BY/view?usp=drivesdk

Drive link- phase 3 presentation

https://drive.google.com/file/d/18EpoxdHWakhkwdiaOfegiPu9wAwljhrz/view?usp=drivesdk

Drive link-phase 4 presentation

https://drive.google.com/file/d/18xekkGSJQC-L-Q_dbyZC1js613ErLxfu/view?usp=drivesdk


Drive link - phase 5 presentation 

https://drive.google.com/file/d/1BMrKwZWM6lKagWUNOZlzz48MAnCgmpwo/view?usp=drivesdk

Summary:
The project on "Air Quality Analysis and Prediction in Tamil Nadu" aims to develop a predictive model to assess and forecast air quality parameters by utilizing historical and real-time data. The project involves multiple key components

Objectives: 
Problem Identification:

Addressing the challenges posed by air pollution in Tamil Nadu due to various factors such as industrialization, urbanization, and environmental conditions.

Data Utilization:

 Utilizing diverse datasets containing air quality parameters, meteorological data, and geographical information to build a predictive model.

Prediction and Mitigation:

  Developing a machine learning-based model to predict air quality levels and contributing to strategies for better environmental management and public health improvement.

Methodology: Design Thinking Approach:

 Employing a user-centered and iterative design thinking process to empathize, define, ideate, prototype, test, implement, and iterate through the development phases.

Data Preprocessing:

 Cleaning, handling missing values, and scaling the dataset to ensure data quality and uniformity for modeling.

Exploratory Data Analysis:

 Analyzing data through statistics, visualizations, and correlations to understand underlying patterns and relationships between variables.

Machine Learning Model Development:

 Selecting appropriate algorithms and developing models to predict air quality parameters based on historical and current data.
Deliverables: Code and Model 

Development:

 Providing Python-based scripts and programs for data preprocessing, exploratory data analysis, model development, and evaluation.

Documentation:

 Offering detailed documentation outlining problem statements, design thinking approach, dataset description, preprocessing steps, model selection, and innovative techniques applied during the development.


Program:

import pandas as pd Load the dataset data = pd.read_csv('air_quality_data.csv')

Checking for missing values missing_values = data.isnull().sum() print("Missing Values:") print(missing_values)

Handling missing values (this is a simple example) Replace missing values in SO2, NO2, and RSPM/PM10 columns with their means data['SO2'].fillna(data['SO2'].mean(), inplace=True) data['NO2'].fillna(data['NO2'].mean(), inplace=True) data['RSPM/PM10'].fillna(data['RSPM/PM10'].mean(), inplace=True)

Data normalization (scaling the data to a common range) from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() data[['SO2', 'NO2', 'RSPM/PM10']] = scaler.fit_transform(data[['SO2', 'NO2', 'RSPM/PM10']])

Feature selection (in this example, all columns except 'Location' are used as features) features = data.drop('Location', axis=1)

You can proceed with further analysis, visualization, or model development using the 'features' DataFrame

For example, performing exploratory data analysis, building predictive models, or generating visualizations.

Save the preprocessed data to a new CSV file if needed data.to_csv('preprocessed_air_quality_data.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

 Load the air quality dataset
data = pd.read_csv('air_quality_data.csv')  # Replace 'air_quality_data.csv' with your dataset filename

Handling missing values (if any)
data.fillna(data.mean(), inplace=True)  # Filling missing values with the mean of each column

 Feature selection and splitting the dataset into features and target variable
X = data[['SO2', 'NO2']]  # Features: SO2 and NO2
y = data['RSPM/PM10']     # Target variable: RSPM/PM10

Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Model fitting
model = LinearRegression()
model.fit(X_train, y_train)

Prediction on the test set
y_pred = model.predict(X_test)

Model evaluation
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

import pandas as pd

# Load the dataset
data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your actual dataset file

# Handling missing values
data.fillna(method='ffill', inplace=True)  # Forward-fill missing values

# Removing duplicates if any
data.drop_duplicates(inplace=True)

# Outlier detection and handling (assuming RSPM/PM10, SO2, NO2 are columns)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    return df

data = remove_outliers(data, 'RSPM/PM10')
data = remove_outliers(data, 'SO2')
data = remove_outliers(data, 'NO2')

# Format or preprocess data further if needed (e.g., date parsing, feature engineering)

# Save preprocessed data to a new CSV file
data.to_csv('preprocessed_data.csv', index=False)
