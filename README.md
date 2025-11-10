# NYC Motor Vehicle Accidents: Impact of Weather on Crash Severity
# Overview

This project explores how weather conditions influence motor vehicle accident severity in New York City, using data from the Motor Vehicle Collisions  Crashes dataset published by the NYCDOT and NYPD on Data.gov
The analysis uses Python (Pandas, Seaborn, Scikit-learn) to clean, explore, and model accident data, examining patterns across boroughs, time periods, and weather types. Logistic Regression and Random Forest models were applied to predict whether an accident would be serious (injury or fatality) or non-serious based on weather and contextual factors.

# Objectives

Analyze how different weather conditions (rain, snow, fog, etc.) affect crash severity.

Identify which NYC boroughs and time periods experience the highest number of serious accidents.

Build predictive models to classify accident severity using environmental and situational features.

Provide data-driven insights to assist traffic authorities in improving safety planning.

# Key Research Questions

Which weather conditions are most associated with severe accidents?

Do rush hours see more serious crashes compared to off-hours?

How does crash distribution vary across boroughs?

Can ML models accurately predict accident severity using weather, time, and location data?

# Dataset

** Source: NYC Motor Vehicle Collisions – Crashes (Data.gov)

** Records: ~1.4 million observations (2016–Present)
# Key Columns Used:

crash_date, borough, zip_code, latitude, longitude

weather_description, precipitation, temp_max, temp_min

crash_time_period, number_of_injuries, number_of_deaths

contributing_factor_vehicles, serious_accident (target variable)

Data Preprocessing & Cleaning

# Steps performed:

Formatted Dates – Converted crash_date to datetime for time-based analysis.

Handled Missing Values – Dropped incomplete rows for key fields (borough, weather_description, number_of_injuries, number_of_deaths).

Filtered Invalid Entries – Removed “Unspecified” weather descriptions.

Feature Encoding – Converted categorical variables (borough, weather_description, crash_time_period) using one-hot encoding.

Feature Engineering – Created a binary target variable serious_accident (True = injury/death, False = property damage only).

# Exploratory Data Analysis (EDA)

Visualizations were used to explore accident trends across weather and time.

** 1. Accident Severity Distribution

Shows the imbalance between serious and non-serious crashes.

sns.countplot(x='serious_accident', data=accident_data_cleaned)

** 2. Borough-wise Accident Distribution

Compares accident counts across NYC boroughs (Brooklyn, Queens, Bronx, Manhattan, Staten Island).
→ Brooklyn and Queens show the highest accident frequencies.

** 3. Time Period Analysis

Visualizes how crashes vary by rush hours vs. off-hours peak times (12–3 PM, 3–6 PM) have higher frequencies.

** 4. Correlation Heatmap

Analyzes relationships among numeric columns like number_of_injuries, number_of_deaths, and precipitation.
→ Positive correlation between injuries and deaths; weak correlation with precipitation.

** 5. Weather Impact on Severity

Shows which weather conditions contribute to higher accident severity.
→ Conditions involving rain, fog, and snow correspond to higher proportions of serious crashes.

# Machine Learning Models

Two classification models were trained to predict serious_accident (binary):

1. Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

Accuracy: 72%

Precision (Non-serious): 0.72

Recall (Serious): 0.00 → Model failed to identify serious accidents due to class imbalance.

2. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

Accuracy: 70%

Precision: 0.36 (Serious), 0.72 (Non-serious)

Recall: 0.07 (Serious), 0.95 (Non-serious)

Random Forest captured nonlinear relationships better and slightly improved minority-class recognition.

# Model Evaluation

Metrics Used:

Accuracy

Precision

Recall

F1 Score

# Confusion Matrix

** Cross-validation for model robustness

Random Forest outperformed Logistic Regression in identifying serious accidents but both models struggled due to class imbalance, highlighting the need for resampling (SMOTE/oversampling) in future work.

# Visual Insights

Key findings from the analysis:

Brooklyn and Queens have the most crashes overall.

Rain and snow increase the risk of serious crashes.

Rush hours (especially 3–6 PM) are peak periods for accidents.

Logistic Regression underperformed due to skewed data; Random Forest showed more balanced results.

# Tech Stack

** Language: Python
** Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
** Environment: Jupyter Notebook / VS Code
** Dataset: NYC DOT / NYPD Open Data Portal

# Conclusion

Adverse weather and peak-hour traffic significantly influence crash severity in NYC.
Random Forest proved more effective than Logistic Regression in capturing nonlinear weather–severity relationships.
Addressing data imbalance could improve predictive accuracy for serious accidents.
Future work includes adding geospatial clustering and time-series forecasting for weather–accident prediction.
