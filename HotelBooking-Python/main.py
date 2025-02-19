# IMPORTING NECESSARY LIBRARIES
import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sort_dataframeby_monthorweek as sd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# LOADING THE HOTEL - BOOKING DATA
# Define the file name of the CSV file
file_name = 'hotel_bookings.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_name)

# Display the first few rows of the DataFrame
print(data.head())

# List all the column names in your DataFrame
print(data.columns)

# Check for null values in the DataFrame
null_values = data.isnull()

# Summarize the null values
null_summary = null_values.sum()

# Display the summary of null values
print("Summary of Null Values:")
print(null_summary)

# Check if there are any null values present
if null_summary.any():
    print("\nThere are null values in the DataFrame.")
else:
    print("\nThere are no null values in the DataFrame.")

# filling null values with zero
data.fillna(0, inplace=True)

# adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time
fil = (data['children'] == 0) & (data['adults'] == 0) & (data['babies'] == 0)
print(data[fil])

data = data[~fil]
print(data)

# LOADING THE HOTEL - REVIEW DATA
# Define the file name of the CSV file
file = 'hotel_reviews.csv'

# Read the CSV file into a pandas DataFrame
data_re = pd.read_csv(file)

# Display the first few rows of the DataFrame
print(data_re)

# List all the column names in your DataFrame
print(data_re.columns)

# Check for null values in the DataFrame
null_values = data_re.isnull()

# Summarize the null values
null_summary = null_values.sum()

# Display the summary of null values
print("Summary of Null Values in Hotel_Review data:")
print(null_summary)

# Check if there are any null values present
if null_summary.any():
    print("\nThere are null values in the Hotel_Review.")
else:
    print("\nThere are no null values in the Hotel_Review.")

# EXPLORATORY DATA ANALYSIS (EDA)
# Group the data by country and count the number of guests from each country
country_wise_guests = data[data['is_canceled'] == 0]['country'].value_counts().reset_index()
country_wise_guests.columns = ['country', 'No of guests']
print(country_wise_guests)

basemap = folium.Map()
guests_map = px.choropleth(country_wise_guests, locations=country_wise_guests['country'],
                           color=country_wise_guests['No of guests'], hover_name=country_wise_guests['country'])
guests_map.show()

# Boxplot room type
# Filter out canceled bookings
df = data[data['is_canceled'] == 0]

# Boxplot room type
boxplot = px.box(data_frame=df, x='reserved_room_type', y='adr', color='hotel', template='plotly')
boxplot.show()

# Explore the price per night
data_resort = df[df['hotel'] == 'Resort Hotel']
data_city = df[df['hotel'] == 'City Hotel']

resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()

final_hotel = resort_hotel.merge(city_hotel, on='arrival_date_month')
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']

print(final_hotel)

def sort_month(edata, column_name):
    return sd.Sort_Dataframeby_Month(edata, column_name)

final_prices = sort_month(final_hotel, 'month')

print(final_prices)

# Plot with matplotlib
plt.plot(final_prices['month'], final_prices['price_for_resort'], label='Resort Hotel')
plt.plot(final_prices['month'], final_prices['price_for_city_hotel'], label='City Hotel')

# Add title and labels
plt.title('Room price per night over the Months')
plt.xlabel('Month')
plt.ylabel('Price')
plt.legend()
# Rotate x-axis labels
plt.xticks(rotation=45)  # Rotate the labels by 45 degrees
# Save the plot
plt.savefig('line_plot.png')

# Show the plot
plt.show()

# The most busy month
resort_guests = data_resort['arrival_date_month'].value_counts().reset_index()
resort_guests.columns = ['month', 'no of guests']
print(resort_guests)
city_guests = data_city['arrival_date_month'].value_counts().reset_index()
city_guests.columns = ['month', 'no of guests']
print(city_guests)
final_guests = resort_guests.merge(city_guests, on='month')
final_guests.columns = ['month', 'no of guests in resort', 'no of guest in city hotel']
print(final_guests)
final_guests = sort_month(final_guests, 'month')
print(final_guests)
line_plot = px.line(final_guests, x='month', y=['no of guests in resort', 'no of guest in city hotel'],
                    title='Total number of guests per month', template='plotly')

line_plot.show()

# EDA ON HOTEL REVIEW DATA SET
# Descriptive statistics analysis
description_stats = data_re.describe()
print(description_stats)

# Visualize the distribution of variables
plt.figure(figsize=(10, 6))
sns.histplot(data=data_re, x='Average_Score', kde=True)
plt.title('Distribution of Average Score')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data_re, x='Reviewer_Score')
plt.title('Boxplot of Reviewer Score')
plt.xlabel('Reviewer Score')
plt.show()

# Correlation analysis
# Select only numeric columns
numeric_data_re = data_re.select_dtypes(include=['float64', 'int64'])
# Correlation analysis
correlation_matrix = numeric_data_re.corr()

# Select variables for pair plot
variables = ['Average_Score', 'Reviewer_Score', 'Review_Total_Negative_Word_Counts',
             'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts',
             'Total_Number_of_Reviews_Reviewer_Has_Given', 'days_since_review']

# Create pair plot
sns.pairplot(data_re[variables])
plt.suptitle('Pair Plot of Variables in data_re', y=1.02)
plt.show()

# DATA PREPROCESSING
# Check for the correlation
# Select only numeric columns
numeric_df = data.select_dtypes(include=['number'])
# Set the size of the figure
plt.figure(figsize=(24, 12))
# Compute the correlation matrix
corr = numeric_df.corr()
# Change the color scheme of the heatmap using the cmap parameter
sns.heatmap(corr, annot=True, linewidths=1, cmap='coolwarm')
# Show the heatmap
plt.show()

# Compute the correlation coefficients
correlation = numeric_df.corr()['is_canceled'].abs().sort_values(ascending=False)
# Print the correlation coefficients
print(correlation)

# Handling features
# Define a list of columns that are not useful
useless_col = ['days_in_waiting_list', 'arrival_date_year', 'arrival_date_year', 'assigned_room_type', 'booking_changes', 'reservation_status', 'country',
'days_in_waiting_list']
# Drop the unnecessary columns
data.drop(useless_col, axis=1, inplace=True)

# Get categorical columns
cat_cols = [col for col in data.columns if data[col].dtype == 'O']
# Create DataFrame with categorical columns
cat_df = data[cat_cols]
# Create DataFrame with numerical columns
num_df = data.drop(cat_cols, axis=1)
# Display categorical DataFrame
print("Categorical DataFrame:")
print(cat_df.head())

# Convert 'reservation_status_date' column to datetime
cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])
# Extract year, month, and day from 'reservation_status_date' column
cat_df['year'] = cat_df['reservation_status_date'].dt.year
cat_df['month'] = cat_df['reservation_status_date'].dt.month
cat_df['day'] = cat_df['reservation_status_date'].dt.day
# Drop 'reservation_status_date' and 'arrival_date_month' columns
cat_df.drop(['reservation_status_date', 'arrival_date_month'], axis=1, inplace=True)
# Display the updated DataFrame
print(cat_df.head())

# Encoding categorical variables
cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
cat_df['meal'] = cat_df['meal'].map({'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})
cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                                         'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})
cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
                                                                     'GDS': 4})
cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                 'L': 7, 'B': 8})
cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})
cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})
cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})
# Display the updated DataFrame
print(cat_df.head())

num_df = data.drop(columns=cat_cols, axis=1)
num_df.drop('is_canceled', axis=1, inplace=True)
print(num_df)

# Normalize numerical variables using logarithmic transformation
num_df['lead_time'] = np.log(num_df['lead_time'] + 1)
num_df['arrival_date_week_number'] = np.log(num_df['arrival_date_week_number'] + 1)
num_df['arrival_date_day_of_month'] = np.log(num_df['arrival_date_day_of_month'] + 1)
num_df['agent'] = np.log(num_df['agent'] + 1)
num_df['company'] = np.log(num_df['company'] + 1)
num_df['adr'] = np.log(num_df['adr'] + 1)
# Display the updated DataFrame
print(num_df.head())

# Fill missing values in the 'adr' column with the mean value
num_df['adr'] = num_df['adr'].fillna(value=num_df['adr'].mean())
print(num_df.head())

# Concatenating features (X) and labels (y)
X = pd.concat([cat_df, num_df], axis=1)
y = data['is_canceled']

# Printing the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Displaying the first few rows of the training set
print("First few rows of the training set (X_train):")
print(X_train.head())

# Displaying the first few rows of the test set
print("\nFirst few rows of the test set (X_test):")
print(X_test.head())

# Displaying the first few rows of the training labels
print("\nFirst few rows of the training labels (y_train):")
print(y_train.head())

# Displaying the first few rows of the test labels
print("\nFirst few rows of the test labels (y_test):")
print(y_test.head())

    # MODEL BUILDING
# Logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
conf = confusion_matrix(y_test, y_pred_lr)
clf_report = classification_report(y_test, y_pred_lr)

print(f"Accuracy Score of Logistic Regression is : {acc_lr}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# KNN

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
conf = confusion_matrix(y_test, y_pred_knn)
clf_report = classification_report(y_test, y_pred_knn)

print(f"Accuracy Score of KNN is: {acc_knn}")
print(f"Confusion Matrix:\n{conf}")
print(f"Classification Report:\n{clf_report}")

# Decision Tree Classifier
# Initialize and train the Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dtc = dtc.predict(X_test)

# Calculate accuracy score
acc_dtc = accuracy_score(y_test, y_pred_dtc)

# Compute confusion matrix
conf = confusion_matrix(y_test, y_pred_dtc)

# Generate classification report
clf_report = classification_report(y_test, y_pred_dtc)

# Print results
print(f"Accuracy Score of Decision Tree is : {acc_dtc}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# Random Forest Classifier
# Initialize and train the Random Forest classifier
rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rd_clf = rd_clf.predict(X_test)

# Calculate the accuracy score
acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)

# Calculate the confusion matrix
conf = confusion_matrix(y_test, y_pred_rd_clf)

# Generate the classification report
clf_report = classification_report(y_test, y_pred_rd_clf)

# Print the evaluation results
print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# Gradient Boosting Classifier
# Initialize and train the Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_gb = gb.predict(X_test)

# Calculate the accuracy score
acc_gb = accuracy_score(y_test, y_pred_gb)

# Generate the confusion matrix
conf = confusion_matrix(y_test, y_pred_gb)

# Generate the classification report
clf_report = classification_report(y_test, y_pred_gb)

# Print the results
print(f"Accuracy Score of Gradient Boosting Classifier is : {acc_gb}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# Extra Trees Classifier
etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

y_pred_etc = etc.predict(X_test)

acc_etc = accuracy_score(y_test, y_pred_etc)
conf = confusion_matrix(y_test, y_pred_etc)
clf_report = classification_report(y_test, y_pred_etc)

print(f"Accuracy Score of Extra Trees Classifier is : {acc_etc}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# MODEL PERFORMANCE COMPARISON
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier',
              'Gradient Boosting Classifier', 'Extra Trees Classifier'],
    'Score': [acc_lr, acc_knn, acc_dtc, acc_rd_clf, acc_gb, acc_etc]
})

print(models.sort_values(by='Score', ascending=False))

# Visualization the result
# Define the models and their corresponding accuracy scores
models = ['Random Forest', 'Extra Trees', 'Decision Tree', 'Gradient Boosting', 'KNN', 'Logistic Regression']
scores = [0.956967, 0.954157, 0.949207, 0.914017, 0.894682, 0.810544]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of Different Models')
plt.xticks(rotation=45)
plt.ylim(0.8, 1.0)  # Set y-axis limits
plt.tight_layout()
plt.show()


# FEATURE ENGINEERING ON HOTEL REVIEW DATA
# Define a function to map Reviewer_Score to review_type
def map_review_type(score):
    if score in ['Low_Reviewer_Score', 'Intermediate_Reviewer_Score']:
        return "Bad_review"
    elif score == 'High_Reviewer_Score':
        return "Good_review"
    else:
        return "Unknown"  # Handle any unexpected values

# Create the 'review_type' column based on 'Reviewer_Score'
data_re["review_type"] = data_re["Reviewer_Score"].apply(map_review_type)

# Select features and target variable
features = ['Average_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']
X = data_re[features]
y = data_re['review_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report for LR on Hotel Reviews:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# CACHE
from joblib import dump, load

# Save cache
dump(data, 'cache.joblib')

# Load cache
cached_data = load('cache.joblib')

