# Bike_data_usuage

#Part 1
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap


# Define file paths for the CSV files containing bike usage data for various weeks in April 2023
files = [
    "C:/Users/.............../363JourneyDataExtract27Mar2023-02Apr2023.csv",
    "C:/Users/................/364JourneyDataExtract03Apr2023-09Apr2023.csv",
    "C:/Users/.........../365JourneyDataExtract10Apr2023-16Apr2023.csv",
    "C:/Users/............./366JourneyDataExtract17Apr2023-23Apr2023.csv",
    "C:/Users/........./367JourneyDataExtract24Apr2023-30Apr2023.csv"
]

# Load and combine CSV files
data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

# Convert date columns to datetime format
data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
data['End date'] = pd.to_datetime(data['End date'], errors='coerce')

# Remove rows with invalid dates
data.dropna(subset=['Start date', 'End date'], inplace=True)

# Ensure numeric columns are in the correct format
numeric_cols = ['Number', 'Start station number', 'End station number', 'Bike number', 'Total duration (ms)']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in numeric columns
data.dropna(subset=numeric_cols, inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Filter data for April 2023
data = data[(data['Start date'] >= '2023-04-01') & (data['Start date'] <= '2023-04-30 23:59:59')]

# Save the cleaned data to a new CSV file
output_path = 'C:/Users/..../April2023_Cleaned_Journey_Data.csv'
data.to_csv(output_path, index=False)

print(f"Filtered data saved to: {output_path}")

##########CUSTOMER ANALYSIS############

# Set the viridis palette
sns.set_palette("viridis")

# Load the data

journey_data = pd.read_csv(output_path)

# Convert 'Start date' and 'End date' to datetime format
journey_data['Start date'] = pd.to_datetime(journey_data['Start date'])
journey_data['End date'] = pd.to_datetime(journey_data['End date'])

# Convert 'Total duration' to timedelta
journey_data['Total duration'] = pd.to_timedelta(journey_data['Total duration'])

# Convert 'Total duration' to minutes for easier analysis
journey_data['Total duration (minutes)'] = journey_data['Total duration'].dt.total_seconds() / 60

# Visualize the distribution of trip durations
plt.figure(figsize=(12, 6))
sns.histplot(journey_data['Total duration (minutes)'], bins=100, kde=True)
plt.title('Distribution of Trip Durations in April')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.show()

# Determine key percentiles
percentiles = journey_data['Total duration (minutes)'].quantile([0.90, 0.95, 0.99])
print("Key Percentiles (minutes):\n", percentiles)

# Filter out outliers based on the 99th percentile cutoff
cutoff = percentiles[0.99]
filtered_journey_data = journey_data[journey_data['Total duration (minutes)'] <= cutoff]

# Visualize the distribution without outliers
plt.figure(figsize=(12, 6))
sns.histplot(filtered_journey_data['Total duration (minutes)'], bins=50, kde=True)
plt.title('Distribution of Trip Durations in April (Filtered)')
plt.xlabel('Trip duration (minutes)')
plt.ylabel('Frequency')
plt.show()

# Analyze daily rental frequency
daily_rentals = filtered_journey_data['Start date'].dt.date.value_counts().sort_index()
print("Rental Frequency:\n", daily_rentals)

# Plot number of trips per day
plt.figure(figsize=(12, 6))
daily_rentals.plot(kind='bar', color=sns.color_palette("viridis", n_colors=1))
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Daily Trips in April 2023')
plt.show()

# Analyze hourly rental patterns
filtered_journey_data['hour_of_day'] = filtered_journey_data['Start date'].dt.hour
hourly_rentals = filtered_journey_data.groupby('hour_of_day').size()

plt.figure(figsize=(14, 7))
hourly_rentals.plot(kind='bar', color=sns.color_palette("magma", n_colors=1))
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rentals')
plt.title('Bike Rentals by Hour in April')
plt.show()

# Identify peak rental hours
peak_hours = hourly_rentals.sort_values(ascending=False).head(5)
print("Peak rental times (hours of the day):\n", peak_hours)

# Analyze day of the week rental patterns
filtered_journey_data['day_of_week'] = filtered_journey_data['Start date'].dt.day_name()
weekly_rentals = filtered_journey_data.groupby('day_of_week').size().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(14, 7))
weekly_rentals.plot(kind='bar', color=sns.color_palette("coolwarm", n_colors=6))
plt.xlabel('Day of Week')
plt.ylabel('Number of Rentals')
plt.title('Bike Rentals by Day of the Week in April')
plt.show()

# Compare weekday vs weekend rental patterns
filtered_journey_data['is_weekend'] = filtered_journey_data['day_of_week'].isin(['Saturday', 'Sunday'])
weekend_vs_weekday_rentals = filtered_journey_data.groupby(['hour_of_day', 'is_weekend']).size().unstack()

# Line Plot of Rentals by Hour for Weekdays and Weekends
weekend_hourly = filtered_journey_data[filtered_journey_data['is_weekend']].groupby('hour_of_day').size()
weekday_hourly = filtered_journey_data[~filtered_journey_data['is_weekend']].groupby('hour_of_day').size()

plt.figure(figsize=(14, 7))
plt.plot(weekday_hourly, label='Weekday', marker='o', color=sns.color_palette("viridis", n_colors=1)[0])
plt.plot(weekend_hourly, label='Weekend', marker='o', color=sns.color_palette("viridis", n_colors=2)[1])
plt.title('Bike Rentals by Hour: Weekdays vs Weekends in April')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rentals')
plt.legend()
plt.show()


#######Geospatial Analysis######

# Group by start and end stations and count the number of trips
start_station_counts = journey_data['Start station'].value_counts().head(5)
end_station_counts = journey_data['End station'].value_counts().head(5)

print("Top 5 Start Stations:\n", start_station_counts)
print("\nTop 5 End Stations:\n", end_station_counts)

# Coordinates for the required start and end stations
stations_info = pd.DataFrame({
    'Station Name': [
        'Hyde Park Corner, Hyde Park', 
        'Waterloo Station 3, Waterloo', 
        'Black Lion Gate, Kensington Gardens', 
        'Belgrove Street, King\'s Cross', 
        'Waterloo Station 1, Waterloo',
        'Hop Exchange, The Borough'
    ],
    'Latitude': [
        51.502777, 51.5033, 51.5058, 51.5308, 51.5033, 51.5052
    ],
    'Longitude': [
        -0.151250, -0.1143, -0.1790, -0.1224, -0.1138, -0.0936
    ]
})

# Create a mapping dictionary if necessary
station_name_mapping = {
    "Belgrove Street , King's Cross": "Belgrove Street, King's Cross",
    # Add more mappings if necessary
}

# Apply the mapping to the start and end station counts
start_station_counts = start_station_counts.rename(index=station_name_mapping)
end_station_counts = end_station_counts.rename(index=station_name_mapping)

# Create dataframes for start and end stations with coordinates
start_stations = pd.DataFrame({
    'Station Name': start_station_counts.index,
    'Latitude': [stations_info.loc[stations_info['Station Name'] == name, 'Latitude'].values[0] for name in start_station_counts.index],
    'Longitude': [stations_info.loc[stations_info['Station Name'] == name, 'Longitude'].values[0] for name in start_station_counts.index],
    'Counts': start_station_counts.values
})

end_stations = pd.DataFrame({
    'Station Name': end_station_counts.index,
    'Latitude': [stations_info.loc[stations_info['Station Name'] == name, 'Latitude'].values[0] for name in end_station_counts.index],
    'Longitude': [stations_info.loc[stations_info['Station Name'] == name, 'Longitude'].values[0] for name in end_station_counts.index],
    'Counts': end_station_counts.values
})

# Plot top 5 start stations
plt.figure(figsize=(10, 6))
plt.bar(start_stations['Station Name'], start_stations['Counts'], color='blue')
plt.title('Top 5 Start Stations in April 2023')
plt.xlabel('Start Station')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()

# Plot top 5 end stations
plt.figure(figsize=(10, 6))
plt.bar(end_stations['Station Name'], end_stations['Counts'], color='red')
plt.title('Top 5 End Stations in April 2023')
plt.xlabel('End Station')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()

# Map the top start stations
map = folium.Map(location=[51.5074, -0.1278], zoom_start=13)  # Centered at London

# Add start stations to the map with distinct color and add to a feature group for interactivity
start_stations_fg = folium.FeatureGroup(name='Start Stations')
for _, station in start_stations.iterrows():
    folium.Marker(
        location=[station['Latitude'], station['Longitude']],
        popup=station['Station Name'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(start_stations_fg)
start_stations_fg.add_to(map)

# Add end stations to the map with a different distinct color and add to a feature group for interactivity
end_stations_fg = folium.FeatureGroup(name='End Stations')
for _, station in end_stations.iterrows():
    folium.Marker(
        location=[station['Latitude'], station['Longitude']],
        popup=station['Station Name'],
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(end_stations_fg)
end_stations_fg.add_to(map)

# Draw lines between start and end stations with different colors and add to a feature group for interactivity
lines_fg = folium.FeatureGroup(name='Routes')
for _, start_station in start_stations.iterrows():
    for _, end_station in end_stations.iterrows():
        folium.PolyLine(
            locations=[
                [start_station['Latitude'], start_station['Longitude']],
                [end_station['Latitude'], end_station['Longitude']]
            ],
            color='purple',  # Color for the line
            weight=2.5,
            opacity=0.5
        ).add_to(lines_fg)
lines_fg.add_to(map)

# Add heatmap for start stations and add to a feature group for interactivity
heatmap_start_fg = folium.FeatureGroup(name='Start Stations Heatmap')
heat_data_start = [[row['Latitude'], row['Longitude']] for index, row in start_stations.iterrows()]
HeatMap(heat_data_start, gradient={0.4: 'blue', 0.6: 'cyan', 0.8: 'lime', 1: 'yellow'}).add_to(heatmap_start_fg)
heatmap_start_fg.add_to(map)

# Add heatmap for end stations and add to a feature group for interactivity
heatmap_end_fg = folium.FeatureGroup(name='End Stations Heatmap')
heat_data_end = [[row['Latitude'], row['Longitude']] for index, row in end_stations.iterrows()]
HeatMap(heat_data_end, gradient={0.4: 'red', 0.6: 'orange', 0.8: 'yellow', 1: 'green'}).add_to(heatmap_end_fg)
heatmap_end_fg.add_to(map)

# Add layer control to the map
folium.LayerControl().add_to(map)

# Save the map to an HTML file to view in a browser
map.save("C:/Users/...../stations_map.html")

print("Map created and saveed as stations_map.html")


#######MULTIVARIATE ANALYSIS####

# Convert date columns to datetime
journey_data['Start date'] = pd.to_datetime(journey_data['Start date'])
journey_data['End date'] = pd.to_datetime(journey_data['End date'])

# Extract additional time-related features
journey_data['Hour'] = journey_data['Start date'].dt.hour  # Extract hour from 'Start date'
journey_data['Day of Week'] = journey_data['Start date'].dt.day_name()  # Extract day name from 'Start date'
journey_data['Day'] = journey_data['Start date'].dt.day  # Extract day from 'Start date'

# Calculate trip duration in minutes
journey_data['Trip Duration (min)'] = journey_data['Total duration (ms)'] / 60000

# Data Cleaning: Drop rows with missing or incorrect values
journey_data.dropna(subset=['Start date', 'End date', 'Total duration (ms)'], inplace=True)

# Analyze correlations between variables
correlation_matrix = journey_data[['Trip Duration (min)', 'Hour', 'Day of Week']].copy()

# Convert 'Day of Week' to categorical codes
correlation_matrix['Day of Week'] = correlation_matrix['Day of Week'].astype('category').cat.codes

# Calculate correlation matrix
correlation_results = correlation_matrix.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_results, annot=True, cmap='Blues', center=0)
plt.title('Correlation Heatmap for April 2023 Bike Trips')
plt.show()

# Prepare the pivot table for hourly rentals by day of the week
pivot_table = journey_data.pivot_table(index='Hour', columns='Day of Week', values='Start date', aggfunc='count')

# Plot heatmap of hourly rentals by day of the week
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt='d')
plt.title('Hourly Rentals by Day of the Week for April 2023')
plt.xlabel('Day of the Week')
plt.ylabel('Hour of the Day')
plt.show()

# Combined scatter plot of trip duration vs. hour and line plot of average trip duration by hour and day of the week
fig, ax2 = plt.subplots(figsize=(14, 10))

# Scatter plot of trip duration vs. hour of the day
sns.scatterplot(data=journey_data, x='Hour', y='Trip Duration (min)', hue='Day of Week', palette='viridis', ax=ax2)
ax2.set_title('Trip Duration vs. Hour of the Day for April 2023')
ax2.set_xlabel('Hour of the Day')
ax2.set_ylabel('Trip Duration (minutes)')
ax2.legend(title='Day of the Week')

plt.show()

# Additional Analysis: Boxplot of trip duration by day of the week
plt.figure(figsize=(12, 8))
sns.boxplot(data=journey_data, x='Day of Week', y='Trip Duration (min)', palette='coolwarm')
plt.title('Trip Duration by Day of the Week for April 2023')
plt.xlabel('Day of the Week')
plt.ylabel('Trip Duration (minutes)')
plt.xticks(rotation=45)
plt.show()

# Calculate average trip duration and trip counts for each start and end station
start_station_stats = journey_data.groupby('Start station').agg(
    avg_duration=('Trip Duration (min)', 'mean'),
    trip_count=('Start station', 'size')
).reset_index()

end_station_stats = journey_data.groupby('End station').agg(
    avg_duration=('Trip Duration (min)', 'mean'),
    trip_count=('End station', 'size')
).reset_index()

# Combine start and end station stats for plotting
start_station_stats['Station Type'] = 'Start'
end_station_stats['Station Type'] = 'End'
combined_stats = pd.concat([start_station_stats, end_station_stats], ignore_index=True)

# Scatter plot to visualize the relationship for start and end stations
plt.figure(figsize=(12, 8))
sns.scatterplot(data=combined_stats, x='trip_count', y='avg_duration', hue='Station Type', palette={'Start': 'blue', 'End': 'red'}, s=100)
plt.title('Relationship between Trip Duration and Number of Trips for Start and End Stations')
plt.xlabel('Number of Trips')
plt.ylabel('Average Trip Duration (minutes)')
plt.xscale('log')
plt.yscale('log')
plt.legend(title='Station Type')
plt.show()

# Print the correlation for start and end stations
correlation_start = start_station_stats[['avg_duration', 'trip_count']].corr().iloc[0, 1]
correlation_end = end_station_stats[['avg_duration', 'trip_count']].corr().iloc[0, 1]
print(f'\nCorrelation between average trip duration and trip count for start stations: {correlation_start:.2f}')
print(f'Correlation between average trip duration and trip count for end stations: {correlation_end:.2f}')


##Part 2




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load the datasets
bike_data_file = 'C:/Users/............./merged_data_month.csv'
weather_data_file = 'C:/Users/........../london 2023-01-01 to 2023-04-30.csv'

bike_data = pd.read_csv(bike_data_file, low_memory=False)
weather_data = pd.read_csv(weather_data_file)

# Convert date columns to datetime
bike_data['Start date'] = pd.to_datetime(bike_data['Start date'], errors='coerce')
bike_data['End date'] = pd.to_datetime(bike_data['End date'], errors='coerce')
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], errors='coerce')

# Filter data to include only records from 1st Feb 00:00 AM to 30th April 23:59
start_date = '2023-02-01 00:00:00'
end_date = '2023-04-30 23:59:59'

filtered_bike_data = bike_data[(bike_data['Start date'] >= start_date) & (bike_data['Start date'] <= end_date)]
filtered_weather_data = weather_data[(weather_data['datetime'] >= start_date) & (weather_data['datetime'] <= end_date)]

# Filter the bike data for "Hyde Park Corner, Hyde Park"
station_name = "Hyde Park Corner, Hyde Park"
station_bike_data = filtered_bike_data[(filtered_bike_data['Start station'] == station_name) | (filtered_bike_data['End station'] == station_name)]

# Save the cleaned data if needed
station_bike_data.to_csv('C:/Users/..........S/station_bike_data_filtered.csv', index=False)
filtered_weather_data.to_csv('C:/Users/.........../weather_data_filtered.csv', index=False)

# Group by day and analyze daily trends within the specified period
filtered_weather_data['day'] = filtered_weather_data['datetime'].dt.date

# Group by day and count the number of records per day
daily_weather_trends = filtered_weather_data.groupby('day').size()

######FEATURE ENGINEERING######

# Load cleaned bike usage data
cleaned_bike_data_file = 'C:/Users/......../station_bike_data_filtered.csv'
cleaned_bike_data = pd.read_csv(cleaned_bike_data_file, low_memory=False)

# Load cleaned weather data
cleaned_weather_data_file = 'C:/Users/........./weather_data_filtered.csv'
cleaned_weather_data = pd.read_csv(cleaned_weather_data_file)

# Convert Start Date and End Date to datetime
cleaned_bike_data['Start date'] = pd.to_datetime(cleaned_bike_data['Start date'], errors='coerce')
cleaned_bike_data['End date'] = pd.to_datetime(cleaned_bike_data['End date'], errors='coerce')

# Convert the datetime column in weather data to datetime
cleaned_weather_data['datetime'] = pd.to_datetime(cleaned_weather_data['datetime'], errors='coerce')

# Extract time-based features from the Start date
cleaned_bike_data['hour'] = cleaned_bike_data['Start date'].dt.hour
cleaned_bike_data['day_of_week'] = cleaned_bike_data['Start date'].dt.dayofweek
cleaned_bike_data['month'] = cleaned_bike_data['Start date'].dt.month
cleaned_bike_data['day'] = cleaned_bike_data['Start date'].dt.day

# Selecting relevant weather features
weather_features = cleaned_weather_data[['datetime', 'temp', 'humidity', 'windspeed', 'solarradiation', 'feelslike']]

# Rename 'datetime' to 'date' to facilitate merging later
weather_features.rename(columns={'datetime': 'date'}, inplace=True)

# Extract the date part to merge with bike data
weather_features['date'] = weather_features['date'].dt.date

# Extract the date part to merge with weather data
cleaned_bike_data['date'] = cleaned_bike_data['Start date'].dt.date

# Analyze daily trends in bike usage
daily_bike_trends = cleaned_bike_data.groupby('date').size().reset_index(name='Bike Rentals')

# Plot daily trends in bike usage
plt.figure(figsize=(12, 6))
plt.plot(daily_bike_trends['date'], daily_bike_trends['Bike Rentals'])
plt.title('Daily Trends in Bike Usage')
plt.xlabel('Date')
plt.ylabel('Number of Bike Rentals')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_bike_trends_plot.png')  # Save the plot as a file
plt.show()

# Aggregate weather data to daily level
daily_weather = weather_features.groupby('date').agg({
    'temp': 'mean',
    'humidity': 'mean',
    'windspeed': 'mean',
    'solarradiation': 'mean',
    'feelslike': 'mean'
}).reset_index()

# Merge daily bike rentals with daily weather data on date
bike_weather_merged = pd.merge(daily_bike_trends, daily_weather, on='date', how='inner')

# Engineer new features: Interaction terms between weather conditions and bike rentals
bike_weather_merged['temp_bike_interaction'] = bike_weather_merged['Bike Rentals'] * bike_weather_merged['temp']
bike_weather_merged['humidity_bike_interaction'] = bike_weather_merged['Bike Rentals'] * bike_weather_merged['humidity']
bike_weather_merged['windspeed_bike_interaction'] = bike_weather_merged['Bike Rentals'] * bike_weather_merged['windspeed']
bike_weather_merged['feelslike_bike_interaction'] = bike_weather_merged['Bike Rentals'] * bike_weather_merged['feelslike']

# Conduct correlative tests and statistical analyses
required_columns = ['Bike Rentals', 'temp', 'humidity', 'windspeed', 
                    'temp_bike_interaction', 'humidity_bike_interaction', 'windspeed_bike_interaction',
                    'solarradiation', 'feelslike_bike_interaction']

# Compute the correlation matrix with available columns
available_columns = [col for col in required_columns if col in bike_weather_merged.columns]
correlation_matrix = bike_weather_merged[available_columns].corr()
# Plot heatmap of the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # Save the heatmap as a file
plt.show()

# Normalize numerical variables
scaler = StandardScaler()
numerical_features = ['temp', 'humidity', 'windspeed', 
                      'temp_bike_interaction', 'humidity_bike_interaction', 'windspeed_bike_interaction',
                       'solarradiation', 'feelslike_bike_interaction']
scaled_features = scaler.fit_transform(bike_weather_merged[numerical_features])
scaled_features_df = pd.DataFrame(scaled_features, columns=numerical_features)

# Combine all features into a final dataframe
final_data = pd.concat([bike_weather_merged[['Bike Rentals']], scaled_features_df], axis=1)

# Save the final dataframe for further analysis
final_data.to_csv('C:/Users/......../final_bike_weather_data.csv', index=False)

# Add the feelslike_bike_interaction to the dataset
bike_weather_merged['feelslike_bike_interaction'] = bike_weather_merged['Bike Rentals'] * bike_weather_merged['feelslike']

# Subset data to include the new interaction term
subset_data = bike_weather_merged[['Bike Rentals', 'temp_bike_interaction', 'humidity_bike_interaction', 'windspeed_bike_interaction', 'feelslike_bike_interaction']]

# Create the pairplot
sns.pairplot(subset_data)
plt.suptitle('Relationships between Bike Rentals and Interaction Terms', y=1.02)
plt.savefig('interaction_relationships_pairplot_updated.png')
plt.show()

######MODEL TRAINING###########

# Load the final dataset
final_dataset_file = 'C:/Users/....../final_bike_weather_data.csv'
final_bike_weather_data = pd.read_csv(final_dataset_file)

# Define input features and target variable
X = final_bike_weather_data.drop('Bike Rentals', axis=1)
y = final_bike_weather_data['Bike Rentals']

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Perform clustering
kmeans = KMeans(n_clusters=1, random_state=42)
bike_weather_merged['Cluster'] = kmeans.fit_predict(bike_weather_merged.drop(columns=['Bike Rentals', 'date']))
#bike_weather_merged['Cluster'] = kmeans.fit_predict(X_scaled)
# Split the data into features and target variable
X = bike_weather_merged.drop(columns=['Bike Rentals', 'date'])
y = bike_weather_merged['Bike Rentals']

# Standardize the features
# scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Function to train and evaluate models
def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=500, random_state=42)
    }
    
    performance_metrics = {}
    model_predictions = {}
    
    for model_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)
        model_predictions[model_name] = predictions
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        performance_metrics[model_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    
    return performance_metrics, model_predictions

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_with_clusters, y, test_size=0.3, random_state=42, shuffle=False)

# Train and evaluate models
performance_metrics, model_predictions = train_evaluate_models(X_train, X_test, y_train, y_test)

# Display performance metrics
performance_metrics_df = pd.DataFrame(performance_metrics).T
print("\nModel Performance Metrics:")
print(tabulate(performance_metrics_df, headers='keys', tablefmt='psql'))

# Save the performance metrics to CSV files
performance_metrics_df.to_csv('C:/Users/......./model_performance_metrics.csv', index=True)

# Improved plotting of actual vs predicted bike rentals for each model
plt.figure(figsize=(14, 8))

# Plot actual bike rentals as scatter plot
plt.scatter(range(len(y_test)), y_test, label='Actual Bike Rentals', color='orange', marker='o')

# Plot predicted bike rentals for each model as line plot
for model_name, predictions in model_predictions.items():
    plt.plot(predictions, label=f'Predicted by {model_name}')

# Adding labels, title, legend, and grid
plt.xlabel('Index')
plt.ylabel('Bike Rentals')
plt.title('Actual vs Predicted Bike Rentals (Clustering)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside plot
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()



# Hyperparameter tuning for Random Forest using Grid Search
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
rf_predictions = best_rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest - Best Parameters:", grid_search_rf.best_params_)
print("Random Forest - Best CV Score:", -grid_search_rf.best_score_)
print("Random Forest - Test Set MSE:", rf_mse)
print("Random Forest - Test Set R2:", rf_r2)

# Hyperparameter tuning for XGBoost using Random Search
param_dist_xgb = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': np.arange(3, 10, 1),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'subsample': np.linspace(0.5, 1.0, 10),
    'colsample_bytree': np.linspace(0.5, 1.0, 10)
}

random_search_xgb = RandomizedSearchCV(estimator=XGBRegressor(objective='reg:squarederror', random_state=42), param_distributions=param_dist_xgb, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search_xgb.fit(X_train, y_train)
best_xgb_model = random_search_xgb.best_estimator_
xgb_predictions = best_xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\nXGBoost - Best Parameters:", random_search_xgb.best_params_)
print("XGBoost - Best CV Score:", -random_search_xgb.best_score_)
print("XGBoost - Test Set MSE:", xgb_mse)
print("XGBoost - Test Set R2:", xgb_r2)

# Hyperparameter tuning for Gradient Boosting using Grid Search
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search_gb = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param_grid_gb, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train, y_train)
best_gb_model = grid_search_gb.best_estimator_
gb_predictions = best_gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)

print("\nGradient Boosting - Best Parameters:", grid_search_gb.best_params_)
print("Gradient Boosting - Best CV Score:", -grid_search_gb.best_score_)
print("Gradient Boosting - Test Set MSE:", gb_mse)
print("Gradient Boosting - Test Set R2:", gb_r2)

# Hyperparameter tuning for Decision Tree using Grid Search
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42), param_grid=param_grid_dt, cv=5, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)
best_dt_model = grid_search_dt.best_estimator_
dt_predictions = best_dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

print("\nDecision Tree - Best Parameters:", grid_search_dt.best_params_)
print("Decision Tree - Best CV Score:", -grid_search_dt.best_score_)
print("Decision Tree - Test Set MSE:", dt_mse)
print("Decision Tree - Test Set R2:", dt_r2)

# Hyperparameter tuning for Neural Network using Grid Search
param_grid_nn = {
    'hidden_layer_sizes': [(50, 50, 50), (100, 100, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

grid_search_nn = GridSearchCV(estimator=MLPRegressor(max_iter=500, random_state=42), param_grid=param_grid_nn, cv=5, scoring='neg_mean_squared_error')
grid_search_nn.fit(X_train, y_train)
best_nn_model = grid_search_nn.best_estimator_
nn_predictions = best_nn_model.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)

print("\nNeural Network - Best Parameters:", grid_search_nn.best_params_)
print("Neural Network - Best CV Score:", -grid_search_nn.best_score_)
print("Neural Network - Test Set MSE:", nn_mse)
print("Neural Network - Test Set R2:", nn_r2)

# Compare performance of the best models
print("\nPerformance Comparison:")
print(f"Random Forest: MSE = {rf_mse}, R2 = {rf_r2}")
print(f"XGBoost: MSE = {xgb_mse}, R2 = {xgb_r2}")
print(f"Gradient Boosting: MSE = {gb_mse}, R2 = {gb_r2}")
print(f"Decision Tree: MSE = {dt_mse}, R2 = {dt_r2}")
print(f"Neural Network: MSE = {nn_mse}, R2 = {nn_r2}")




