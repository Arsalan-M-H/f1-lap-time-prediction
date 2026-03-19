import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


pd.set_option('display.max_columns', None)

# Load lap times csv dataset
lap_times = pd.read_csv('lap_times.csv')

# Load pit stops dataset
pit_stops = pd.read_csv('pit_stops.csv')

# Load races csv dataset
races = pd.read_csv('races.csv')

# Merge lap_times and pit stops to understand which laps were pit laps for drivers in each race
lap_times_with_pit = pd.merge(lap_times, pit_stops, on=['driverId', 'lap', 'raceId'],how='left')
print(lap_times_with_pit.head())

# Dropping columns that are not required
lap_times_with_pit = lap_times_with_pit.drop(['duration', 'time_x', 'time_y'], axis = 1)

# Filling in null or nan values with 0 for non pit_laps, setting pit_lap to one on the lap a driver pitted
lap_times_with_pit['stop'] = lap_times_with_pit['stop'].fillna(0)
lap_times_with_pit['milliseconds_y'] = lap_times_with_pit['milliseconds_y'].fillna(0)
lap_times_with_pit['stop'] = (lap_times_with_pit['stop'] != 0).astype(int)
print(lap_times_with_pit.head())

# Changing column names for better readability
lap_times_with_pit = lap_times_with_pit.rename(columns = {'milliseconds_y' : 'pit_stop_duration', 'stop' : 'pit_stop', 'milliseconds_x' : 'lap_time'})
print(lap_times_with_pit.columns)

# Converting lap times and pit stops from milliseconds to seconds, 
lap_times_with_pit['lap_time'] = lap_times_with_pit['lap_time']/1000
lap_times_with_pit['pit_stop_duration'] = lap_times_with_pit['pit_stop_duration']/1000
print(lap_times_with_pit['lap_time'].describe())

# Merging the dataset with races to include Circuit ID in our final dataset
lap_times_with_pit = pd.merge(lap_times_with_pit, races, on = 'raceId', how = 'left')
print(lap_times_with_pit.columns)

# Dropping all columns that are not required
lap_times_with_pit = lap_times_with_pit.drop(['date','time','url','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date','fp3_time','quali_date','quali_time','sprint_date','sprint_time'], axis = 1)
print(lap_times_with_pit.columns)

# Filtering only races from 2014 onwards as this represents the hybrid era of F1, including earlier years will not assist our model as current car technology would not be the same
lap_times_with_pit = lap_times_with_pit[lap_times_with_pit['year'] >= 2014]
print(lap_times_with_pit['name'].unique())


# Creating a dataframe grouped by raceId, aggregating the std and mean lap times to discover wet races, wet races would have a higher average lap time and a higher std due to a wide spread in lap times
wet_race_identifying1 = lap_times_with_pit.groupby(['raceId', 'name'])['lap_time'].std()
wet_race_identifying2 = lap_times_with_pit.groupby(['raceId', 'name'])['lap_time'].mean()
wet_race_identifying3 = pd.merge(wet_race_identifying1, wet_race_identifying2, on = ['raceId', 'name'], how = 'inner')
# Lap times identified with higher average lap time and higher std
wet_race_id = [1141, 1124, 1128, 1118, 1100, 1083, 1072, 1062, 1057, 1053, 1045, 1038, 1039, 976, 967, 948, 914, 908,]
lap_times_with_pit = lap_times_with_pit[~lap_times_with_pit['raceId'].isin(wet_race_id)]

# Creating a function as we will need to calculate outliers on more than one column
def Statistics(dataset):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5*IQR
    lower_limit = Q1 - 1.5*IQR

    return Q1, Q3, IQR, upper_limit, lower_limit

# Calculating outliers for lap_times
lap_time_Q1, lap_time_Q3, IQR, lap_time_upper_limit, lap_time_lower_limit = Statistics(lap_times_with_pit['lap_time'])
print('lap_time_statistics')
print('Q1: ', lap_time_Q1)
print('Q3: ', lap_time_Q3)
print('IQR: ', IQR)
print('Upper_limit: ', lap_time_upper_limit)
print('Lower limit: ', lap_time_lower_limit)

# Understanding how many lap time outliers are within our dataset
lap_time_outliers = lap_times_with_pit[(lap_times_with_pit['lap_time'] < lap_time_lower_limit) | (lap_times_with_pit['lap_time'] > lap_time_upper_limit)]
print(lap_time_outliers.count())

# Removing outliers from main dataset and comparing IQR statistics after
lap_times_with_pit = lap_times_with_pit[(lap_times_with_pit['lap_time'] >lap_time_lower_limit) & (lap_times_with_pit['lap_time'] <lap_time_upper_limit)]
print(lap_times_with_pit['lap_time'].describe())

# Creating a stint and tyre age column for the model to understand tyre degredation to lap time and new stint meaning new tyres
lap_times_with_pit = lap_times_with_pit.sort_values(['raceId', 'driverId', 'lap']).copy()
lap_times_with_pit['lap_after_pit'] = (lap_times_with_pit.groupby(['raceId', 'driverId'])['pit_stop'].shift(1).fillna(0))
lap_times_with_pit['stint'] = lap_times_with_pit.groupby(['raceId', 'driverId'])['lap_after_pit'].cumsum()
lap_times_with_pit['tyre_age'] = lap_times_with_pit.groupby(['raceId', 'driverId', 'stint']).cumcount() + 1
print(lap_times_with_pit.head(40))

# Creating new dataframe to remove outliers from pit stop duration, these includes pit stops where front wing changes or other quick fixes would have been done within the pit stop
pit_laps = lap_times_with_pit['pit_stop_duration'][lap_times_with_pit['pit_stop'] == 1]

# Understanding current IQR stats and then using the earlier created function to calculate stats, then removing them
print(pit_laps.describe())
pit_laps_q1, pit_laps_q3, IQR_pit, pit_laps_upper, pit_laps_lower = Statistics(pit_laps)

# Understanding how many outliers there are out of all laps that were pit laps
pit_outliers = pit_laps[(pit_laps < pit_laps_lower) | (pit_laps > pit_laps_upper)]
print(pit_outliers.count())
print(pit_laps.count())

# Filtering main dataset to remove pit stop durations that are outliers
lap_times_with_pit = lap_times_with_pit[(lap_times_with_pit['pit_stop'] == 0) | ((lap_times_with_pit['pit_stop_duration'] >= pit_laps_lower) & (lap_times_with_pit['pit_stop_duration'] <= pit_laps_upper))]
pit_laps_new = lap_times_with_pit['pit_stop_duration'][lap_times_with_pit['pit_stop'] == 1]

# Checking to make sure new IQR stats are appropriate for our model
print(pit_laps_new.describe())

# Checking to see if duplicate or missing values exist within our main dataset
print(lap_times_with_pit.duplicated().sum())
print(lap_times_with_pit.isnull().sum())

# Creating columns for previous lap time by shifting lap time down by 1 column, calculating lap delta by taking the difference of lap_time and the lap_time from the previous row
lap_times_with_pit['prev_lap_time'] = (lap_times_with_pit.groupby(['raceId', 'driverId'])['lap_time'].shift(1))
lap_times_with_pit['lap_time_delta'] = (lap_times_with_pit.groupby(['raceId', 'driverId'])['lap_time'].diff())
lap_times_with_pit['prev_lap_time'] = lap_times_with_pit['prev_lap_time'].fillna(0)
lap_times_with_pit['lap_time_delta'] = lap_times_with_pit['lap_time_delta'].fillna(0)
print(lap_times_with_pit.head())

# Creating a new dataframe to visualise and ensure features we are leaning towards will work in our model
potential_features = lap_times_with_pit.copy()
potential_features = potential_features.sample(1000)
potential_features = potential_features.drop('name', axis = 1) 
correlation_matrix = potential_features.corr()

# This next section creates heatmaps using the correlation matrix and creates a variety of scatter plots to understand linearity and feature relationships

plt.figure()
sns.heatmap(correlation_matrix)
plt.show()
plt.clf()
# Heatmap does indicate many non linear relationships between features, pit_stop and pit_stop_duration are showing a high correlation which makes sense, our dataset is designed to have 0 in the pit_stop column when a drive does not pit and therefore the pit_stop_duration is also 0. This would be a factor for the high correlation between the 2 features.
# All other features are showing non linear relationship behaviours

plt.figure()
plt.scatter(potential_features['tyre_age'], potential_features['lap_time'])
plt.title('tyre age vs lap time')
plt. xlabel('tyre age')
plt.ylabel('lap time')
plt.show()
plt.clf()
# There is no clear linear relationship, but there is a slight trend where lap times do decrease when tyre_age gets higher, there are still the odd laps at higher tyre age that are quicker, these can be a result of drs, slipstream, lower fuel loads


plt.figure()
plt.scatter(potential_features['lap'], potential_features['lap_time'])
plt.title('lap number vs lap time')
plt. xlabel('lap number')
plt.ylabel('lap time')
plt.show()
plt.clf()
# There is no clear linear relationship between lap and lap time, this does support assumptions as different races account for different pit windows and various race strategies, while earlier pitting cars would be newer tires putting in faster laps, cars running a longer stint prioritising track position would be on older tires resulting in slower lap times. Factors like fuel loads impact lap time as well, cars would be heavily fueled before the start of the race, with fuel loads decreasing as the number of laps increase. This could result in faster lap times towards the end of a race.

plt.figure()
sns.boxplot(x = potential_features['circuitId'], y = potential_features['lap_time'])
plt.title('circuit vs lap time')
plt. xlabel('circuitId')
plt.ylabel('lap time')
plt.show()
plt.clf()
# There is no clear linear relationship between circuitId and lap time. This is expected as circuitId represents different tracks rather than a continuous variable. Each circuit has different characteristics such as length, layout, and average speed which impact lap times. Faster circuits such as Monza generally produce shorter lap times, while tighter circuits with more corners tend to result in slower lap times. Different race strategies, fuel loads, and tyre conditions across drivers also contribute to the variation in lap times observed across circuits.

# From exploring correlation and visualising relationships between features, there is no clear linear relationship between features in the dataset. For this reason, we will be using a tree based model. 

lap_times_with_pit['race_progress'] = (lap_times_with_pit.groupby('raceId')['lap'].transform(lambda x: x / x.max()))

lap_times_with_pit['prev_lap_delta'] = lap_times_with_pit.groupby(['raceId','driverId'])['prev_lap_time'].diff().fillna(0)

lap_times_with_pit['fuel_load'] = (lap_times_with_pit.groupby('raceId')['lap'].transform(lambda x: 1 - (x / x.max())))

lap_times_with_pit['tyre_usage'] = (lap_times_with_pit.groupby(['raceId', 'driverId', 'stint'])['tyre_age'].transform(lambda x: x / x.max()))

lap_times_with_pit['tyre_deg'] = lap_times_with_pit['tyre_age'] ** 2

print(lap_times_with_pit.head(20))

# Creating final dataset that will be used in the model and dropping features not required
final_dataset = lap_times_with_pit.copy()
final_dataset = final_dataset.drop(['raceId', 'position', 'year', 'round', 'name', 'stint', 'pit_stop', 'lap_after_pit', 'driverId', 'tyre_usage'], axis = 1)
#final_dataset['stint'] = final_dataset['stint'].astype(int)
final_dataset['tyre_age'] = final_dataset['tyre_age'].astype(int)
print(final_dataset.columns)


# Upon designing the function to determine how the race strategy simulation is going to work, I noticed some data leakage issues with lap time delta. Lap time delta in our training dataset uses the current lap time from the current row and substracts the previous lap time. If our model is trained on this dataset, that would mean we require the lap time delta of the lap being predicted, before we even know the lap time. Predicting lap times with delta in the model would require us to input future information into the model as a feature. For this reason, I have dropped the lap time delta column from the dataframe. I am still keeping previous lap_time in the dataset, as this logic will allow us to later create lap time delta using our predictions.
# I decided to replace lap time delta with previous lap time delta, this takes the previous lap and calculates the delta using the lap previous to that. This allows us to avoid any data leakage.
final_dataset = final_dataset.drop(['lap_time_delta'], axis = 1)
print(final_dataset.head(40))
lap_time_X = final_dataset.drop(['lap_time', 'pit_stop_duration'], axis = 1)  
print(lap_time_X.columns)
lap_time_Y = final_dataset['lap_time']
print(lap_time_Y.head())
lap_time_X_train, lap_time_X_test, lap_time_Y_train, lap_time_Y_test = train_test_split(lap_time_X, lap_time_Y, test_size=0.2, random_state=42)

print(final_dataset.count())
print(final_dataset['lap_time'].mean())
print(final_dataset['lap_time'].median())
print(final_dataset['lap_time'].std())
# The difference between our mean and median is 0.8 seconds suggesting a slight right skew, this would indicate certain lap times at the top end slightly inflating the mean. However, with 223,855 rows it does point towards a is reasonably balanced without extreme skewness.
# The standard deviation is quite high at 13.63 seconds which indicates high variability in lap times, the challenge we are facing is that there are no features provided for the model to understand the reason behind that variability, features such as fuel load, tyre deg and race progress were approximated, but not accurate enough. Features that are missing that would help are fuel load, tyre deg, starting tyre compound, tyre compound after pit, and track temperature. 
# Tried different combinations of features to train the model on. In every instance, prev_lap_time held a feature importance of over 0.65. The model is relying heavily on prev_lap_time and we do not have the features or data to include to balance feature importance.
# Training the model with delta brought the mae and rmse down significantly. However, upon creating the race strategy simulation function it was clear that inputting lap delta before the lap had been predicted created data leakage.

lap_model = RandomForestRegressor(n_estimators = 300, max_depth = 20, min_samples_split = 5, min_samples_leaf = 2, max_features = 2, random_state = 42, n_jobs = -1)
lap_model.fit(lap_time_X_train, lap_time_Y_train)
lap_predictions = lap_model.predict(lap_time_X_test)

lap_mae = mean_absolute_error(lap_time_Y_test, lap_predictions)
lap_rmse = np.sqrt(mean_squared_error(lap_time_Y_test, lap_predictions))   

# Calculating average pit stop per circuitId to add onto total race time, adding this as a model input would require future information that is not available at prediction time
average_pit_duration = lap_times_with_pit[lap_times_with_pit['pit_stop'] == 1]
average_pit_duration = average_pit_duration.groupby(['circuitId'])['pit_stop_duration'].mean()
print(average_pit_duration.head(10))

# Lap model evaluation
print(lap_mae)
print(lap_rmse)

importance_df = pd.DataFrame({
    'feature': lap_time_X.columns,
    'importance': lap_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(importance_df)


# Created the race strategy function, inputting total desired laps, pit laps, driver id and circuit id will allow the function to iterate through each lap taking the lap time, previous lap time, and pit stop duration on pit laps and returning each one as a list
def race_strategy(total_laps, pit_lap, cID):
    laps = total_laps
    if type(pit_lap) == int:
        pit_lap = [pit_lap]
    pit_lap = sorted(pit_lap)
    circuitID = cID
    previous_lap_time = 0
    current_tyre_age = 1
    lap_after_pit_lap = [p + 1 for p in pit_lap]
    i = 1
    total_race_time = 0
    lap_times = []
    previous_lap_times = []
    previous_lap_delta = 0
    tyre_deg = 1 ** 2
    race_progress = 1 / total_laps
    fuel_load = 1 - (1 / total_laps)


    while i <= laps:
        features = pd.DataFrame([{'lap': i, 'circuitId': circuitID, 'tyre_age': current_tyre_age, 'prev_lap_time': previous_lap_time, 'race_progress': race_progress, 'prev_lap_delta': previous_lap_delta, 'fuel_load': fuel_load, 'tyre_deg': tyre_deg}])
        prediction = lap_model.predict(features)

        total_race_time += prediction[0]
        lap_times.append(prediction[0])
        if i == 1:
            previous_lap_times.append(0)
        else:
            previous_lap_times.append(lap_times[-2])
        
        i += 1

        previous_lap_time = prediction[0]

        race_progress = i / total_laps
        

        if i in lap_after_pit_lap:
            current_tyre_age = 1
        else:
            current_tyre_age += 1

        tyre_deg = current_tyre_age ** 2
        fuel_load = 1 - (i / total_laps)
        if len(lap_times) >= 2:
            previous_lap_delta = lap_times[-1] - lap_times[-2]
        else:
            previous_lap_delta = 0
    
    pit_stop_time = average_pit_duration.get(circuitID, 0) * len(pit_lap)
    total_race_time += pit_stop_time

    return total_race_time, lap_times, previous_lap_times


# This function will take the inputs from the race strategy function plus the lap times, previous lap times, and pit stop duration to create a dataframe for easier visualisation and model analysis
def predicted_strategy_df(laps, pit_lap, cID, lap_times, previous_lap_times):
    
    laps = laps
    circuitId = cID
    if type(pit_lap) == int:
        pit_lap = [pit_lap]
    lap_times = lap_times
    previous_lap_times = previous_lap_times

    predicted_data = pd.DataFrame()
    predicted_data['lap'] = range(1, laps + 1)
    predicted_data['circuitId'] = cID
    predicted_data['pit_stop'] = 0
    predicted_data.loc[predicted_data['lap'].isin(pit_lap), 'pit_stop'] = 1
    predicted_data['pit_stop_duration'] = 0.0
    predicted_data.loc[predicted_data['pit_stop'] == 1, 'pit_stop_duration'] = average_pit_duration.get(circuitId, 0)
    predicted_data['lap_after_pit'] = predicted_data['pit_stop'].shift(1).fillna(0)
    predicted_data['stint'] = predicted_data['lap_after_pit'].cumsum()
    predicted_data['tyre_age'] = predicted_data.groupby('stint').cumcount() + 1
    predicted_data['lap_time'] = lap_times
    predicted_data['total_race_time'] = (predicted_data['lap_time'] + predicted_data['pit_stop_duration']).cumsum()
    predicted_data['prev_lap_time'] = previous_lap_times
    predicted_data['race_progress'] = predicted_data['lap'] / laps
    predicted_data['prev_lap_delta'] = predicted_data['prev_lap_time'].diff().fillna(0)
    predicted_data['fuel_load'] = 1 - predicted_data['race_progress']
    predicted_data['tyre_deg'] = predicted_data['tyre_age'] ** 2
    predicted_data = predicted_data.drop('lap_after_pit', axis=1)

    return predicted_data



# Comparing different pit strategies from 1 pit stop, 2 pit stops, and 3 pit stops
total_race_time, lap_times, previous_lap_times = race_strategy(57, [30], 1)
predicted_data = predicted_strategy_df(57, [30], 1, lap_times, previous_lap_times)
print(predicted_data.head())
predicted_data.to_csv('lap_time_predictions_1_stop.csv', index = False)

total_race_time, lap_times, previous_lap_times = race_strategy(57, [16, 40], 1)
predicted_data = predicted_strategy_df(57, [16, 40], 1, lap_times, previous_lap_times)
print(predicted_data.head())
predicted_data.to_csv('lap_time_predictions_2_stop.csv', index = False)

total_race_time, lap_times, previous_lap_times = race_strategy(57, [16, 30, 43], 1)
predicted_data = predicted_strategy_df(57, [16, 30, 43], 1, lap_times, previous_lap_times)
print(predicted_data.head())
predicted_data.to_csv('lap_time_predictions_3_stop.csv', index = False)