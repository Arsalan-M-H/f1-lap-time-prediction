# F1 Race Strategy & Lap Time Prediction

## Project Overview

This project analyses Formula 1 race data to model and simulate **lap times and race strategies** using machine learning. The goal is to understand how factors such as tyre degradation, fuel load, and race progression influence lap performance, and to use these insights to simulate different pit stop strategies.

The project combines data engineering, exploratory analysis, and a tree-based machine learning model to predict lap times and evaluate total race duration under different strategy scenarios.

The dataset consists of lap-level race data, pit stop information, and race metadata. By merging and engineering these datasets, the project builds a simulation framework that can estimate race outcomes based on different pit stop decisions.

## Dataset

The dataset is sourced from publicly available Formula 1 data and contains detailed lap-by-lap race information.

### Key tables used in the analysis

| Table | Description |
|------|-------------|
| lap_times | Contains lap-by-lap timing data for each driver |
| pit_stops | Contains pit stop events and durations |
| races | Contains race metadata including circuit and year |

## Project Objectives

The analysis and modelling focus on answering key performance and strategy questions:

- How do lap times change throughout a race?
- How does tyre degradation impact lap performance?
- What is the effect of fuel load on lap time?
- How do pit stops influence total race time?
- What is the average pit stop duration per circuit?
- Which pit strategies (1-stop, 2-stop, 3-stop) result in faster total race times?

## Data Processing & Feature Engineering

The dataset required significant preprocessing to ensure reliable modelling:

- Merged lap times with pit stop data to identify pit laps
- Converted pit stop indicators into binary features
- Converted time values from milliseconds to seconds
- Filtered races from 2014 onwards to reflect modern F1 regulations
- Removed wet races to reduce variability caused by extreme conditions
- Removed outliers using the IQR method for:
  - Lap times
  - Pit stop durations
- Engineered sequential and race-based features:
  - `tyre_age` (laps since last pit stop)
  - `stint` (continuous run between pit stops)
  - `race_progress` (lap / total laps)
  - `fuel_load` (decreasing over race)
  - `prev_lap_time`
  - `prev_lap_delta`
  - `tyre_deg` (non-linear tyre degradation)

## Machine Learning Model

A **Random Forest Regressor** was used due to the non-linear relationships observed between features.

### Model details:
- Model: RandomForestRegressor
- Handles non-linear relationships effectively
- Robust to noise and feature interactions
- Feature importance used to interpret model behaviour

### Evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Key insight:
The model relies heavily on **previous lap time**, indicating strong sequential dependency in lap performance.

## Data Leakage Consideration

A key challenge identified was **data leakage**:

- Using `lap_time_delta` would require knowledge of the current lap time before prediction
- This was removed and replaced with `prev_lap_delta` to ensure only past information is used

Additionally, splitting the dataset randomly can introduce sequence overlap, as laps from the same race may appear in both training and test sets. A grouped split by race was tested and resulted in lower performance, indicating that race-specific patterns strongly influence predictions.

## Race Strategy Simulation

A custom simulation function was developed to model race strategies.

### Inputs:
- Total laps
- Pit stop laps
- Circuit ID

### Outputs:
- Predicted lap times
- Total race time
- Lap-by-lap breakdown of race performance

The simulation iteratively predicts each lap using:
- Previous lap time
- Tyre age
- Fuel load
- Race progression

Pit stop time is added using the **average pit stop duration per circuit**.

## Strategy Comparison

The model was used to compare different strategies:

- 1-stop strategy
- 2-stop strategy
- 3-stop strategy

Each strategy generates a full race simulation and exports results for further analysis.

## Output

The project generates:

- Predicted lap-by-lap datasets
- Total race time comparisons
- CSV files for each strategy scenario:
  - `lap_time_predictions_1_stop.csv`
  - `lap_time_predictions_2_stop.csv`
  - `lap_time_predictions_3_stop.csv`

## Key Concepts Demonstrated

This project showcases practical data science and machine learning skills:

- Data cleaning and preprocessing
- Handling missing values and outliers
- Feature engineering for sequential data
- Avoiding data leakage in modelling
- Tree-based machine learning (Random Forest)
- Model evaluation and interpretation
- Simulation modelling
- Real-world problem solving using domain assumptions

## Limitations

- No access to real tyre compound data
- Fuel load and tyre degradation are approximated
- Weather and track conditions are not included
- Model heavily depends on previous lap time

## Future Improvements

- Incorporate tyre compound data (soft, medium, hard)
- Include weather and track temperature
- Use time-series or sequential models
- Improve feature balance to reduce reliance on previous lap time
- Integrate real-time data for dynamic strategy simulation
