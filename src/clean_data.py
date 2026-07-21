import pandas as pd
from pathlib import Path
import numpy as np

IMPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "original"
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "cleaned"
IMPORT_LAPS_PATH = IMPORT_DIR / "laps_df.csv"
IMPORT_STINTS_PATH = IMPORT_DIR / "stints_df.csv"
IMPORT_MEETINGS_PATH = IMPORT_DIR / "meetings_df.csv"
IMPORT_PIT_STOP_PATH = IMPORT_DIR / "pit_stops_df.csv"
IMPORT_STARTING_GRID_PATH = IMPORT_DIR / "starting_grid_df.csv"
IMPORT_WEATHER_PATH = IMPORT_DIR / "weather_df.csv"
IMPORT_SESSION_PATH = IMPORT_DIR / "session_df.csv"


def bounds(x):
    Q1 = np.quantile(x,0.25)
    Q3 = np.quantile(x,0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    return lower_bound, upper_bound

def clean_laps():
    laps_df = pd.read_csv(IMPORT_LAPS_PATH)
    # print("Laps missing values")
    # print(laps_df.isna().sum())

    laps_df = laps_df.dropna(subset=["duration_sector_1", "lap_duration", "duration_sector_2", "duration_sector_3"])
    laps_df = laps_df[["meeting_key", "session_key", "driver_number", "lap_number", "date_start", "duration_sector_1", "duration_sector_2", "duration_sector_3", "is_pit_out_lap", "lap_duration"]]

    # print("Laps missing values")
    # print(laps_df.isna().sum())
    # print(laps_df.count())

# Outliers
    lower_bound, upper_bound = bounds(laps_df['lap_duration'])
    # print(lower_bound)
    # print(upper_bound)
    # print(laps_df["lap_duration"].max())
    # print(laps_df["lap_duration"].min())
    lap_outliers = laps_df[(laps_df["lap_duration"] < lower_bound) | (laps_df["lap_duration"] > upper_bound)]
    # print(lap_outliers["lap_duration"].describe())
    # print(laps_df["lap_duration"].describe())
    laps_df = laps_df[(laps_df["lap_duration"] >= lower_bound) & (laps_df["lap_duration"] <= upper_bound)]
    # print(laps_df["lap_duration"].describe())

# Upload
    UPLOAD_LAPS_DIR = UPLOAD_DIR / "laps_df.csv"
    laps_df.to_csv(UPLOAD_LAPS_DIR, index=False)

def clean_stints():
    stints_df = pd.read_csv(IMPORT_STINTS_PATH)
    # print("Stints missing values")
    # print(stints_df.isna().sum())

    stints_df = stints_df.dropna(subset=["lap_start", "lap_end", "compound", "tyre_age_at_start"])
    stints_df = stints_df[stints_df["compound"] != "TEST_UNKNOWN"]
    stints_df = stints_df[stints_df["compound"] != "UNKNOWN"]
    #print(stints_df["compound"].unique())
    # print("Stints missing values")
    # print(stints_df.isna().sum())
    # print(stints_df.count())

# Upload
    UPLOAD_STINTS_DIR = UPLOAD_DIR / "stints_df.csv"
    stints_df.to_csv(UPLOAD_STINTS_DIR, index=False)

def clean_meetings():
    meetings_df = pd.read_csv(IMPORT_MEETINGS_PATH)
    # print("Meetings missing values")
    # print(meetings_df.isna().sum())

# Upload
    UPLOAD_MEETINGS_DIR = UPLOAD_DIR / "meetings_df.csv"
    meetings_df.to_csv(UPLOAD_MEETINGS_DIR, index=False)

def clean_starting_grid():
    starting_grid_df = pd.read_csv(IMPORT_STARTING_GRID_PATH)
    # print(starting_grid_df.isna().sum())
    starting_grid_df = starting_grid_df[["position", "driver_number", "meeting_key", "session_key"]]
    # print(starting_grid_df.isna().sum())
    # print(starting_grid_df["position"].describe())

# Upload
    UPLOAD_STARTING_GRID_DIR = UPLOAD_DIR / "starting_grid_df.csv"
    starting_grid_df.to_csv(UPLOAD_STARTING_GRID_DIR, index=False)

def clean_weather():
    weather_df = pd.read_csv(IMPORT_WEATHER_PATH)
    # print(weather_df.isna().sum())
    # print(weather_df["track_temperature"].describe())
    # print(weather_df["rainfall"].describe())
    # print(weather_df["humidity"].describe())

# Upload
    UPLOAD_WEATHER_DIR = UPLOAD_DIR / "weather_df.csv"
    weather_df.to_csv(UPLOAD_WEATHER_DIR, index=False)

def clean_sessions():
    session_df = pd.read_csv(IMPORT_SESSION_PATH)
    # print(session_df.isna().sum())    

# Upload
    UPLOAD_SESSIONS_DIR = UPLOAD_DIR / "sessions_df.csv"
    session_df.to_csv(UPLOAD_SESSIONS_DIR, index=False)

def clean_data():
    try:
        clean_laps()
        clean_stints()
        clean_meetings()
        clean_starting_grid()
        clean_weather()
        clean_sessions()
        print("All cleaning functions completed successfully.")
    except Exception as e:
        print(f"Cleaning failed: {e}")

