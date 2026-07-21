import pandas as pd
from pathlib import Path

IMPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "cleaned"
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "final"
IMPORT_LAPS_PATH = IMPORT_DIR / "laps_df.csv"
IMPORT_STINTS_PATH = IMPORT_DIR / "stints_df.csv"
IMPORT_MEETINGS_PATH = IMPORT_DIR / "meetings_df.csv"
IMPORT_PIT_STOP_PATH = IMPORT_DIR / "pit_stops_df.csv"
IMPORT_STARTING_GRID_PATH = IMPORT_DIR / "starting_grid_df.csv"
IMPORT_WEATHER_PATH = IMPORT_DIR / "weather_df.csv"

def merge_stints():
    laps_df = pd.read_csv(IMPORT_LAPS_PATH)
    stints_df = pd.read_csv(IMPORT_STINTS_PATH)

    for i, laps in laps_df.iterrows():
        matching_rows = stints_df[
            (laps["meeting_key"] == stints_df["meeting_key"]) &
            (laps["session_key"] == stints_df["session_key"]) &
            (laps["driver_number"] == stints_df["driver_number"]) &
            (laps["lap_number"] >= stints_df["lap_start"]) &
            (laps["lap_number"] <= stints_df["lap_end"])
        ]
        
        if len(matching_rows) > 0:
            stint = matching_rows.iloc[0]
            laps_df.loc[i, "compound"] = stint["compound"]
            laps_df.loc[i, "tyre_age"] = stint["tyre_age_at_start"] + (laps["lap_number"] - stint["lap_start"])
            laps_df.loc[i, "stint"] = stint["stint"]
        else:
            laps_df.loc[i, "compound"] = "UNKNOWN"
            laps_df.loc[i, "tyre_age"] = "UNKNOWN"
            laps_df.loc[i, "stint"] = "UNKNOWN"

# Upload
    UPLOAD_LAPS_PATH = UPLOAD_DIR / "laps_df.csv"
    laps_df.to_csv(UPLOAD_LAPS_PATH, index=False)

def feature_engineer():
    merge_stints()