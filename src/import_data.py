import requests
import pandas as pd
from pathlib import Path


def importing_data():
    # Getting laps data
    laps_url = "https://api.openf1.org/v1/laps"
    try:
        response = requests.get(laps_url, timeout=30)
        response.raise_for_status()
        laps_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching laps data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return

    laps_df = pd.DataFrame(laps_data)
    # print(laps_df.head(20))
    # print(laps_df.columns)
    # print(laps_df.count())

# Upload
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "original"
    LAPS_PATH = DATA_DIR / "laps_df.csv"
    laps_df.to_csv(LAPS_PATH, index=False)


    # Getting stints data
    stints_url = "https://api.openf1.org/v1/stints"
    try:
        response = requests.get(stints_url, timeout=30)
        response.raise_for_status()
        stints_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching stints data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return

    stints_df = pd.DataFrame(stints_data)
    # print(stints_df.head(20))
    # print(stints_df.columns)
    # print(stints_df.count())

# Upload
    STINTS_PATH = DATA_DIR / "stints_df.csv"
    stints_df.to_csv(STINTS_PATH, index=False)


    # Getting meetings data
    meetings_url = "https://api.openf1.org/v1/meetings"
    try:
        response = requests.get(meetings_url, timeout=30)
        response.raise_for_status()
        meetings_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching meetings data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return

    meetings_df = pd.DataFrame(meetings_data)
    # print(meetings_df.head(20))
    # print(meetings_df.columns)
    # print(meetings_df.count())

# Upload
    MEETINGS_PATH = DATA_DIR / "meetings_df.csv"
    meetings_df.to_csv(MEETINGS_PATH, index=False)

    # Getting pit stop data
    pit_url = "https://api.openf1.org/v1/pit"
    try:
        response = requests.get(pit_url, timeout=30)
        response.raise_for_status()
        pit_stop_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching pit stop data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return
    
    pit_stop_df = pd.DataFrame(pit_stop_data)
    # print(pit_stop_df.head(20))
    # print(pit_stop_df.columns)
    # print(pit_stop_df.count())

# Upload
    PIT_STOP_PATH = DATA_DIR / "pit_stops_df.csv"
    pit_stop_df.to_csv(PIT_STOP_PATH, index=False)

    # Getting starting grid data
    starting_grid_url = "https://api.openf1.org/v1/starting_grid"
    try:
        response = requests.get(starting_grid_url, timeout=30)
        response.raise_for_status()
        starting_grid_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching starting grid data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return
    
    starting_grid_df = pd.DataFrame(starting_grid_data)
    # print(starting_grid_df.head(20))
    # print(starting_grid_df.columns)
    # print(starting_grid_df.count())

# Upload
    STARTING_GRID_PATH = DATA_DIR / "starting_grid_df.csv"
    starting_grid_df.to_csv(STARTING_GRID_PATH, index=False)

    # Getting weather data
    weather_url = "https://api.openf1.org/v1/weather"
    try:
        response = requests.get(weather_url, timeout=30)
        response.raise_for_status()
        weather_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return
    
    weather_df = pd.DataFrame(weather_data)
    # print(weather_df.head(20))
    # print(weather_df.columns)
    # print(weather_df.count())

# Upload
    WEATHER_PATH = DATA_DIR / "weather_df.csv"
    weather_df.to_csv(WEATHER_PATH, index=False)

    # Getting session data
    session_url = "https://api.openf1.org/v1/sessions"
    try:
        response = requests.get(session_url, timeout=30)
        response.raise_for_status()
        session_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching session data: {e}")
        return
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return
    
    session_df = pd.DataFrame(session_data)
    # print(session_df.head(20))
    # print(session_df.columns)
    # print(session_df.count())

# Upload
    SESSION_PATH = DATA_DIR / "session_df.csv"
    session_df.to_csv(SESSION_PATH, index=False)

    print("All import functions completed successfully.")
