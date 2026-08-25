"""Fitbit sleep module for LifeSnaps Dataset"""

# ---------------------------- import libraries and modules ---------------------------
from typing import List, Optional

import pandas as pd
from lifesnaps.constants import SLEEP_HARD_CODED_PROPERTIES
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_fitbit_sleep_data(
    event_name: str = "sleep",
    event_timestamp: str = "endTime",
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "fitbit",
    event_properties: Optional[List[str]] = None,
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches sleep data from the Fitbit collection in the LifeSnaps Dataset.

    Args:
        event_name (str): The name of the event type to query.
        event_timestamp (str): The timestamp field in the data.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "fitbit".
        event_properties (List[str], optional): The properties of the event to fetch. Defaults to None.
        timestamp_format (str, optional): The datetime format for parsing event timestamps. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched sleep event data in long format.

    Raises:
        Exception: If event_properties is not provided.
    """
    # ------------------------------- Connect to MongoDB ------------------------------
    client = MongoClient(mongo_host)
    db = client[database_name]
    collection = db[collection_name]
    # ------------------- Query the collection for a specific event -------------------
    query = {"type": event_name}
    items = list(collection.find(query))
    
    # --------------------------------- Transform data --------------------------------
    data = []
    for item in items:
        sleep_levels = item["data"].get("levels", {}).get("summary", [])
        row = {
            "id": str(item["_id"]),
            "user_id": str(item["id"]),
            "event_time": item["data"].get(event_timestamp, None),
            "event_name": str(item["type"]),
            # -------------------------- sleep levels - deep --------------------------
            "deep_sleep_count": sleep_levels.get("deep", {}).get("count", None),
            "deep_sleep_minutes": sleep_levels.get("deep", {}).get("minutes", None),
            "deep_sleep_thirtyDayAvgMinutes": sleep_levels.get("deep", {}).get(
                "thirtyDayAvgMinutes", None
            ),
            # -------------------------- sleep levels - wake --------------------------
            "wake_sleep_count": sleep_levels.get("wake", {}).get("count", None),
            "wake_sleep_minutes": sleep_levels.get("wake", {}).get("minutes", None),
            "wake_sleep_thirtyDayAvgMinutes": sleep_levels.get("wake", {}).get(
                "thirtyDayAvgMinutes", None
            ),
            # -------------------------- sleep levels - deep --------------------------
            "light_sleep_count": sleep_levels.get("light", {}).get("count", None),
            "light_sleep_minutes": sleep_levels.get("light", {}).get("minutes", None),
            "light_sleep_thirtyDayAvgMinutes": sleep_levels.get("light", {}).get(
                "thirtyDayAvgMinutes", None
            ),
            # -------------------------- sleep levels - rem --------------------------
            "rem_sleep_count": sleep_levels.get("rem", {}).get("count", None),
            "rem_sleep_minutes": sleep_levels.get("rem", {}).get("minutes", None),
            "rem_sleep_thirtyDayAvgMinutes": sleep_levels.get("rem", {}).get(
                "thirtyDayAvgMinutes", None
            ),
        }

        if event_properties:
            # Extract nested fields from the data object add the event properties
            for prop in event_properties:
                row[prop] = item["data"].get(prop, None)
        else:
            raise Exception("You should enter event_properties!")
        data.append(row)

    # -------------------------------- Create DataFrame -------------------------------
    df_wide = pd.DataFrame(data)

    # ------------------------- Convert event_time to datetime ------------------------
    if timestamp_format:
        df_wide["event_time"] = pd.to_datetime(
            df_wide["event_time"], format=timestamp_format, errors="coerce"
        )
    else:
        df_wide["event_time"] = pd.to_datetime(df_wide["event_time"], errors="coerce")

    # ----------------------- Melt the DataFrame to long format -----------------------
    df_long = df_wide.melt(
        id_vars=["id", "user_id", "event_time", "event_name"],
        value_vars=event_properties + SLEEP_HARD_CODED_PROPERTIES,
        var_name="event_property",
        value_name="event_value",
    )

    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]

    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )

    return df_long
