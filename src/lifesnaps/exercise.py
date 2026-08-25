"""Fitbit exercise module for LifeSnaps Dataset"""

# ---------------------------- import libraries and modules ---------------------------
from typing import List, Optional

import pandas as pd
from lifesnaps.constants import EXERCISE_HARD_CODED_PROPERTIES
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_fitbit_exercise_data(
    event_name: str = "exercise",
    event_timestamp: str = "lastModified",
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "fitbit",
    event_properties: Optional[List[str]] = None,
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches exercise data from the Fitbit collection in the LifeSnaps Dataset.

    Args:
        event_name (str): The name of the event type to query.
        event_timestamp (str): The timestamp field in the data.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "fitbit".
        event_properties (List[str], optional): The properties of the event to fetch. Defaults to None.
        timestamp_format (str, optional): The datetime format for parsing event timestamps. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched exercise event data in long format.

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
        activity_levels = item["data"].get("activityLevel", [])
        heart_rate_zones = item["data"].get("heartRateZones", [])
        active_zone_minutes = (
            item["data"].get("activeZoneMinutes", {}).get("minutesInHeartRateZones", [])
        )

        row = {
            "id": str(item["_id"]),
            "user_id": str(item["id"]),
            "event_time": item["data"].get(event_timestamp, None),
            "event_name": str(item["type"]),
            # ----------------------------- activityLevel -----------------------------
            "sedentary_activityLevel_minutes": next(
                (d["minutes"] for d in activity_levels if d["name"] == "sedentary"),
                None,
            ),
            "lightly_activityLevel_minutes": next(
                (d["minutes"] for d in activity_levels if d["name"] == "lightly"), None
            ),
            "fairly_activityLevel_minutes": next(
                (d["minutes"] for d in activity_levels if d["name"] == "fairly"), None
            ),
            "very_activityLevel_minutes": next(
                (d["minutes"] for d in activity_levels if d["name"] == "very"), None
            ),
            # ------------------------- manualValuesSpecified -------------------------
            "manual_calories": item["data"]
            .get("manualValuesSpecified", {})
            .get("calories", None),
            "manual_distance": item["data"]
            .get("manualValuesSpecified", {})
            .get("distance", None),
            "manual_steps": item["data"]
            .get("manualValuesSpecified", {})
            .get("steps", None),
            # --------------------- heartRateZones - Out of Range ---------------------
            "heartRateZones_OutOfRange_min": next(
                (d["min"] for d in heart_rate_zones if d["name"] == "Out of Range"),
                None,
            ),
            "heartRateZones_OutOfRange_max": next(
                (d["max"] for d in heart_rate_zones if d["name"] == "Out of Range"),
                None,
            ),
            "heartRateZones_OutOfRange_minutes": next(
                (d["minutes"] for d in heart_rate_zones if d["name"] == "Out of Range"),
                None,
            ),
            "heartRateZones_OutOfRange_caloriesOut": next(
                (
                    d["caloriesOut"]
                    for d in heart_rate_zones
                    if d["name"] == "Out of Range" and "caloriesOut" in d
                ),
                None,
            ),
            # ----------------------- heartRateZones - Fat Burn -----------------------
            "heartRateZones_FatBurn_min": next(
                (d["min"] for d in heart_rate_zones if d["name"] == "Fat Burn"), None
            ),
            "heartRateZones_FatBurn_max": next(
                (d["max"] for d in heart_rate_zones if d["name"] == "Fat Burn"), None
            ),
            "heartRateZones_FatBurn_minutes": next(
                (d["minutes"] for d in heart_rate_zones if d["name"] == "Fat Burn"),
                None,
            ),
            "heartRateZones_FatBurn_caloriesOut": next(
                (
                    d["caloriesOut"]
                    for d in heart_rate_zones
                    if d["name"] == "Fat Burn" and "caloriesOut" in d
                ),
                None,
            ),
            # ------------------------ heartRateZones - Cardio ------------------------
            "heartRateZones_Cardio_min": next(
                (d["min"] for d in heart_rate_zones if d["name"] == "Cardio"), None
            ),
            "heartRateZones_Cardio_max": next(
                (d["max"] for d in heart_rate_zones if d["name"] == "Cardio"), None
            ),
            "heartRateZones_Cardio_minutes": next(
                (d["minutes"] for d in heart_rate_zones if d["name"] == "Cardio"), None
            ),
            "heartRateZones_Cardio_caloriesOut": next(
                (
                    d["caloriesOut"]
                    for d in heart_rate_zones
                    if d["name"] == "Cardio" and "caloriesOut" in d
                ),
                None,
            ),
            # ------------------------- heartRateZones - Peak -------------------------
            "heartRateZones_Peak_min": next(
                (d["min"] for d in heart_rate_zones if d["name"] == "Peak"), None
            ),
            "heartRateZones_Peak_max": next(
                (d["max"] for d in heart_rate_zones if d["name"] == "Peak"), None
            ),
            "heartRateZones_Peak_minutes": next(
                (d["minutes"] for d in heart_rate_zones if d["name"] == "Peak"), None
            ),
            "heartRateZones_Peak_caloriesOut": next(
                (
                    d["caloriesOut"]
                    for d in heart_rate_zones
                    if d["name"] == "Peak" and "caloriesOut" in d
                ),
                None,
            ),
            # -------------------- activeZoneMinutes - totalMinutes -------------------
            "activeZoneMinutes_totalMinutes": item["data"]
            .get("activeZoneMinutes", {})
            .get("totalMinutes", None),
            # -------------------- activeZoneMinutes - Out of Range -------------------
            "activeZoneMinutes_OutOfRange_minutes": next(
                (
                    d["minutes"]
                    for d in active_zone_minutes
                    if d["zoneName"] == "Out of Range"
                ),
                None,
            ),
            # ----------------------- activeZoneMinutes - Cardio ----------------------
            "activeZoneMinutes_Cardio_minutes": next(
                (
                    d["minutes"]
                    for d in active_zone_minutes
                    if d["zoneName"] == "Cardio"
                ),
                None,
            ),
            # ---------------------- activeZoneMinutes - Fat Burn ---------------------
            "activeZoneMinutes_FatBurn_minutes": next(
                (
                    d["minutes"]
                    for d in active_zone_minutes
                    if d["zoneName"] == "Fat Burn"
                ),
                None,
            ),
            # ------------------------ activeZoneMinutes - Peak -----------------------
            "activeZoneMinutes_Peak_minutes": next(
                (d["minutes"] for d in active_zone_minutes if d["zoneName"] == "Peak"),
                None,
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
        value_vars=event_properties + EXERCISE_HARD_CODED_PROPERTIES,
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
