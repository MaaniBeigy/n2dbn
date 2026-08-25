"""SEMA module for LifeSnaps Dataset"""

# ---------------------------- import libraries and modules ---------------------------
from typing import Optional

import pandas as pd
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_sema_notification_received(
    survey_name: str,
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "sema",
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches data of notification received from the sema collection in the LifeSnaps Dataset.

    Args:
        survey_name (str): The name of the survey to query.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "sema".
        timestamp_format (str, optional): The datetime format for fetching event timestamps in MongoDB. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame containing the fetched data in long format.

    """
    # ------------------------------- Connect to MongoDB ------------------------------
    client = MongoClient(mongo_host)
    db = client[database_name]
    collection = db[collection_name]

    # ------------------- Query the collection for a specific event -------------------
    # --------------------------------- Load raw data ---------------------------------
    items = list(
        collection.find(
            {
                "$and": [
                    {"data.SURVEY_NAME": survey_name},
                    {"data.SCHEDULED_TS": {"$ne": None}},
                ],
            },
            {
                "_id": 1,
                "user_id": 1,
                "data.SCHEDULED_TS": 1,
            },
        )
    )

    # --------------------------------- Transform data --------------------------------

    data = []
    for item in items:
        row = {
            "id": str(item["_id"]),
            "user_id": str(item["user_id"]),
            "event_time": item["data"].get("SCHEDULED_TS", None),
            "event_name": "notification_received",
            "event_property": "notification",
            "event_value": survey_name,
        }
        data.append(row)

    # -------------------------------- Create DataFrame -------------------------------
    df_long = pd.DataFrame(data)

    # ------------------------- Convert event_time to datetime ------------------------
    if timestamp_format:
        df_long["event_time"] = pd.to_datetime(
            df_long["event_time"], format=timestamp_format, errors="coerce"
        )
    else:
        df_long["event_time"] = pd.to_datetime(df_long["event_time"], errors="coerce")

    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]

    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )

    return df_long


def fetch_sema_notification_read(
    survey_name: str,
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "sema",
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches data of notification read from the sema collection in the LifeSnaps Dataset.

    Args:
        survey_name (str): The name of the survey to query.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "sema".
        timestamp_format (str, optional): The datetime format for fetching event timestamps in MongoDB. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame containing the fetched data in long format.

    """
    # ------------------------------- Connect to MongoDB ------------------------------
    client = MongoClient(mongo_host)
    db = client[database_name]
    collection = db[collection_name]

    # ------------------- Query the collection for a specific event -------------------
    # --------------------------------- Load raw data ---------------------------------
    items = list(
        collection.find(
            {
                "$and": [
                    {"data.SURVEY_NAME": survey_name},
                    {"data.STARTED_TS": {"$ne": None}},
                ],
            },
            {
                "_id": 1,
                "user_id": 1,
                "data.STARTED_TS": 1,
            },
        )
    )

    # --------------------------------- Transform data --------------------------------

    data = []
    for item in items:
        row = {
            "id": str(item["_id"]),
            "user_id": str(item["user_id"]),
            "event_time": item["data"].get("STARTED_TS", None),
            "event_name": "notification_read",
            "event_property": "notification",
            "event_value": survey_name,
        }
        data.append(row)

    # -------------------------------- Create DataFrame -------------------------------
    df_long = pd.DataFrame(data)

    # ------------------------- Convert event_time to datetime ------------------------
    if timestamp_format:
        df_long["event_time"] = pd.to_datetime(
            df_long["event_time"], format=timestamp_format, errors="coerce"
        )
    else:
        df_long["event_time"] = pd.to_datetime(df_long["event_time"], errors="coerce")

    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]

    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )

    return df_long
