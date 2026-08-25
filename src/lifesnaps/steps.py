"""SEMA step goals module for LifeSnaps Dataset"""

# ---------------------------- import libraries and modules ---------------------------
from typing import Optional

import pandas as pd
from lifesnaps.helper import try_convert_to_numeric
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_sema_step_goal_data(
    survey_name: str = "Step Goal Survey",
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "sema",
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches data of Step Goal Survey from the sema collection in the LifeSnaps Dataset.

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
                    {"data.COMPLETED_TS": {"$ne": None}},
                ],
            }
        )
    )

    # --------------------------------- Transform data --------------------------------

    data = []
    for item in items:
        row = {
            "id": str(item["_id"]),
            "user_id": str(item["user_id"]),
            "event_time": item["data"].get("COMPLETED_TS", None),
            "event_name": "Step goal survey completed",
            "total response time": item["data"].get("TOTAL_RT", {}),
            "steps": try_convert_to_numeric(item["data"].get("STEPS", {})),
            "steps response time": try_convert_to_numeric(
                item["data"].get("STEPS_RT", {})
            ),
        }
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
        value_vars=[
            "total response time",
            "steps",
            "steps response time",
        ],
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
