"""stai module for LifeSnaps Dataset"""

# State-Trait Anxiety Inventory (S-STAI)
# ---------------------------- import libraries and modules ---------------------------
from typing import Optional

import pandas as pd
from lifesnaps.constants import STAI_EVENT_PROPERTIES, STAI_INVERSE_QUESTION_CODES
from lifesnaps.helper import (
    convert_5_to_4_likert,
    inverse_scores,
    try_convert_to_numeric,
)
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_stai_event_data(
    event_name: str = "stai",
    event_timestamp: str = "submitdate",
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "surveys",
    timestamp_format: Optional[str] = "%d/%m/%Y %H:%M",
) -> pd.DataFrame:
    """
    Fetches data from the stai collection in the LifeSnaps Dataset.

    Args:
        event_name (str): The name of the event type to query.
        event_timestamp (str): The timestamp field in the data.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "surveys".
        timestamp_format (str, optional): The datetime format for fetching event timestamps in MongoDB. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame containing the fetched event data in long format.

    Raises:
        Exception: If event_properties is not provided.
    """
    # ------------------------------- Connect to MongoDB ------------------------------
    client = MongoClient(mongo_host)
    db = client[database_name]
    collection = db[collection_name]

    # ------------------- Query the collection for a specific event -------------------
    # --------------------------------- Load raw data ---------------------------------
    query = {"type": event_name}
    items = list(collection.find(query))

    # --------------------------------- Transform data --------------------------------

    data = []
    for item in items:
        row = {
            "id": str(item["_id"]),
            "user_id": str(item["user_id"]),
            "event_time": item["data"].get(event_timestamp, None),
            "event_name": str(item["type"]),
        }
        # Extract nested fields from the data object and add the event properties
        for prop in STAI_EVENT_PROPERTIES:
            value = item["data"].get(prop, None)

            # Convert 5-likert to 4-likert scale.
            if value is not None:
                value = convert_5_to_4_likert(value)
            # Apply inverse_scores if the value is true
            if STAI_INVERSE_QUESTION_CODES.get(prop, True):
                value = inverse_scores(value, min=1, max=4)
            row[prop] = value
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
        value_vars=STAI_EVENT_PROPERTIES,
        var_name="event_property",
        value_name="event_value",
    )

    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]
    # ------- Remove duplicates for each user_id, event_time, and event_property ------
    df_long = df_long.drop_duplicates(
        subset=["user_id", "event_time", "event_property"], keep='last'
    )
    # -------- Calculate the stai_stress_score for each user_id and event_time --------
    stai_score_df = df_long[
        df_long["event_property"].isin(STAI_EVENT_PROPERTIES)
    ].copy()

    # Group by user_id and event_time and calculate the sum of event_value for each group
    stai_score_df = (
        stai_score_df.groupby(["user_id", "event_time"])["event_value"]
        .sum()
        .reset_index()
    )

    # ------------------- Rename columns to match the desired format ------------------
    stai_score_df.rename(columns={"event_value": "event_value"}, inplace=True)
    stai_score_df["event_property"] = "stai_stress_score"
    stai_score_df["event_name"] = event_name
    # ----------------- Concatenate the stai_score_df back to df_long -----------------
    df_long = pd.concat([df_long, stai_score_df], ignore_index=True)
    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )

    return df_long
