"""ttm module for LifeSnaps Dataset"""

# Stages and Processes of Change Questionnaire
# ---------------------------- import libraries and modules ---------------------------
from typing import Optional

import pandas as pd
from lifesnaps.constants import TTM_EVENT_PROPERTIES, TTM_FACTORS
from lifesnaps.helper import try_convert_to_numeric, define_stage_of_change
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_ttmspbf_event_data(
    event_name: str = "ttmspbf",
    event_timestamp: str = "submitdate",
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "surveys",
    timestamp_format: Optional[str] = "%d/%m/%Y %H:%M",
) -> pd.DataFrame:
    """
    Fetches data from the ttm collection in the LifeSnaps Dataset.

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
        for prop in TTM_EVENT_PROPERTIES:
            value = item["data"].get(prop, None)

            # Convert the value to a numeric type if applicable
            if value is not None:
                value = try_convert_to_numeric(value)
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
        value_vars=TTM_EVENT_PROPERTIES,
        var_name="event_property",
        value_name="event_value",
    )

    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]
    # ------- Remove duplicates for each user_id, event_time, and event_property ------
    df_long = df_long.drop_duplicates(
        subset=["user_id", "event_time", "event_property"], keep="last"
    )
    # -------------------------- Calculate the factor scores --------------------------
    factor_scores = {}
    for factor_name, props in TTM_FACTORS.items():
        factor_scores[factor_name] = (
            df_long[df_long["event_property"].isin(props)]
            .groupby(["user_id", "event_time"])["event_value"]
            .mean()
            .reset_index()
            .rename(columns={"event_value": factor_name})
        )

    # --------------- Merge the factor scores back to a single DataFrame --------------
    df_factors = df_long[["user_id", "event_time"]].drop_duplicates().copy()
    for factor_name, score_df in factor_scores.items():
        df_factors = df_factors.merge(
            score_df, on=["user_id", "event_time"], how="left"
        )

    # ----------------------------- Calculate ttm_status -----------------------------
    # Create a new DataFrame with the 'ttm_stage' event_property
    ttm_stage_df = df_long[df_long["event_property"] == "stage"].copy()

    # Apply the define_stage_of_change function to the 'event_value' to create the 'ttm_stage' values
    ttm_stage_df["event_value"] = ttm_stage_df["event_value"].apply(
        define_stage_of_change
    )

    # Update the 'event_property' to 'ttm_stage' and 'event_value_type' to 'str'
    ttm_stage_df["event_property"] = "ttm_stage"
    ttm_stage_df["event_value_type"] = "str"

    # ------------ Convert df_factors to long format and append to df_long ------------
    factor_long = df_factors.melt(
        id_vars=["user_id", "event_time"],
        value_vars=list(TTM_FACTORS.keys()),
        var_name="event_property",
        value_name="event_value",
    )

    # ----------------------------- Concatenate all scores ----------------------------
    # Adding event_name to factor_long DataFrame
    factor_long["event_name"] = event_name
    df_long = pd.concat([df_long, factor_long, ttm_stage_df], ignore_index=True)

    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )

    return df_long
