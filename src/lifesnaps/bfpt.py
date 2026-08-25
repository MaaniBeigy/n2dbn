"""bfpt module for LifeSnaps Dataset"""
# Survey to assess users on the Big Five model of Openness, Conscientiousness, Extraversion, Agreeableness & Neuroticism.
# ---------------------------- import libraries and modules ---------------------------
from typing import Optional

import pandas as pd
from lifesnaps.constants import (
    BFPT_EVENT_PROPERTIES,
    BFPT_INVERSE_QUESTION_CODES,
    IPIP_FACTOR_CODES,
)
from lifesnaps.helper import inverse_scores, try_convert_to_numeric
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_bfpt_event_data(
    event_name: str = "bfpt",
    event_timestamp: str = "submitdate",
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "surveys",
    timestamp_format: Optional[str] = "%d/%m/%Y %H:%M",
) -> pd.DataFrame:
    """
    Fetches data from the bfpt collection in the LifeSnaps Dataset.

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
        for prop in BFPT_EVENT_PROPERTIES:
            value = item["data"].get(prop, None)

            # Convert the value to a numeric type if applicable
            if value is not None:
                value = try_convert_to_numeric(value)
            # Apply inverse_ipip_score if the property is keyed negatively
            if not BFPT_INVERSE_QUESTION_CODES.get(prop, True):
                value = inverse_scores(value, min=1, max=5)
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
        value_vars=BFPT_EVENT_PROPERTIES,
        var_name="event_property",
        value_name="event_value",
    )

    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]
    # ------- Remove duplicates for each user_id, event_time, and event_property ------
    df_long = df_long.drop_duplicates(
        subset=["user_id", "event_time", "event_property"], keep="last"
    )
    # ---------------------------- Calculate factor scores ----------------------------
    factor_scores = {}
    for factor_name, factor_code in {
        "extraversion": 1,
        "agreeableness": 2,
        "conscientiousness": 3,
        "stability": 4,
        "intellect": 5,
    }.items():
        factor_scores[factor_name] = (
            df_long[
                df_long["event_property"].isin(
                    [k for k, v in IPIP_FACTOR_CODES.items() if v == factor_code]
                )
            ]
            .groupby("user_id")["event_value"]
            .sum()
            .reset_index()
        )

    # Create long-format DataFrame for the factor scores
    factor_df_list = []
    for factor_name, score_df in factor_scores.items():
        score_df["event_property"] = factor_name
        score_df["event_value"] = score_df["event_value"].astype(float)
        score_df["event_time"] = df_long["event_time"].min()
        score_df["event_name"] = event_name
        factor_df_list.append(score_df)

    df_factors = pd.concat(factor_df_list, ignore_index=True)

    # Concatenate the factor scores DataFrame with df_long
    df_long = pd.concat([df_long, df_factors], ignore_index=True)

    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )

    return df_long
