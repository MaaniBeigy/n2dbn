"""Fitbit module for LifeSnaps Dataset"""

# ---------------------------- import libraries and modules ---------------------------
from typing import Dict, List, Optional

import pandas as pd
from lifesnaps.helper import try_convert_to_numeric
from pymongo import MongoClient

# ------------------------------------- functions -------------------------------------


def fetch_fitbit_event_data(
    event_name: str,
    event_timestamp: str,
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "fitbit",
    event_properties: Optional[List[str]] = None,
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches data from the Fitbit collection in the LifeSnaps Dataset.

    Args:
        event_name (str): The name of the event type to query.
        event_timestamp (str): The timestamp field in the data.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "fitbit".
        event_properties (List[str], optional): The properties of the event to fetch. Defaults to None.
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
            "user_id": str(item["id"]),
            "event_time": item["data"].get(event_timestamp, None),
            "event_name": str(item["type"]),
        }
        if event_properties:
            # Extract nested fields from the data object and add the event properties
            for prop in event_properties:
                # Check if the property exists in 'data' and then within 'value'
                if item["data"].get(prop):
                    value = item["data"].get(prop, None)
                elif (
                    item["data"].get(prop) is None
                    and item["data"].get("value") is not None
                ):
                    value = item["data"]["value"].get(prop, None)
                else:
                    value = None

                # Convert the value to a numeric type if applicable
                if value is not None:
                    value = try_convert_to_numeric(value)

                row[prop] = value
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
        value_vars=event_properties,
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


def fetch_fitbit_event_data_aggregated(
    event_name: str,
    event_timestamp: str,
    mongo_host: str = "mongodb://localhost:27017/",
    database_name: str = "rais_anonymized",
    collection_name: str = "fitbit",
    event_properties: Optional[
        Dict[str, str]
    ] = None,  # Key: property, Value: aggregation rule
    timestamp_format: Optional[str] = None,
    time_bin_size: Optional[int] = None,
    timestamp_aggregation_rule: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches and aggregates data from the Fitbit collection in the LifeSnaps Dataset.

    Args:
        event_name (str): The name of the event type to query.
        event_timestamp (str): The timestamp field in the data.
        mongo_host (str, optional): The MongoDB host URI. Defaults to "mongodb://localhost:27017/".
        database_name (str, optional): The name of the database. Defaults to "rais_anonymized".
        collection_name (str, optional): The name of the collection. Defaults to "fitbit".
        event_properties (Dict[str, str], optional): The properties of the event to fetch and their aggregation rules.
        timestamp_format (str, optional): The datetime format for fetching event timestamps in MongoDB. Defaults to None.
        time_bin_size (int, optional): The size of the time bins for aggregation, in minutes. Defaults to None.
        timestamp_aggregation_rule (str, optional): The aggregation rule (e.g., "$max") for the timestamp field. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated event data in long format.
    """
    # ------------------------------- Connect to MongoDB ------------------------------
    client = MongoClient(mongo_host)
    db = client[database_name]
    collection = db[collection_name]

    # --------------------------- mongodb aggregation query  --------------------------
    pipeline = [{"$match": {"type": event_name}}]

    if timestamp_format == "%m/%d/%Y %H:%M:%S":
        pipeline.extend(
            [
                {
                    "$addFields": {
                        "dateParts": {"$split": [f"$data.{event_timestamp}", "/"]}
                    }
                },
                {
                    "$addFields": {
                        "dayTimeParts": {
                            "$split": [{"$arrayElemAt": ["$dateParts", 2]}, " "]
                        }
                    }
                },
                {
                    "$addFields": {
                        "year": {
                            "$substr": [{"$arrayElemAt": ["$dayTimeParts", 0]}, 0, 2]
                        },
                        "month": {"$arrayElemAt": ["$dateParts", 0]},
                        "day": {"$arrayElemAt": ["$dateParts", 1]},
                        "time": {"$arrayElemAt": ["$dayTimeParts", 1]},
                    }
                },
                {
                    "$addFields": {
                        "fullDateTime": {
                            "$concat": [
                                {
                                    "$cond": [
                                        {"$lt": [{"$toInt": "$year"}, 50]},
                                        {"$concat": ["20", "$year"]},
                                        {"$concat": ["19", "$year"]},
                                    ]
                                },
                                "/",
                                "$month",
                                "/",
                                "$day",
                                " ",
                                "$time",
                            ]
                        }
                    }
                },
                {
                    "$addFields": {
                        "parsedDateTime": {
                            "$dateFromString": {
                                "dateString": "$fullDateTime",
                                "format": "%Y/%m/%d %H:%M:%S",
                            }
                        }
                    }
                },
            ]
        )
    else:
        pipeline.append(
            {
                "$addFields": {
                    "parsedDateTime": {
                        "$dateFromString": {
                            "dateString": f"$data.{event_timestamp}",
                            "format": timestamp_format,
                        }
                    }
                }
            }
        )

    # -------------------------- Group stage for aggregation --------------------------
    group_stage = {
        "$group": {
            "_id": {
                "interval": {
                    "$dateTrunc": {
                        "date": "$parsedDateTime",
                        "unit": "minute",
                        "binSize": time_bin_size,
                    }
                },
                "user_id": "$id",
            },
            "lastDateTime": {timestamp_aggregation_rule: "$parsedDateTime"},
        }
    }

    property_aliases = {}
    for prop, agg_rule in event_properties.items():
        prop_alias = prop.replace(".", "_")
        property_aliases[prop] = prop_alias
        nested_path = f"$data.{prop}"
        group_stage["$group"][f"{prop_alias}_total"] = {
            agg_rule: {"$toDouble": nested_path}
        }

    pipeline.append(group_stage)

    # -------------------------------- Projection stage -------------------------------
    project_stage = {
        "$project": {
            "_id": 0,
            "user_id": "$_id.user_id",
            "lastDateTime": 1,
        }
    }

    for prop, prop_alias in property_aliases.items():
        project_stage["$project"][f"{prop_alias}_total"] = 1

    pipeline.append(project_stage)

    # ------------------------------ Load aggregated data -----------------------------
    items = list(collection.aggregate(pipeline))

    # --------------------------------- Transform data --------------------------------
    data = []
    for item in items:
        base_row = {
            "user_id": str(item["user_id"]),
            "event_time": item["lastDateTime"],
            "event_name": event_name,
        }
        for prop, prop_alias in property_aliases.items():
            row = base_row.copy()
            value = item.get(f"{prop_alias}_total", None)
            if value is not None:
                row["event_property"] = prop
                row["event_value"] = (
                    round(value, 2) if isinstance(value, (int, float)) else value
                )
                data.append(row)

    # -------------------------------- Create DataFrame -------------------------------
    df_long = pd.DataFrame(data)
    # --------------------- Remove rows with null event_value data --------------------
    df_long = df_long[df_long["event_value"].notna()]

    # ------------------------ Add the event_value_type column ------------------------
    df_long["event_value_type"] = df_long["event_value"].apply(
        lambda x: type(x).__name__
    )
    df_long["event_property"] = df_long["event_property"].str.replace(
        "value.", "", regex=False
    )

    return df_long
