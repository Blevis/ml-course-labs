"""
Starting tables for each week, rebuilt from the raw Olist files.

Every lab from Week 2 onwards begins with one of these, so you can start any
week even if you missed the one before. Each function takes a few seconds and
always gives the same result as that week's solution notebook.

Usage, from inside a week folder:

    import sys
    sys.path.insert(0, "..")
    from checkpoints import week_01

    orders = week_01()
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw" / "olist"

# The usable window. Months before 2017 and after August 2018 have almost no
# orders, and none of the late-2018 ones were ever delivered.
START = "2017-01-01"
END = "2018-09-01"

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _check_data() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"No data at {RAW}. Run `python data/download.py` from the repository root first."
        )


def week_01() -> pd.DataFrame:
    """Delivered orders in the usable window, with the customer's state and the
    is_late target. One row per order.

    is_late is 1 when the order arrived on a later calendar day than promised.
    Orders that were never delivered are left out, because we cannot label them.
    """
    _check_data()
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=DATE_COLUMNS)
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")

    df = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")

    delivered = df["order_delivered_customer_date"].notna()
    in_window = df["order_purchase_timestamp"].between(START, END, inclusive="left")
    df = df[delivered & in_window].copy()

    delivered_day = df["order_delivered_customer_date"].dt.normalize()
    df["is_late"] = (delivered_day > df["order_estimated_delivery_date"]).astype(int)

    return df.reset_index(drop=True)    # original file order, so random splits match the notebooks


# --- Week 02 -----------------------------------------------------------------

# Split boundaries by purchase date: train on the past, validate on the next
# two months, test on the most recent four.
TRAIN_END = "2018-03-01"
VALID_END = "2018-05-01"

# Two categories are missing from Olist's own translation table.
EXTRA_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers",
}

# A generous box around Brazil. Coordinates outside it are errors.
LAT_RANGE = (-34.0, 6.0)
LNG_RANGE = (-74.0, -34.0)


def clean_city(name: str) -> str:
    """Lower case, no accents, and drop anything after a slash or comma,
    so 'Sao Paulo / SP' and 'são paulo' both become 'sao paulo'."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = re.split(r"[/,\\]", name)[0]
    return " ".join(name.lower().split())


def week_02() -> pd.DataFrame:
    """One clean row per delivered order, joined from all tables except reviews,
    with a `split` column: train, valid, or test.

    Reviews are left out on purpose: almost all are written after delivery,
    so they cannot be used to predict lateness at purchase time.
    """
    _check_data()
    orders = week_01()

    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv")
    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
    geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")
    names = pd.read_csv(RAW / "product_category_name_translation.csv")

    # Products: fix the misspelled columns, translate categories
    products = products.rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    })
    translation = dict(zip(names["product_category_name"], names["product_category_name_english"]))
    translation.update(EXTRA_TRANSLATIONS)
    products["category"] = products["product_category_name"].map(translation).fillna("unknown")

    # Sellers: consistent city names
    sellers["seller_city"] = sellers["seller_city"].map(clean_city)

    # Geolocation: drop duplicates and impossible points, one location per zip prefix
    geo = geo.drop_duplicates()
    inside = geo["geolocation_lat"].between(*LAT_RANGE) & geo["geolocation_lng"].between(*LNG_RANGE)
    zip_coords = geo[inside].groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()

    # Items to one row per order
    items = items.merge(products[["product_id", "category", "product_weight_g"]], on="product_id", how="left")
    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", lambda w: w.sum(min_count=1)),
    )
    first_item = items[items["order_item_id"] == 1].set_index("order_id")[["seller_id", "category"]]
    first_item = first_item.join(
        sellers.set_index("seller_id")[["seller_zip_code_prefix", "seller_city", "seller_state"]], on="seller_id"
    )

    # Payments to one row per order
    per_payment = payments.groupby("order_id").agg(
        n_payments=("payment_sequential", "size"),
        total_paid=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
    )
    per_payment["payment_type"] = payments.sort_values("payment_sequential").groupby("order_id")["payment_type"].first()

    # Join everything, checking the row count never changes
    n = len(orders)
    df = orders.merge(customers[["customer_id", "customer_unique_id", "customer_zip_code_prefix"]], on="customer_id", how="left")
    df = df.join(per_order, on="order_id").join(first_item, on="order_id").join(per_payment, on="order_id")
    df = df.join(zip_coords.rename(columns=lambda c: c.replace("geolocation", "customer")), on="customer_zip_code_prefix")
    df = df.join(zip_coords.rename(columns=lambda c: c.replace("geolocation", "seller")), on="seller_zip_code_prefix")
    assert len(df) == n, "a join changed the number of orders"

    # Flag impossible timelines rather than dropping them: the target does not depend on them
    df["bad_timeline"] = (
        (df["order_delivered_carrier_date"] < df["order_purchase_timestamp"])
        | (df["order_delivered_customer_date"] < df["order_delivered_carrier_date"])
    ).astype(int)

    # The split, by purchase date
    purchase = df["order_purchase_timestamp"]
    df["split"] = "test"
    df.loc[purchase < VALID_END, "split"] = "valid"
    df.loc[purchase < TRAIN_END, "split"] = "train"

    return df.reset_index(drop=True)    # original file order, so random splits match the notebooks


# --- Week 03 -----------------------------------------------------------------

# Features computed from a single row. Safe to build before or after the split.
ROW_WISE_FEATURES = [
    "distance_km",
    "same_state",
    "window_days",
    "freight_ratio",
    "total_freight",
    "total_weight_g",
    "n_items",
    "total_price",
]
STATE_COLUMNS = ["customer_state", "seller_state"]


def haversine_km(lat1, lng1, lat2, lng2):
    """Distance in kilometres between two points on the Earth's surface."""
    import numpy as np

    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def week_03() -> pd.DataFrame:
    """The Week 2 table plus the row-wise features chosen in Week 3."""
    df = week_02()
    df["distance_km"] = haversine_km(df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["window_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"].dt.normalize()).dt.days
    df["freight_ratio"] = df["total_freight"] / df["total_price"]
    return df


def week_03_pipeline():
    """The model chosen in Week 3, not yet fitted: the row-wise features and the
    two states, into a logistic regression. Fit it on the training split."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, ROW_WISE_FEATURES),
        ("states", OneHotEncoder(handle_unknown="ignore"), STATE_COLUMNS),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])