import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

import shap
import joblib

load_dotenv() 

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

DATA_DIR = "data"
SQLITE_DB_PATH = os.path.join(DATA_DIR, "viral_videos.db")
TRENDING_VIDEOS_TABLE = "trending_videos"
VIDEO_METADATA_TABLE = "video_metadata"

os.makedirs(DATA_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(DATA_DIR, "trained_model.pkl")
PREPROCESSOR_SAVE_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")
SHAP_SAMPLE_PATH = os.path.join(DATA_DIR, "shap_sample_data.pkl")
SHAP_FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "shap_feature_names.pkl")

REGIONS = ["US", "GB", "IN", "CA", "DE"]

FETCH_INTERVAL_HOURS = 1
NUM_FETCH_CYCLES = 30

MIN_TRENDING_APPEARANCES = 2


def initialize_youtube_api():
    try:
        return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        print("Please check your YouTube API Key and network connection.")
        return None

def save_to_sqlite_trending_data(data, db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(data)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

def save_to_sqlite_video_metadata(data, db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            channel_title TEXT,
            publish_time TEXT,
            category_id TEXT,
            description TEXT,
            tags TEXT,
            duration TEXT,
            view_count INTEGER,
            like_count INTEGER,
            comment_count INTEGER,
            region TEXT,
            last_updated_timestamp TEXT
        )
    """)
    conn.commit()

    for item in data:
        video_id = item["video_id"]
        view_count = int(item.get("view_count", 0))
        like_count = int(item.get("like_count", 0))
        comment_count = int(item.get("comment_count", 0))
        last_updated = datetime.now().isoformat()

        cursor.execute(f"""
            INSERT OR REPLACE INTO {table_name} (
                video_id, title, channel_title, publish_time, category_id,
                description, tags, duration, view_count, like_count, comment_count,
                region, last_updated_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, item["title"], item["channel_title"], item["publish_time"],
            item["category_id"], item["description"], item["tags"], item["duration"],
            view_count, like_count, comment_count, item["region"], last_updated
        ))
    conn.commit()
    conn.close()


def fetch_trending_videos_for_region(youtube_service, region_code, current_fetch_timestamp, max_results=50):
    region_trending_data = []
    region_unique_metadata = []
    try:
        request = youtube_service.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=max_results
        )
        response = request.execute()

        for item in response.get("items", []):
            video_id = item["id"]
            view_count = item["statistics"].get("viewCount", 0)
            like_count = item["statistics"].get("likeCount", 0)
            comment_count = item["statistics"].get("commentCount", 0)

            trending_entry = {
                "video_id": video_id,
                "region": region_code,
                "fetch_timestamp": current_fetch_timestamp,
                "view_count_at_fetch": view_count,
                "like_count_at_fetch": like_count,
                "comment_count_at_fetch": comment_count,
            }
            region_trending_data.append(trending_entry)

            video_metadata = {
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "channel_title": item["snippet"]["channelTitle"],
                "publish_time": item["snippet"]["publishedAt"],
                "category_id": item["snippet"]["categoryId"],
                "description": item["snippet"].get("description", ""),
                "tags": "|".join(item["snippet"].get("tags", [])),
                "duration": item["contentDetails"]["duration"],
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": comment_count,
                "region": region_code,
            }
            region_unique_metadata.append(video_metadata)

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while fetching for {region_code}:\n{e.content}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching for {region_code}: {e}")
    return region_trending_data, region_unique_metadata


def fetch_and_store_trending_videos_concurrently(youtube_service):
    all_fetched_trending_data = []
    all_unique_video_metadata = []

    current_fetch_timestamp = datetime.now().isoformat()

    with ThreadPoolExecutor(max_workers=len(REGIONS)) as executor:
        futures = [executor.submit(fetch_trending_videos_for_region, youtube_service, region, current_fetch_timestamp)
                   for region in REGIONS]

        for future in futures:
            trending_data, unique_metadata = future.result()
            all_fetched_trending_data.extend(trending_data)
            all_unique_video_metadata.extend(unique_metadata)

    if all_fetched_trending_data:
        save_to_sqlite_trending_data(all_fetched_trending_data, SQLITE_DB_PATH, TRENDING_VIDEOS_TABLE)
        save_to_sqlite_video_metadata(all_unique_video_metadata, SQLITE_DB_PATH, VIDEO_METADATA_TABLE)


def run_scheduled_data_collection(youtube_service, num_cycles=NUM_FETCH_CYCLES, interval_hours=FETCH_INTERVAL_HOURS):
    for i in range(num_cycles):
        fetch_and_store_trending_videos_concurrently(youtube_service)
        if i < num_cycles - 1:
            time.sleep(interval_hours * 3600)


def load_data_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT
            tv.video_id,
            tv.region AS trending_region,
            tv.fetch_timestamp,
            tv.view_count_at_fetch,
            tv.like_count_at_fetch,
            tv.comment_count_at_fetch,
            vm.title,
            vm.channel_title,
            vm.publish_time,
            vm.category_id,
            vm.description,
            vm.tags,
            vm.duration,
            vm.view_count AS latest_view_count,
            vm.like_count AS latest_like_count,
            vm.comment_count AS latest_comment_count
        FROM {TRENDING_VIDEOS_TABLE} AS tv
        JOIN {VIDEO_METADATA_TABLE} AS vm
        ON tv.video_id = vm.video_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def parse_duration(duration_iso8601):
    if not isinstance(duration_iso8601, str):
        return np.nan
    try:
        duration_iso8601 = duration_iso8601.replace('PT', '')
        total_seconds = 0
        if 'H' in duration_iso8601:
            hours, duration_iso8601 = duration_iso8601.split('H')
            total_seconds += int(hours) * 3600
        if 'M' in duration_iso8601:
            minutes, duration_iso8601 = duration_iso8601.split('M')
            total_seconds += int(minutes) * 60
        if 'S' in duration_iso8601:
            seconds, _ = duration_iso8601.split('S')
            total_seconds += int(seconds.replace('S', ''))
        return total_seconds / 60
    except Exception:
        return np.nan

def clean_and_transform_data(df):
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors='coerce', utc=True)
    df["fetch_timestamp"] = pd.to_datetime(df["fetch_timestamp"], errors='coerce', utc=True)

    df["view_count"] = pd.to_numeric(df["view_count_at_fetch"], errors='coerce').fillna(0).astype(int)
    df["like_count"] = pd.to_numeric(df["like_count_at_fetch"], errors='coerce').fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df["comment_count_at_fetch"], errors='coerce').fillna(0).astype(int)

    df.dropna(subset=["publish_time", "view_count"], inplace=True)
    df["tags"] = df["tags"].fillna("")
    df["description"] = df["description"].fillna("")

    df["duration_minutes"] = df["duration"].apply(parse_duration)
    df.dropna(subset=["duration_minutes"], inplace=True)

    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_weekday"] = df["publish_time"].dt.day_name()
    df["publish_month"] = df["publish_time"].dt.month_name()
    df["publish_year"] = df["publish_time"].dt.year
    df["time_to_trend_hours"] = (df["fetch_timestamp"] - df["publish_time"]).dt.total_seconds() / 3600
    df["time_to_trend_hours"] = df["time_to_trend_hours"].apply(lambda x: max(x, 0))

    df["likes_per_view"] = df.apply(lambda row: row["like_count"] / row["view_count"] if row["view_count"] > 0 else 0, axis=1)
    df["comments_per_view"] = df.apply(lambda row: row["comment_count"] / row["view_count"] if row["view_count"] > 0 else 0, axis=1)

    df["cleaned_title"] = df["title"].str.lower().replace(r"[^\w\s]", "", regex=True).fillna("")
    df["cleaned_tags"] = df["tags"].str.lower().replace(r"[^\w\s|]", "", regex=True).fillna("")

    return df


def perform_eda(df):
    plt.figure(figsize=(12, 6))
    top_channels = df["channel_title"].value_counts().head(10)
    sns.barplot(x=top_channels.values, y=top_channels.index, palette="viridis")
    plt.title("Top 10 Channels by Number of Trending Appearances")
    plt.xlabel("Number of Trending Appearances")
    plt.ylabel("Channel Title")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["duration_minutes"], bins=50, kde=True)
    plt.title("Distribution of Video Durations (Minutes)")
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Frequency")
    plt.xlim(0, df["duration_minutes"].quantile(0.99))
    plt.tight_layout()
    plt.show()

    all_titles = " ".join(df["cleaned_title"].dropna())
    if all_titles:
        wordcloud_titles = WordCloud(width=800, height=400, background_color="white").generate(all_titles)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_titles, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Trending Video Titles")
        plt.show()

    all_tags = " ".join(df["cleaned_tags"].dropna().str.replace("|", " "))
    if all_tags:
        wordcloud_tags = WordCloud(width=800, height=400, background_color="white").generate(all_tags)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_tags, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Trending Video Tags")
        plt.show()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["likes_per_view"], bins=50, kde=True)
    plt.title("Distribution of Likes per View")
    plt.xlabel("Likes per View Ratio")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(df["comments_per_view"], bins=50, kde=True)
    plt.title("Distribution of Comments per View")
    plt.xlabel("Comments per View Ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    numerical_cols = [
        "view_count", "like_count", "comment_count", "duration_minutes",
        "likes_per_view", "comments_per_view", "time_to_trend_hours"
    ]
    numerical_cols_existing = [col for col in numerical_cols if col in df.columns]
    if numerical_cols_existing:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols_existing].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Numerical Features")
        plt.tight_layout()
        plt.show()

    fig_hour, ax_hour = plt.subplots(figsize=(10, 6))
    sns.countplot(x='publish_hour', data=df, palette='cubehelix', ax=ax_hour)
    ax_hour.set_title('Number of Trending Videos by Publish Hour')
    ax_hour.set_xlabel('Publish Hour')
    ax_hour.set_ylabel('Count')
    plt.tight_layout()
    plt.show()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig_weekday, ax_weekday = plt.subplots(figsize=(10, 6))
    sns.countplot(x='publish_weekday', data=df, order=weekday_order, palette='crest', ax=ax_weekday)
    ax_weekday.set_title('Number of Trending Videos by Publish Weekday')
    ax_weekday.set_xlabel('Publish Weekday')
    ax_weekday.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def feature_engineer(df):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    trending_counts = pd.read_sql_query(f"SELECT video_id, COUNT(*) as trending_appearances FROM {TRENDING_VIDEOS_TABLE} GROUP BY video_id", conn)
    conn.close()

    unique_videos_df = df.drop_duplicates(subset=['video_id']).copy()
    unique_videos_df = unique_videos_df.merge(trending_counts, on='video_id', how='left')
    unique_videos_df['trending_appearances'] = unique_videos_df['trending_appearances'].fillna(0).astype(int)

    unique_videos_df["is_trending"] = (unique_videos_df["trending_appearances"] >= MIN_TRENDING_APPEARANCES).astype(int)

    df = df.merge(unique_videos_df[['video_id', 'is_trending']], on='video_id', how='left')

    df["combined_text"] = df["cleaned_title"] + " " + df["cleaned_tags"].str.replace("|", " ")
    df["combined_text"] = df["combined_text"].fillna("")

    df["likes_per_view_x_duration"] = df["likes_per_view"] * df["duration_minutes"]
    df["views_per_hour_to_trend"] = df.apply(lambda row: row["view_count"] / row["time_to_trend_hours"] if row["time_to_trend_hours"] > 0 else 0, axis=1)

    return df


def build_ml_pipeline(df):
    features = [
        "view_count", "like_count", "comment_count", "duration_minutes",
        "likes_per_view", "comments_per_view", "time_to_trend_hours",
        "publish_hour", "publish_weekday", "publish_month", "publish_year",
        "likes_per_view_x_duration", "views_per_hour_to_trend",
        "combined_text"
    ]
    target = "is_trending"

    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target]

    if y.nunique() < 2:
        print("Warning: Target variable 'is_trending' has less than 2 unique classes. Cannot perform classification.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = [
        "view_count", "like_count", "comment_count", "duration_minutes",
        "likes_per_view", "comments_per_view", "time_to_trend_hours",
        "publish_hour", "likes_per_view_x_duration", "views_per_hour_to_trend"
    ]
    categorical_features = ["publish_weekday", "publish_month", "publish_year"]
    text_features = "combined_text"

    preprocessor_transformers = []

    num_feats_exist = [f for f in numerical_features if f in X_train.columns]
    if num_feats_exist:
        preprocessor_transformers.append(('num', StandardScaler(), num_feats_exist))

    cat_feats_exist = [f for f in categorical_features if f in X_train.columns]
    if cat_feats_exist:
        preprocessor_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats_exist))

    if text_features in X_train.columns:
        preprocessor_transformers.append(('text', TfidfVectorizer(max_features=10000, stop_words='english'), text_features))


    preprocessor = ColumnTransformer(
        transformers=preprocessor_transformers,
        remainder='drop'
    )

    log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))])
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

    param_grid_log_reg = {
        'classifier__C': [0.1, 1, 10]
    }
    param_grid_rf = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    }

    models_to_tune = {
        "Logistic Regression": (log_reg_pipeline, param_grid_log_reg),
        "Random Forest": (rf_pipeline, param_grid_rf),
    }

    best_model = None
    best_score = -1

    for name, (pipeline, param_grid) in models_to_tune.items():
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.best_estimator_.predict(X_test)
        y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    return best_model, preprocessor, X_test, y_test


def save_ml_assets(model, preprocessor, X_test, feature_names_original):
    try:
        joblib.dump(model, MODEL_SAVE_PATH)
    except Exception as e:
        print(f"Error saving model: {e}")

    try:
        joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    except Exception as e:
        print(f"Error saving preprocessor: {e}")

    if model and len(X_test) > 0:
        try:
            if isinstance(model.named_steps['classifier'], (RandomForestClassifier)):
                fitted_preprocessor = model.named_steps['preprocessor']
                X_test_transformed = fitted_preprocessor.transform(X_test)

                transformed_feature_names = []
                for name, transformer_pipeline, cols in fitted_preprocessor.transformers:
                    if transformer_pipeline != 'drop' and hasattr(transformer_pipeline, 'get_feature_names_out'):
                        if name == 'num':
                            transformed_feature_names.extend(transformer_pipeline.get_feature_names_out(cols))
                        else:
                             transformed_feature_names.extend(transformer_pipeline.get_feature_names_out())
                transformed_feature_names = np.array(transformed_feature_names)

                explainer = shap.TreeExplainer(model.named_steps['classifier'])
                shap_values = explainer.shap_values(X_test_transformed)

                num_samples_for_shap = min(50, len(X_test_transformed))
                if num_samples_for_shap > 0:
                    shap_sample_data = X_test_transformed[:num_samples_for_shap]
                    joblib.dump(shap_sample_data, SHAP_SAMPLE_PATH)
                    joblib.dump(transformed_feature_names, SHAP_FEATURE_NAMES_PATH)

                    shap.initjs()
                    if isinstance(shap_values, list):
                        shap_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], shap_sample_data[0,:], feature_names=transformed_feature_names, matplotlib=True)
                    else:
                        shap_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], shap_sample_data[0,:], feature_names=transformed_feature_names, matplotlib=True)
                    plt.show()

        except Exception as e:
            print(f"Error generating or saving SHAP assets: {e}")
    else:
        pass


def main():
    youtube_service = initialize_youtube_api()
    if not youtube_service:
        return

    run_scheduled_data_collection(youtube_service, num_cycles=NUM_FETCH_CYCLES, interval_hours=FETCH_INTERVAL_HOURS)

    df_raw = load_data_from_sqlite(SQLITE_DB_PATH)
    if df_raw.empty:
        return

    df_cleaned = clean_and_transform_data(df_raw.copy())

    perform_eda(df_cleaned.copy())

    df_engineered = feature_engineer(df_cleaned.copy())

    target = "is_trending"

    if 'is_trending' not in df_engineered.columns or df_engineered['is_trending'].nunique() < 2:
        print("Insufficient data or target label variance for ML training. Skipping ML pipeline.")
        return

    trained_model, preprocessor, X_test, y_test = build_ml_pipeline(df_engineered.copy())

    save_ml_assets(trained_model, preprocessor, X_test, X_test.columns.tolist())

    if trained_model:
        sample_data = pd.DataFrame([{
            "video_id": "hypothetical_id_123",
            "title": "Amazing new cat compilation funny cute moments",
            "channel_title": "CatLoversDaily",
            "publish_time": datetime.now().isoformat(),
            "category_id": "15",
            "description": "This is a fun video where we try a new challenge!",
            "tags": "challenge,viral,fun,new",
            "duration": "PT5M30S",
            "view_count": 500000,
            "like_count": 50000,
            "comment_count": 1500,
            "region": "US",
            "duration_minutes": parse_duration("PT5M30S"),
            "publish_hour": datetime.now().hour,
            "publish_weekday": datetime.now().strftime('%A'),
            "publish_month": datetime.now().strftime('%B'),
            "publish_year": datetime.now().year,
            "time_to_trend_hours": 2.5,
            "likes_per_view": 50000 / 500000,
            "comments_per_view": 1500 / 500000,
            "cleaned_title": "amazing new cat compilation funny cute moments",
            "cleaned_tags": "cat|funny|cute|compilation|pets|animals",
            "combined_text": "amazing new cat compilation funny cute moments cat funny cute compilation pets animals",
            "likes_per_view_x_duration": (50000 / 500000) * parse_duration("PT5M30S"),
            "views_per_hour_to_trend": 500000 / 2.5
        }])

        sample_data_for_prediction = sample_data[[col for col in sample_data.columns if col in df_engineered.columns and col != target]]

        try:
            prediction = trained_model.predict(sample_data_for_prediction)[0]
            prediction_proba = trained_model.predict_proba(sample_data_for_prediction)[:, 1][0]
        except Exception as e:
            print(f"Error during single prediction: {e}")
            print("Please ensure your sample_data columns match the training data used by the preprocessor.")


if __name__ == "__main__":
    main()
