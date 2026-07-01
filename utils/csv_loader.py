import pandas as pd
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

NEWS_SCHEDULE_PATH = "config/news_sched.csv"


class CSVLoader:
    """Load transcripts from CSV File"""

    @staticmethod
    def _validate_required_columns(df, required):
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info(f"Validated {len(required)} required columns")

    @staticmethod
    def _clean_columns(df, required):
        df = df[required]

    @staticmethod
    def _deduplicate(df: pd.DataFrame):
        df_dedup = df.drop_duplicates(subset=['Broadcast Date','Broadcast Time','Channel Name'])
        logger.info(f"Before deduplication: {len(df)} rows")
        logger.info(f"After deduplication: {len(df_dedup)} rows")
        logger.info(f"Rows dropped: {len(df)-len(df_dedup)}")
        return df_dedup

    @staticmethod
    def _normalize_whitespace(df: pd.DataFrame):
        df_stripped = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df_stripped

    @staticmethod
    def _parse_datetime(df: pd.DataFrame):
        df['Broadcast Date'] = pd.to_datetime(df['Broadcast Date'], errors='coerce')

        # Extract time with AM/PM, discard everything after (timezone, etc.)
        df['Broadcast Time'] = df['Broadcast Time'].str.extract(r'(\d{1,2}:\d{2}\s(?:AM|PM))', expand=False)

        # Log sample times after stripping
        logger.info(f"Sample times after stripping GMT: {df['Broadcast Time'].head(5).tolist()}")

        # Parse with 12-hour format
        df['Broadcast Time'] = pd.to_datetime(df['Broadcast Time'], format='%I:%M %p', errors='coerce').dt.time

        # Find unparseable times
        invalid_mask = df['Broadcast Time'].isna()
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} unparseable times. First few examples:")
            for idx in df[invalid_mask].head(10).index:
                logger.warning(f"  Row {idx}: Raw value before parsing attempted: '{df.loc[idx, 'Broadcast Time']}'")
        else:
            logger.info(f"Successfully parsed all {len(df)} broadcast times")

        return df

    @staticmethod
    def _load_news_schedule(path: str) -> dict:
        """Parse news_sched.csv into {(day, channel): [(start, end), ...]}."""
        sched_df = pd.read_csv(path)
        schedule = {}
        for _, row in sched_df.iterrows():
            key = (row['day'], row['channel'])
            start = pd.to_datetime(row['start']).time()
            end = pd.to_datetime(row['end']).time()
            schedule.setdefault(key, []).append((start, end))
        logger.info(f"Loaded news schedule: {len(sched_df)} windows across "
                    f"{len(schedule)} (day, channel) pairs from {path}")
        return schedule

    @staticmethod
    def _filter_news_schedule(df: pd.DataFrame, schedule: dict) -> pd.DataFrame:
        """Drop rows whose Broadcast Time falls outside all news windows for (day, channel).

        Rows with an unparseable date or time, or whose channel has no entry in the
        schedule, are kept (conservative — unknown = no filter applied).
        """
        def in_schedule(row):
            date = row['Broadcast Date']
            t = row['Broadcast Time']
            channel = row['Channel Name']

            if pd.isna(date) or t is None:
                return True

            key = (date.day_name(), channel)
            if key not in schedule:
                return True

            return any(start <= t <= end for start, end in schedule[key])

        mask = df.apply(in_schedule, axis=1)
        kept = int(mask.sum())
        dropped = int((~mask).sum())
        logger.info(f"News schedule filter: kept {kept} rows, dropped {dropped} rows")
        return df[mask].reset_index(drop=True)

    @staticmethod
    def _sort_by_hierarchy(df: pd.DataFrame):
        """Sort dataframe by hierarchy: Channel Name → Broadcast Date → Broadcast Time."""
        df_sorted = df.sort_values(
            by=['Channel Name', 'Broadcast Date', 'Broadcast Time'],
            na_position='last'
        )
        logger.info(f"Sorted by: Channel Name -> Broadcast Date -> Broadcast Time")
        return df_sorted

    @staticmethod
    def load(file_path: str):
        df = pd.read_csv(file_path, encoding="utf-8")
        required_cols = [
            "Broadcast Date",
            "Broadcast Time",
            "Program Title",
            "Channel Name",
            "Plain Text"
        ]
        CSVLoader._validate_required_columns(df, required_cols)
        CSVLoader._clean_columns(df, required_cols)
        df = CSVLoader._deduplicate(df)
        df = CSVLoader._normalize_whitespace(df)
        df = CSVLoader._parse_datetime(df)
        schedule = CSVLoader._load_news_schedule(NEWS_SCHEDULE_PATH)
        df = CSVLoader._filter_news_schedule(df, schedule)
        df = CSVLoader._sort_by_hierarchy(df)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
