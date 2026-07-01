import uuid
import pandas as pd
from datetime import timedelta

from utils.logger import get_logger

logger = get_logger(__name__)

class GroupTranscript:
    """Group transcripts based on datetime and program"""

    def __init__(self, db_manager, df: pd.DataFrame):
        self.db = db_manager
        self.df = df

    def group(self) -> pd.DataFrame:
        df = self.df.copy()

        df = df.sort_values(
            by=['Channel Name', 'Program Title', 'Broadcast Date', 'Broadcast Time']
        ).reset_index(drop=True)

        grouped_rows = []

        for (channel, program, date), group_df in df.groupby(
            ['Channel Name', 'Program Title', 'Broadcast Date'], sort=False
        ):
            group_df = group_df.reset_index(drop=True)
            consolidated = self._consolidate_consecutive(group_df)
            program_id = self._generate_program_id(str(date), program, channel)

            for row in consolidated:
                row['group_id'] = self._generate_group_id(
                    str(date), program, channel, str(row['Broadcast Time']), row['Plain Text']
                )
                row['program_id'] = program_id
                row['Channel Name'] = channel
                row['Program Title'] = program
                row['Broadcast Date'] = date
                grouped_rows.append(row)

        result_df = pd.DataFrame(grouped_rows)

        transcript_data = []
        for _, row in result_df.iterrows():
            transcript_data.append({
                'group_id': row['group_id'],
                'broadcast_date': row['Broadcast Date'].strftime('%Y-%m-%d'),
                'broadcast_time': str(row['Broadcast Time']),
                'program_title': row['Program Title'],
                'channel_name': row['Channel Name'],
                'plain_text': row['Plain Text'],
                'program_id': row['program_id'],
            })

        logger.info(f"Grouped into {len(result_df)} consolidated rows")
        self.db.insert_grouped(transcript_data)
        logger.info("Successfully populated the `grouped` table in the Database")

    def _consolidate_consecutive(self, group_df: pd.DataFrame) -> list:
        consolidated = []
        current_rows = [group_df.iloc[0]]

        for i in range(1, len(group_df)):
            prev_time = self._to_datetime(current_rows[-1]['Broadcast Time'])
            curr_time = self._to_datetime(group_df.iloc[i]['Broadcast Time'])

            if curr_time - prev_time <= timedelta(minutes=1):
                current_rows.append(group_df.iloc[i])
            else:
                consolidated.append(self._merge_rows(current_rows))
                current_rows = [group_df.iloc[i]]

        consolidated.append(self._merge_rows(current_rows))
        return consolidated

    @staticmethod
    def _generate_program_id(broadcast_date: str, program_title: str, channel_name: str) -> str:
        key = f"{broadcast_date}|{program_title}|{channel_name}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    @staticmethod
    def _generate_group_id(broadcast_date: str, program_title: str, channel_name: str,
                           broadcast_time: str, plain_text: str) -> str:
        key = f"{broadcast_date}|{program_title}|{channel_name}|{broadcast_time}|{plain_text}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    @staticmethod
    def _merge_rows(rows: list) -> dict:
        first = rows[0]
        plain_text = ' '.join(str(r['Plain Text']) for r in rows if pd.notna(r['Plain Text']))
        return {
            'Broadcast Time': first['Broadcast Time'],
            'Plain Text': plain_text
        }

    @staticmethod
    def _to_datetime(t) -> pd.Timestamp:
        if isinstance(t, pd.Timestamp):
            return t
        return pd.Timestamp.combine(pd.Timestamp.today().date(), t)
