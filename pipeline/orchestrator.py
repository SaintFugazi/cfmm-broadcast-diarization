import os
import pandas as pd

from .group import GroupTranscript
from .punctuation import PunctuationRestoration
from .relevance import RelevanceAgent
from .segmentation import SegmentationAgent, IndexedSegmentationAgent
from .attribution import AttributionAgent
from .verify import VerificationAgent
from .count import TokenCounter

from utils.db_manager import DatabaseManager
from utils.logger import get_logger
from utils.csv_loader import CSVLoader
from config.constants import (
    RowStatus,
    COUNT_OVER,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    VERIFICATION_CONFIDENCE_THRESHOLD,
)

logger = get_logger(__name__)

STEPS = ["group", "relevance", "punctuation", "count", "segmentation", "attribution"]

class DiarizationPipeline:
    """Main Diarization Orchestration"""

    def __init__(self, filename: str, limit: int = None, stop_after: str = None):
        self.filename = filename
        self.limit = limit
        self.stop_after = stop_after
        self.file = f"data/input/{self.filename}.csv"
        db_name = f"{self.filename}_limit_{limit}.db" if limit else f"{self.filename}.db"
        self.db = DatabaseManager(db_name)

        # Cost tracking across all LLM stages
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.stage_costs = {}  # {stage_name: (in_tokens, out_tokens)}

    def _should_stop(self, step: str) -> bool:
        if self.stop_after and self.stop_after == step:
            logger.info(f"--test '{step}': stopping after this step.")
            return True
        return False

    def _accumulate_costs(self, stage_name: str, agent) -> None:
        """Extract token counts from an agent and accumulate into pipeline totals."""
        in_tokens = getattr(agent, 'total_input_tokens', 0)
        out_tokens = getattr(agent, 'total_output_tokens', 0)
        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens
        self.stage_costs[stage_name] = (in_tokens, out_tokens)

    def run(self):
        """Execute Diarization Pipeline"""

        try:
            logger.info("=" * 60)
            logger.info("DIARIZATION PIPELINE START")
            if self.stop_after:
                logger.info(f"TEST MODE: stopping after '{self.stop_after}'")
            logger.info("=" * 60)

            # Step 1: Ingest
            print("\n[STEP 1] Data Ingestion")
            raw_input = self._ingest_csv()

            # Step 2: Group
            print("\n[STEP 2] Grouping of Transcripts")
            self._group(raw_input)
            if self._should_stop("group"):
                return

            # Step 3: Relevance Filter (news vs. non-news) — runs on raw PENDING rows
            # so non-news content is dropped before punctuation is paid for.
            print("\n[STEP 3] Relevance Filtering")
            df = self.db.get_row_by_status(RowStatus.PENDING, "grouped")
            if len(df) > 0:
                logger.info("Filtering PENDING rows for relevance...")
                self._filter_relevance(df)
            else:
                logger.info("No PENDING rows detected from 'grouped' table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_R, "grouped")
            if len(df_failed) > 0:
                logger.info("Filtering FAILED RELEVANCE rows...")
                self._filter_relevance(df_failed)
            else:
                logger.info("No FAILED RELEVANCE rows detected from 'grouped' table. Proceeding...")

            if self._should_stop("relevance"):
                return

            # Step 4: Punctuation — only RELEVANT rows, non-news already excluded
            print("\n[STEP 4] Punctuation Restoration")
            df = self.db.get_row_by_status(RowStatus.RELEVANT, "grouped")
            if len(df) > 0:
                logger.info("Punctuating RELEVANT rows...")
                self._punctuate(df)
            else:
                logger.info("No RELEVANT rows detected from 'grouped' table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_P, "grouped")
            if len(df_failed) > 0:
                logger.info("Punctuating FAILED PUNCTUATION rows...")
                self._punctuate(df_failed)
            else:
                logger.info("No FAILED PUNCTUATION rows detected from 'grouped' table. Proceeding...")

            if self._should_stop("punctuation"):
                return

            # Step 5: Token Count
            print("\n[STEP 5] Transcript Size Classification")
            df = self.db.get_punctuated_uncounted()
            if len(df) > 0:
                logger.info(f"Counting tokens for {len(df)} uncounted PUNCTUATED rows...")
                self._count(df)
            else:
                logger.info("No uncounted PUNCTUATED rows detected from `grouped` table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_C, "grouped")
            if len(df_failed) > 0:
                logger.info("Counting tokens for FAILED COUNT rows...")
                self._count(df_failed)
            else:
                logger.info("No FAILED COUNT rows detected from `grouped` table. Proceeding...")

            if self._should_stop("count"):
                return

            # Step 6: Segmentation (split each group at speaker changes; no attribution)
            print("\n[STEP 6] Segmentation")
            df = self.db.get_row_by_status(RowStatus.PUNCTUATED, "grouped")
            if len(df) > 0:
                logger.info("Segmenting PUNCTUATED rows...")
                self._segment(df)
            else:
                logger.info("No PUNCTUATED rows detected from `grouped` table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_S, "grouped")
            if len(df_failed) > 0:
                logger.info("Segmenting FAILED SEGMENTATION rows...")
                self._segment(df_failed)
            else:
                logger.info("No FAILED SEGMENTATION rows detected from `grouped` table. Proceeding...")

            if self._should_stop("segmentation"):
                return

            # Step 7: Attribution (assign speaker/role, clean text, score confidence)
            print("\n[STEP 7] Speaker Attribution")
            df = self.db.get_row_by_status(RowStatus.SEGMENTED, "grouped")
            if len(df) > 0:
                logger.info("Attributing SEGMENTED rows...")
                self._attribute(df)
            else:
                logger.info("No SEGMENTED rows detected from `grouped` table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_A, "grouped")
            if len(df_failed) > 0:
                logger.info("Attributing FAILED ATTRIBUTION rows...")
                self._attribute(df_failed)
            else:
                logger.info("No FAILED ATTRIBUTION rows detected from `grouped` table. Proceeding...")

            if self._should_stop("attribution"):
                return

            # Step 8: Verification
            print("\n[STEP 8] Verification/Correction")
            rows = self.db.get_low_confidence_dialogues(VERIFICATION_CONFIDENCE_THRESHOLD)
            if rows:
                logger.info(f"Verifying {len(rows)} low-confidence dialogue(s)...")
                self._verify(rows)
            else:
                logger.info("No low-confidence dialogues to verify. Skipping.")

            # Step 9: Boundary Stitching
            print("\n[STEP 9] Cross-Group Boundary Stitching")
            self._stitch_boundaries()

            # Step 10: Export
            print("\n[STEP 10] Exporting dialogues to Excel")
            self._export()

            # Pipeline cost summary
            self._log_pipeline_cost_summary()

        except Exception as e:
            logger.error(f"Diarization Pipeline failed: {e}", exc_info=True)
            raise

    def _ingest_csv(self):
        """Load Raw Critical Mention Extract"""
        df = CSVLoader.load(self.file)
        if self.limit:
            df = df.head(self.limit)
            logger.info(f"Limited to {self.limit} rows")
        return df
    
    def _group(self, input):
        """Group continuous transcripts"""
        GroupTranscript(self.db, input).group()

    def _punctuate(self, df):
        """Provides proper punctuation to raw transcripts"""
        agent = PunctuationRestoration(self.db, df)
        agent.punctuate()
        self._accumulate_costs("punctuation", agent)

    def _count(self, df):
        """Classifies group as either OVER or UNDER depending on Transcript Size"""
        agent = TokenCounter(self.db, df)
        agent.count()
        self._accumulate_costs("count", agent)

    def _filter_relevance(self, df):
        """Flags non-news groups as NOT_RELEVANT so they are excluded from later steps"""
        agent = RelevanceAgent(self.db, df)
        agent.filter()
        self._accumulate_costs("relevance", agent)

    def _segment(self, df):
        """Splits each group into speaker-change segments (no attribution), routed by the
        Step 5 size classification: UNDER (or unclassified) rows use the standard
        text-echoing segmenter; OVER rows use the indexed segmenter, whose output does not
        scale with transcript length."""
        rows = df if isinstance(df, list) else df.to_dict("records")
        under = [r for r in rows if r.get("count") != COUNT_OVER]
        over = [r for r in rows if r.get("count") == COUNT_OVER]

        if under:
            logger.info(f"Segmenting {len(under)} UNDER group(s) with the standard agent...")
            agent = SegmentationAgent(self.db, under)
            agent.segment()
            self._accumulate_costs("segmentation", agent)

        if over:
            logger.info(f"Segmenting {len(over)} OVER group(s) with the indexed agent...")
            agent = IndexedSegmentationAgent(self.db, over)
            agent.segment()
            self._accumulate_costs("segmentation_indexed", agent)

    def _attribute(self, df):
        """Attributes each group's segments to speakers, cleans text, scores confidence."""
        agent = AttributionAgent(self.db, df)
        agent.attribute()
        self._accumulate_costs("attribution", agent)
    
    def _verify(self, rows):
        """Verifies/corrects low-confidence diarized dialogues"""
        agent = VerificationAgent(self.db, rows)
        agent.verify()
        self._accumulate_costs("verification", agent)

    def _stitch_boundaries(self):
        """Merge dialogues that straddle consecutive group boundaries when the speaker is the same.

        Iterates groups in chronological order per (channel, date). For each pair of adjacent
        groups, if the last dialogue of group N and the first dialogue of group N+1 share the
        same speaker (and neither is UNKNOWN), the latter is appended to the former and deleted.
        """
        from itertools import groupby

        groups = self.db.get_ordered_groups()
        merged = 0

        key = lambda g: (g["broadcast_date"], g["channel_name"])
        for _, channel_groups in groupby(groups, key=key):
            channel_groups = list(channel_groups)
            for i in range(len(channel_groups) - 1):
                group_a = channel_groups[i]
                group_b = channel_groups[i + 1]

                last_of_a = self.db.get_boundary_dialogue(group_a["group_id"], "last")
                first_of_b = self.db.get_boundary_dialogue(group_b["group_id"], "first")

                if not last_of_a or not first_of_b:
                    continue

                speaker_a = (last_of_a.get("speaker") or "").strip()
                speaker_b = (first_of_b.get("speaker") or "").strip()

                if not speaker_a or not speaker_b:
                    continue
                if speaker_a.upper() == "UNKNOWN" or speaker_b.upper() == "UNKNOWN":
                    continue
                if speaker_a.lower() != speaker_b.lower():
                    continue

                merged_text = last_of_a["dialogue"].rstrip() + " " + first_of_b["dialogue"].lstrip()
                self.db.merge_boundary_dialogues(
                    last_of_a["dialogue_id"],
                    first_of_b["dialogue_id"],
                    merged_text,
                )
                merged += 1
                logger.info(
                    f"Boundary stitched: [{group_a['broadcast_time']}] -> [{group_b['broadcast_time']}] "
                    f"speaker='{speaker_a}'"
                )

        logger.info(f"Boundary stitching complete: {merged} merge(s) applied.")

    def _export(self):
        """Build the windowed final table and export it to Excel."""
        self.db.build_final_table()
        rows = self.db.get_final_export()
        if not rows:
            logger.info("No windows to export. Skipping.")
            return

        os.makedirs("data/output", exist_ok=True)
        suffix = f"_limit_{self.limit}" if self.limit else ""
        out_path = f"data/output/{self.filename}{suffix}_dialogues.xlsx"

        df = pd.DataFrame(rows)
        df.to_excel(out_path, index=False)
        logger.info(f"Exported {len(df)} window(s) to {out_path}")
        print(f"  Saved: {out_path}")

    def _log_pipeline_cost_summary(self):
        """Print a complete cost breakdown for the entire pipeline run."""
        total_input_cost = self.total_input_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
        total_output_cost = self.total_output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
        total_cost = total_input_cost + total_output_cost

        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COST BREAKDOWN (ALL STAGES)")
        logger.info("=" * 60)

        for stage_name, (in_tokens, out_tokens) in sorted(self.stage_costs.items()):
            stage_in_cost = in_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
            stage_out_cost = out_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
            stage_total = stage_in_cost + stage_out_cost
            logger.info(f"{stage_name.upper()}:")
            logger.info(f"  Input tokens:  {in_tokens:,}")
            logger.info(f"  Output tokens: {out_tokens:,}")
            logger.info(f"  Cost:          ${stage_total:.6f}")

        logger.info("-" * 60)
        logger.info("TOTALS:")
        logger.info(f"  Total input tokens:  {self.total_input_tokens:,}")
        logger.info(f"  Total output tokens: {self.total_output_tokens:,}")
        logger.info(f"  Input cost:          ${total_input_cost:.6f}")
        logger.info(f"  Output cost:         ${total_output_cost:.6f}")
        logger.info(f"  TOTAL COST:          ${total_cost:.6f}")
        logger.info("=" * 60)