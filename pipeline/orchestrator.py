import os
import pandas as pd

from .group import GroupTranscript
from .punctuation import PunctuationRestoration
from .relevance import RelevanceAgent
from .name_extraction import NameExtractor
from .diarize import DiarizationAgent
from .diarize_indexed import IndexedDiarizationAgent
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

STEPS = ["group", "punctuation", "relevance", "names", "diarize", "count"]

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

            # Step 3: Punctuation
            print("\n[STEP 3] Punctuation Restoration")
            df = self.db.get_row_by_status(RowStatus.PENDING, "grouped")
            if len(df) > 0:
                logger.info("Punctuating PENDING rows...")
                self._punctuate(df)
            else:
                logger.info("No PENDING rows detected from 'grouped' table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_P, "grouped")
            if len(df_failed) > 0:
                logger.info("Puntuating FAILED rows...")
                self._punctuate(df_failed)
            else:
                logger.info("No FAILED PUNCTUATION rows detected from 'grouped table. Proceeding...")

            if self._should_stop("punctuation"):
                return
            
            # Step 4: Token Count
            print("\n[STEP 4] Transcript Size Classification")
            df = self.db.get_row_by_status(RowStatus.PUNCTUATED, "grouped")
            if len(df) > 0:
                logger.info("Counting tokens for PUNCTUATED rows...")
                self._count(df)
            else:
                logger.info("No PUNCTUATED rows detected from `grouped` table. Proceeding to FAILED rows...")
            
            df_failed = self.db.get_row_by_status(RowStatus.FAILED_C, "grouped")
            if len(df_failed) > 0:
                logger.info("Counting tokens for FAILED COUNT rows...")
                self._count(df_failed)
            else:
                logger.info("No FAILED COUNT rows detected from `grouped` table. Proceeding...")
            
            if self._should_stop("count"):
                return

            # Step 5: Relevance Filter (news vs. non-news)
            print("\n[STEP 5] Relevance Filtering")
            df = self.db.get_row_by_status(RowStatus.PUNCTUATED, "grouped")
            if len(df) > 0:
                logger.info("Filtering PUNCTUATED rows for relevance...")
                self._filter_relevance(df)
            else:
                logger.info("No PUNCTUATED rows detected from 'grouped' table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_R, "grouped")
            if len(df_failed) > 0:
                logger.info("Filtering FAILED RELEVANCE rows...")
                self._filter_relevance(df_failed)
            else:
                logger.info("No FAILED RELEVANCE rows detected from 'grouped' table. Proceeding...")

            if self._should_stop("relevance"):
                return

            # Step 6: Name Extraction
            print("\n[STEP 6] BERT-Driven Name Extraction")
            df = self.db.get_row_by_status(RowStatus.RELEVANT, "grouped")
            if len(df) > 0:
                logger.info("Extracting names from RELEVANT rows...")
                self._name_extract(df)
            else:
                logger.info("No RELEVANT rows detected from 'grouped' table. Proceeding to FAILED rows...")

            df_failed = self.db.get_row_by_status(RowStatus.FAILED_N, "grouped")
            if len(df_failed) > 0:
                logger.info("Extracting names for FAILED rows...")
                self._name_extract(df_failed)
            else:
                logger.info("No FAILED NAME EXTRACTION rows detected from 'grouped' table. Proceeding...")

            if self._should_stop("names"):
                return

            # Step 7: Group Diarization
            print("\n[STEP 7] Diarizing group dialogues")
            df = self.db.get_row_by_status(RowStatus.NAMED, "grouped")
            if len(df) > 0:
                logger.info("Diarizing NAMED rows...")
                self._diarize(df)
            else:
                logger.info("No NAMED rows detected from `grouped` table. Proceeding to FAILED rows...")
            
            df_failed = self.db.get_row_by_status(RowStatus.FAILED_D, "grouped")
            if len(df_failed) > 0:
                logger.info("Diarizing FAILED DIARIZATION rows...")
                self._diarize(df_failed)
            else:
                logger.info("No FAILED DIARIZATION rows detected from `grouped` table. Proceeding...")

            if self._should_stop("diarize"):
                return

            # Step 8: Verification
            print("\n[STEP 8] Verification/Correction")
            rows = self.db.get_low_confidence_dialogues(VERIFICATION_CONFIDENCE_THRESHOLD)
            if rows:
                logger.info(f"Verifying {len(rows)} low-confidence dialogue(s)...")
                self._verify(rows)
            else:
                logger.info("No low-confidence dialogues to verify. Skipping.")

            # Step 8: Export
            print("\n[STEP 8] Exporting dialogues to Excel")
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

    def _name_extract(self, df):
        """Extracts names per row"""
        agent = NameExtractor(self.db, df)
        agent.extract()
        self._accumulate_costs("names", agent)

    def _diarize(self, df):
        """Diarizes group dialogues, routed by the Step 4 size classification:
        UNDER (or unclassified) rows use the standard text-echoing diarizer;
        OVER rows use the indexed diarizer, whose output does not scale with
        transcript length."""
        rows = df if isinstance(df, list) else df.to_dict("records")
        under = [r for r in rows if r.get("count") != COUNT_OVER]
        over = [r for r in rows if r.get("count") == COUNT_OVER]

        if under:
            logger.info(f"Diarizing {len(under)} UNDER group(s) with the standard agent...")
            agent = DiarizationAgent(self.db, under)
            agent.diarize()
            self._accumulate_costs("diarization", agent)

        if over:
            logger.info(f"Diarizing {len(over)} OVER group(s) with the indexed agent...")
            agent = IndexedDiarizationAgent(self.db, over)
            agent.diarize()
            self._accumulate_costs("diarization_indexed", agent)
    
    def _verify(self, rows):
        """Verifies/corrects low-confidence diarized dialogues"""
        agent = VerificationAgent(self.db, rows)
        agent.verify()
        self._accumulate_costs("verification", agent)

    def _export(self):
        """Export the dialogues table joined with grouped metadata to Excel."""
        rows = self.db.get_dialogues_export()
        if not rows:
            logger.info("No dialogues to export. Skipping.")
            return

        os.makedirs("data/output", exist_ok=True)
        suffix = f"_limit_{self.limit}" if self.limit else ""
        out_path = f"data/output/{self.filename}{suffix}_dialogues.xlsx"

        df = pd.DataFrame(rows)
        df.to_excel(out_path, index=False)
        logger.info(f"Exported {len(df)} dialogue rows to {out_path}")
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