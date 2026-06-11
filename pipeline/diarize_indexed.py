import re
import json

from google.genai import types

from utils.logger import get_logger
from utils.rate_limit import SafetyBlockedError
from config.constants import RowStatus

from .diarize import DiarizationAgent

logger = get_logger(__name__)

# Sentence-level split: after ./!/? followed by whitespace.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# API-enforced output shape: the model can only return unit ranges, never echoed text.
_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segments": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "start": types.Schema(type=types.Type.INTEGER),
                    "end": types.Schema(type=types.Type.INTEGER),
                    "speaker": types.Schema(type=types.Type.STRING),
                    "confidence": types.Schema(type=types.Type.NUMBER),
                },
                required=["start", "end", "speaker", "confidence"],
            ),
        ),
    },
    required=["segments"],
)


class IndexedDiarizationAgent(DiarizationAgent):
    """Diarization for OVER-sized transcripts (grouped."count" = 'OVER').

    The standard diarizer asks the model to re-emit every segment's text verbatim, so its
    output scales with transcript length and long rows hit the generation ceiling (truncated
    or failed JSON). This agent instead pre-splits the transcript into numbered sentence-level
    units, shows the model the WHOLE numbered transcript (full global context, no chunking),
    and has it return unit index RANGES per speaker turn. Segment text is reconstructed from
    the ranges locally, so:
      - output size is O(number of turns), independent of transcript length;
      - verbatim fidelity is guaranteed (the model never rewrites text);
      - full coverage is verifiable — ranges must tile units 1..N exactly, and violations
        are repaired or rejected rather than silently corrupting the dialogues table.

    Attribution rubric, persistence, vocative backstop, and cost accounting are inherited
    from DiarizationAgent.
    """

    PROMPT_PATH = "prompts/group_diarization_indexed_prompt.yaml"
    STAGE_LABEL = "INDEXED DIARIZATION (OVER rows)"

    def _config_kwargs(self) -> dict:
        kwargs = super()._config_kwargs()
        kwargs["response_schema"] = _RESPONSE_SCHEMA
        return kwargs

    async def _process_group(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            people = self.db.get_person_bio_by_group_id(group_id)
            units = self._split_into_units(row.get("plain_text") or "")
            user_prompt = self.user_template.format(
                people_data=self._build_people_data(people),
                unit_count=len(units),
                numbered_transcript=self._build_numbered_block(units),
            )

            try:
                text, in_tokens, out_tokens, cached_tokens = await self._call_llm_with_retry(user_prompt)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.total_cached_tokens += cached_tokens

                ranges = self._parse_ranges(text)
                ranges = self._validate_and_repair(ranges, len(units), group_id)
                segments = self._reconstruct_segments(ranges, units)
                self._persist_segments(row, segments, people)
                self.segments_total += len(segments)

                self.db.update_status("grouped", group_id, RowStatus.DIARIZED)
                self.processed += 1
            except SafetyBlockedError as e:
                # Inherited degradation: single UNKNOWN turn + block flag (terminal).
                self._handle_safety_block(row, e)
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_D)
                logger.warning(f"Failed to diarize (indexed) group_id={group_id}: {e}")

    # ------------------------------------------------------------------ #
    # Unit splitting / numbering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_into_units(text: str) -> list:
        """Split a transcript into sentence-level units, preserving order and content.

        Splits on ' >> ' speaker markers first so each marker starts a fresh unit (the
        '>> ' prefix is kept — it is a diarization cue), then on sentence boundaries
        within each part. Reconstruction joins units with single spaces, so units are
        stored whitespace-stripped.
        """
        units = []
        for i, part in enumerate(text.split(" >> ")):
            part = part.strip()
            if not part:
                continue
            if i > 0 and not part.startswith(">>"):
                part = ">> " + part
            for sentence in _SENTENCE_SPLIT.split(part):
                sentence = sentence.strip()
                if sentence:
                    units.append(sentence)
        return units or [text.strip()]

    @staticmethod
    def _build_numbered_block(units: list) -> str:
        return "\n".join(f"[{i}] {unit}" for i, unit in enumerate(units, start=1))

    # ------------------------------------------------------------------ #
    # Range parsing / coverage validation / reconstruction
    # ------------------------------------------------------------------ #
    def _parse_ranges(self, text) -> list:
        """Parse Gemini JSON → list of raw range dicts. Raises on unusable output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON from indexed diarization response: {e}")
        segments = data.get("segments", []) or []
        if not segments:
            raise ValueError("Indexed diarization response contained no segments")
        return segments

    @staticmethod
    def _validate_and_repair(ranges, n_units: int, group_id: str) -> list:
        """Enforce the coverage contract: segments must tile units 1..n_units exactly.

        Minor violations are repaired deterministically (clamping out-of-range indices,
        extending a segment backwards over a gap, trimming overlaps, extending the last
        segment to N). Every repair is logged. If nothing usable remains, raises so the
        group is retried/failed rather than persisted incomplete.
        """
        cleaned = []
        for r in ranges:
            try:
                start, end = int(r.get("start")), int(r.get("end"))
            except (TypeError, ValueError):
                logger.warning(f"group_id={group_id}: dropping segment with non-integer "
                               f"range: {r!r}")
                continue
            if end < start:
                start, end = end, start
            start = max(1, min(start, n_units))
            end = max(1, min(end, n_units))
            cleaned.append({
                "start": start,
                "end": end,
                "speaker": r.get("speaker"),
                "confidence": r.get("confidence"),
            })

        if not cleaned:
            raise ValueError("No valid segment ranges in indexed diarization response")

        cleaned.sort(key=lambda s: (s["start"], s["end"]))

        repaired = []
        cursor = 1  # next unit number that still needs an owner
        for seg in cleaned:
            if seg["end"] < cursor:
                logger.warning(f"group_id={group_id}: dropping fully-overlapped segment "
                               f"[{seg['start']}-{seg['end']}]")
                continue
            if seg["start"] != cursor:
                logger.warning(f"group_id={group_id}: repairing segment start "
                               f"{seg['start']} -> {cursor} (gap/overlap)")
                seg["start"] = cursor
            repaired.append(seg)
            cursor = seg["end"] + 1

        if cursor <= n_units:
            logger.warning(f"group_id={group_id}: extending last segment end "
                           f"{repaired[-1]['end']} -> {n_units} (trailing gap)")
            repaired[-1]["end"] = n_units

        return repaired

    @staticmethod
    def _reconstruct_segments(ranges, units) -> list:
        """Materialize each range into the segment shape _persist_segments expects."""
        segments = []
        for seg in ranges:
            text = " ".join(units[seg["start"] - 1:seg["end"]])
            # Strip the leading '>>' diarization marker — it is not spoken dialogue.
            text = re.sub(r'^>>\s*', '', text)
            segments.append({
                "speaker": seg.get("speaker"),
                "text": text,
                "confidence": seg.get("confidence"),
            })
        return segments
