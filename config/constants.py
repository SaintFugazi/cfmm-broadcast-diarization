class RowStatus:
    """URL processing status constants"""
    PENDING = 'PENDING'
    PUNCTUATED = 'PUNCTUATED'
    RELEVANT = 'RELEVANT'
    NOT_RELEVANT = 'NOT RELEVANT'
    NAMED = 'NAMED'
    B_CLASSIFIED = 'B_CLASSIFIED'
    G_CLASSIFIED = 'G_CLASSIFIED'
    DIARIZED = 'DIARIZED'
    FAILED_P = 'FAILED PUNCTUATION'
    FAILED_C = 'FAILED COUNT'
    FAILED_R = 'FAILED RELEVANCE'
    FAILED_N = 'FAILED NAME EXTRACTION'
    FAILED_D = 'FAILED DIARIZATION'


class BlockStatus:
    """grouped.block_status / dialogues.block_status markers.

    A non-NULL block_status records that a stage could NOT use the LLM for this row
    because Gemini's non-configurable core safety filter (e.g. PROHIBITED_CONTENT)
    rejected the content. Unlike a FAILED_* status, a blocked row is *terminal*: it
    advances on its normal forward status (so reruns skip it and never re-incur the
    deterministic block), with this column flagging it as degraded for audit / requeue.
    Transient errors (429, timeout) still use FAILED_* and remain retryable.
    """
    PUNCTUATION = 'BLOCKED_PUNCTUATION'      # local punctuation fallback was used
    RELEVANCE = 'BLOCKED_RELEVANCE'          # kept RELEVANT without an LLM verdict
    NAMES = 'BLOCKED_NAME_EXTRACTION'        # persons kept, role/context left NULL
    DIARIZATION = 'BLOCKED_DIARIZATION'      # emitted a single UNKNOWN-speaker turn
    VERIFICATION = 'BLOCKED_VERIFICATION'    # left unverified, excluded from re-check


class PunctuationSource:
    """grouped.punctuation_source: which engine restored a row's punctuation."""
    LLM = 'llm'        # normal Gemini punctuation
    LOCAL = 'local'    # local model fallback after a safety block
    RAW = 'raw'        # neither worked; original unpunctuated text kept

# Local punctuation fallback (used only when the Gemini call is safety-blocked).
LOCAL_PUNCTUATION_MODEL = "kredor/punctuate-all"   # multilingual; backs deepmultilingualpunctuation


# Transcript size classification (token counting) settings.
# The standard diarizer re-emits the whole transcript in its output, so transcripts whose
# token count would push the OUTPUT near the model's generation ceiling must go through the
# indexed diarizer instead (which outputs unit ranges, not text). 3000 input tokens ≈ 3500+
# output tokens after JSON overhead — comfortably under the ~8k output ceiling, with margin.
COUNT_OVER = 'OVER'                  # grouped."count" value: too large for the standard diarizer
COUNT_UNDER = 'UNDER'                # grouped."count" value: safe for the standard diarizer
DIARIZATION_TOKEN_THRESHOLD = 1000   # plain_text token count above which a group is OVER
COUNT_CONCURRENCY = 10               # max simultaneous in-flight count_tokens requests
COUNT_MAX_RETRIES = 5                # retry attempts per row on failure
COUNT_BACKOFF_BASE = 2               # exponential backoff base (seconds)
COUNT_BACKOFF_CAP = 60               # max backoff sleep (seconds)

# Punctuation Restoration Agent settings
PUNCTUATION_CONCURRENCY = 5       # max simultaneous in-flight LLM requests
PUNCTUATION_MAX_RETRIES = 5       # retry attempts per row on failure
PUNCTUATION_BACKOFF_BASE = 2      # exponential backoff base (seconds)
PUNCTUATION_BACKOFF_CAP = 60      # max backoff sleep (seconds)

# Relevance Filter (news vs. non-news) LLM agent settings
RELEVANCE_CONCURRENCY = 5            # max simultaneous in-flight LLM requests
RELEVANCE_MAX_RETRIES = 5           # retry attempts per chunk on failure
RELEVANCE_BACKOFF_BASE = 2          # exponential backoff base (seconds)
RELEVANCE_BACKOFF_CAP = 60          # max backoff sleep (seconds)
RELEVANCE_CHUNK_SIZE = 10           # groups bundled per LLM call
RELEVANCE_DELETE_THRESHOLD = 0.8    # min confidence to mark a group NOT_RELEVANT
RELEVANCE_CACHE_TTL = "1800s"       # cache lifetime spanning the relevance run

# Allowlist of keywords that ALWAYS force a group to be kept as RELEVANT, regardless of its
# surface topic. Used both by the deterministic pre-scan in RelevanceAgent AND injected into the
# relevance prompt (single source of truth). Matched case-insensitively on word boundaries.
RELEVANCE_KEYWORDS = [
    "Islam", "Islamic", "Islamophobic", "Islamophobia", "Islamist", "Islamists", "Islamism",
    "Muslim", "Muslims", "Moslem",
    "Jihad", "Jihadist", "Jihadists", "Jihadi", "Jihadis", "Jihadism",
    "Mujaheddin", "Mujahedin",
    "Koran", "Quran", "Qur'an",
    "Sharia", "Shariah", "Shari'a",
    "Headscarf", "Niqab", "Burka", "Burkas", "Burqa", "Hijab",
    "Shia", "Shia's", "Shiite", "Shi'ite", "Sunni", "Sunni's",
    "Hadith", "Hadiths",
    "Prophet Muhammad", "Prophet Mohammed",
    "Mosque", "Mosques", "Masjid", "Madrasa", "Madrassa", "Madrassas",
    "Halal", "Allah", "Allah hu Akbar",
    "Eid", "Ramadan", "Ramadhan",
    "Mecca", "Makkah", "Medina", "Madina",
    "Mullah", "Imam", "Imams", "Mufti", "Caliphate",
    "Hajj", "Umrah", "Umra",
    "Ayatollah", "Shaykh", "Fatwa",
    "Mohammedan", "Wahabi", "Wahhabi", "Salafi", "Salafist",
]

# Global Gemini rate-limit strategy (shared by ALL agents via utils/rate_limit.py).
# Tune GEMINI_RPM_LIMIT to your API tier: this is the single lever that paces every agent.
# 10 is conservative and safe even on the free tier — raise it once you know your quota.
GEMINI_RPM_LIMIT = 150          # requests/min budget enforced across each agent's concurrent tasks
RATE_LIMIT_MIN_BACKOFF = 60    # floor (s) for a 429 backoff, so the per-minute quota window resets
RATE_LIMIT_MAX_RETRIES = 8     # retries reserved specifically for rate-limit (429) errors
REQUEST_TIMEOUT = 180          # seconds before a single hung LLM call is cancelled and retried
PUNCTUATION_MAX_CHUNK_CHARS = 8000  # transcripts longer than this are split before punctuating

# Gemini pricing (USD per 1M tokens) — VERIFY against current pricing for your model
GEMINI_INPUT_COST_PER_1M = 0.25
GEMINI_OUTPUT_COST_PER_1M = 1.50

# Context caching: cached input tokens are billed at a fraction of the normal input rate.
# The large, fixed system instruction is cached once per agent run and reused across every
# per-group call, so it isn't re-sent (and re-charged at full rate) on each request.
GEMINI_CACHE_INPUT_DISCOUNT = 0.25   # cached input billed at ~25% of GEMINI_INPUT_COST_PER_1M
DIARIZATION_CACHE_TTL = "1800s"      # cache lifetime spanning the diarization run
VERIFICATION_CACHE_TTL = "1800s"     # cache lifetime spanning the verification run

# NER (person-name extraction) settings
NER_MODEL_NAME = "dslim/bert-base-NER"
NER_BATCH_SIZE = 32
NER_MAX_LENGTH = 512
NER_DEVICE = None                  # None → auto: cuda if available else cpu

# Who/Why (role + context) LLM agent settings
NAME_EXTRACTION_CONCURRENCY = 5    # max simultaneous in-flight LLM requests
NAME_EXTRACTION_MAX_RETRIES = 5    # retry attempts per chunk on failure
NAME_EXTRACTION_BACKOFF_BASE = 2   # exponential backoff base (seconds)
NAME_EXTRACTION_BACKOFF_CAP = 60   # max backoff sleep (seconds)
WHO_AND_WHY_CHUNK_SIZE = 20        # person-entries bundled per LLM request

# BERT segment classification (change-of-speaker) settings
BERT_MODEL_NAME = "microsoft/deberta-v3-base"       # base architecture for the checkpoint
BERT_CHECKPOINT_PATH = "model/best_deberta_checkpoint.pt"
BERT_NUM_LABELS = 2
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 32
BERT_DEVICE = None                                  # None → auto: cuda if available else cpu

# Group-level diarization (whole-group LLM agent) settings
DIARIZATION_CONCURRENCY = 5            # max simultaneous in-flight LLM requests
DIARIZATION_MAX_RETRIES = 5            # retry attempts per group on failure
DIARIZATION_BACKOFF_BASE = 2           # exponential backoff base (seconds)
DIARIZATION_BACKOFF_CAP = 60           # max backoff sleep (seconds)

# Speaker verification (low-confidence dialogue re-check) settings
VERIFICATION_CONFIDENCE_THRESHOLD = 0.9   # dialogues below this (or NULL) get verified
VERIFICATION_CONCURRENCY = 5              # max simultaneous in-flight LLM requests
VERIFICATION_MAX_RETRIES = 5             # retry attempts per dialogue on failure
VERIFICATION_BACKOFF_BASE = 2            # exponential backoff base (seconds)
VERIFICATION_BACKOFF_CAP = 60            # max backoff sleep (seconds)