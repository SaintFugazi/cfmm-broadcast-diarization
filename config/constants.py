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
    SPLIT = 'SPLIT'
    FAILED_P = 'FAILED PUNCTUATION'
    FAILED_R = 'FAILED RELEVANCE'
    FAILED_N = 'FAILED NAME EXTRACTION'
    FAILED_D = 'FAILED DIARIZATION'


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