"""Central configuration for the Employee Policy Query Agent."""
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = LOG_DIR / "cache"
for _d in (DATA_DIR, CHROMA_DIR, OUTPUT_DIR, REPORTS_DIR, LOG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

AUDIT_LOG = LOG_DIR / "audit_log.csv"
GENAI_LOG = LOG_DIR / "genai_log.csv"
ESCALATION_LOG = LOG_DIR / "escalation_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback_log.csv"

COLLECTION = "hr_policy_kb"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.1

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MMR_FETCH_K = 10
RETRIEVER_K = 4
MEMORY_WINDOW_K = 5
CONFIDENCE_THRESHOLD = 0.6

IDK_TOKEN = "I_DONT_KNOW"

# Legal / sensitive: NO AI answer, hand off to HR immediately.
SENSITIVE_KEYWORDS = [
    "harassment", "harass", "lawyer", "legal action", "sue ", "lawsuit",
    "discrimination", "discriminate", "posh", "assault", "abuse", "bullying",
    "retaliation", "court",
]
# Soft triggers: AI may answer from policy AND offers HR escalation.
SOFT_KEYWORDS = ["resign", "resignation", "terminated", "fired", "grievance"]
HR_REQUEST_PHRASES = [
    "talk to hr", "speak to hr", "speak with hr", "contact hr", "hr representative",
    "human agent", "talk to a human", "talk to someone", "connect me to hr",
]

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"reveal (your|the) (system )?prompt",
    r"you are now (?!an hr)",
    r"act as (?!an hr)",
]

DISCLAIMER = "AI answers are for informational purposes only. For legal or sensitive matters, contact HR."


def _h(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()


# Demo credentials (change before any real deployment).
USERS = {
    "employee": {"pw": _h("employee123"), "role": "Employee"},
    "hr": {"pw": _h("hr123"), "role": "HR Rep"},
    "admin": {"pw": _h("admin123"), "role": "Admin"},
}


def check_login(username: str, password: str):
    u = USERS.get((username or "").strip().lower())
    if u and u["pw"] == _h(password or ""):
        return u["role"]
    return None
