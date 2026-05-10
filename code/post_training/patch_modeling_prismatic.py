"""Patch all 4 LIBERO ckpts' modeling_prismatic.py to remove the
_supports_sdpa property that breaks under transformers>=4.50."""
import re
from pathlib import Path

CKPT_ROOT = Path("/ad-alg/planning-users/liuzhi7/ro_planning/models")
CKPTS = [
    "openvla-7b-finetuned-libero-spatial",
    "openvla-7b-finetuned-libero-object",
    "openvla-7b-finetuned-libero-goal",
    "openvla-7b-finetuned-libero-10",
]

# Old block (with multiple variations of indentation/quote style):
PROPERTY_PATTERN = re.compile(
    r"    @property\s*\n"
    r"    def _supports_sdpa\(self\) -> bool:\s*\n"
    r'        """Check LLM supports SDPA Attention"""\s*\n'
    r"        return self\.language_model\._supports_sdpa",
    re.MULTILINE,
)

REPLACEMENT = "    _supports_sdpa = False  # patched for transformers>=4.50"

# Also clear cached pyc
for ckpt in CKPTS:
    f = CKPT_ROOT / ckpt / "modeling_prismatic.py"
    if not f.exists():
        print(f"missing: {f}")
        continue
    src = f.read_text()
    new = PROPERTY_PATTERN.sub(REPLACEMENT, src)
    if new != src:
        f.write_text(new)
        print(f"patched: {f}")
    else:
        # Maybe already patched
        if "_supports_sdpa = False" in src:
            print(f"already patched: {f}")
        else:
            print(f"no match (regex needs update): {f}")

# Also patch the HF cache version
HF_CACHE = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
if HF_CACHE.exists():
    for entry in HF_CACHE.iterdir():
        if not entry.is_dir() or "openvla" not in entry.name.lower():
            continue
        f = entry / "modeling_prismatic.py"
        if not f.exists():
            continue
        src = f.read_text()
        new = PROPERTY_PATTERN.sub(REPLACEMENT, src)
        if new != src:
            f.write_text(new)
            print(f"patched cache: {f}")
        # Also remove .pyc to force re-compile
        pyc_dir = entry / "__pycache__"
        if pyc_dir.exists():
            for pyc in pyc_dir.glob("modeling_prismatic*.pyc"):
                pyc.unlink()
                print(f"removed cached pyc: {pyc.name}")
