from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferforge.embedded.sdk import MISSING_JS, SDK_JS  # noqa: E402

TARGET = ROOT / "lib" / "sdk" / "template.ts"
ASSET = ROOT / "public" / "sdk" / "inferforge-sdk.js"

HEADER = "export const SDK_JS = String.raw`"
MIDDLE = "`\n\nexport const MISSING_JS = String.raw`"
FOOTER = "`\n"


def guard(name: str, body: str) -> None:
    for token in ("`", "${"):
        if token in body:
            raise SystemExit(f"{name} contains {token!r}; the SDK template must avoid JS template literals.")


def main() -> int:
    guard("SDK_JS", SDK_JS)
    guard("MISSING_JS", MISSING_JS)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(HEADER + SDK_JS + MIDDLE + MISSING_JS + FOOTER, encoding="utf-8", newline="\n")

    ASSET.parent.mkdir(parents=True, exist_ok=True)
    ASSET.write_text(
        SDK_JS.replace("{ENDPOINT}", "")
        .replace("{FALLBACK}", "")
        .replace("{MODEL}", "")
        .replace("{API_KEY}", ""),
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {TARGET.relative_to(ROOT)}")
    print(f"wrote {ASSET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
