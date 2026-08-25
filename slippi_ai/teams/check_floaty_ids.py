# Character id sanity for curriculum floaty set (melee enums).
from __future__ import annotations

import sys


def main() -> int:
    try:
        import melee
    except ImportError:
        print("melee not installed — skip char-id check")
        return 0

    from slippi_ai.teams.curriculum import FLOATY_OR_SLOW_CHARS

    wanted = {
        melee.Character.JIGGLYPUFF: "JIGGLYPUFF",
        melee.Character.PEACH: "PEACH",
        melee.Character.LUIGI: "LUIGI",
        melee.Character.SHEIK: "SHEIK",
        melee.Character.POPO: "POPO",
        melee.Character.NANA: "NANA",
        melee.Character.ZELDA: "ZELDA",
        melee.Character.DOC: "DOC",
        melee.Character.MARIO: "MARIO",
        melee.Character.PIKACHU: "PIKACHU",
        melee.Character.MEWTWO: "MEWTWO",
        melee.Character.YOSHI: "YOSHI",
    }
    bad = []
    for ch, name in wanted.items():
        if int(ch.value) not in FLOATY_OR_SLOW_CHARS:
            bad.append(f"{name}={ch.value} missing from FLOATY_OR_SLOW_CHARS")
    if bad:
        print("FAIL:")
        for b in bad:
            print(" ", b)
        return 1
    print("floaty char ids OK:", {n: int(c.value) for c, n in wanted.items()})
    return 0


if __name__ == "__main__":
    sys.path.insert(0, r"C:\Users\langu\slippi-ai")
    raise SystemExit(main())
