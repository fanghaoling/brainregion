"""Zero-model-call smoke test for the optional ARC-AGI-3 adapter."""
from __future__ import annotations

import argparse
import json
from typing import Any

from .envs.arc_agi import ArcAgiEnv


def _summary(env: ArcAgiEnv) -> dict[str, Any]:
    snapshot = env.snapshot()
    frame = snapshot.get("frame") or []
    first_row = frame[0] if frame else []
    return {
        "game_id": snapshot.get("game_id"),
        "state": snapshot.get("state"),
        "levels_completed": snapshot.get("levels_completed"),
        "win_levels": snapshot.get("win_levels"),
        "available_actions": snapshot.get("available_actions"),
        "frame_encoding": snapshot.get("frame_encoding"),
        "frame_height": len(frame),
        "frame_width": len(first_row),
        "palette": snapshot.get("palette"),
        "solved": env.solved,
        "total_reward": env.total_reward,
    }


def run_smoke(
    *,
    game_id: str = "ls20",
    seed: int = 0,
    actions: list[str] | None = None,
) -> dict[str, Any]:
    env = ArcAgiEnv.create(game_id, seed=seed)
    steps: list[dict[str, Any]] = []
    try:
        for action in actions or []:
            _observation, reward, terminated, info = env.step(action)
            steps.append(
                {
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "info": info,
                }
            )
            if terminated:
                break
        return {"adapter": "arc_agi", "sdk_calls_model": False, **_summary(env), "steps": steps}
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action",
        action="append",
        default=[],
        help="Optional simple action name; repeat to execute a bounded sequence",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            run_smoke(game_id=args.game, seed=args.seed, actions=args.action),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
