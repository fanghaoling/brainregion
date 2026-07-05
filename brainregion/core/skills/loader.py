"""skill YAML loader(skill-centric,声明其 region;modeled on core/regions/loader.py)。

**加载期 fail-fast 校验**(review #3,带文件名 + skill id 报错):必填字段非空;``kind``∈枚举;
``id`` 全局唯一(撞 id raise);``region``∈ regions registry;``kind=provider`` 的 ``ref``∈ ProviderRegistry
(故 bootstrap **先 ProviderRegistry 后 skill YAML**);未知字段严格报错。

region/provider 存在性校验经 **callback** 注入(``region_exists``/``provider_exists``)——保持 core/skills
不依赖 core.regions / ProviderRegistry(解耦;eval/测试可注入假 callback)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from .manifest import SkillManifest

SKILLS_DIR = Path(__file__).resolve().parent

_REQUIRED: tuple[str, ...] = ("id", "name", "region", "kind")
_KNOWN_FIELDS: frozenset[str] = frozenset(
    {"id", "name", "region", "kind", "description", "tags", "version", "ref", "status", "metadata"}
)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    vals = value if isinstance(value, list) else [value]
    out: list[str] = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return tuple(out)


def _manifest_from_dict(
    data: dict, path: Path, *,
    region_exists: Callable[[str], bool] | None,
    provider_exists: Callable[[str], bool] | None,
    known_ids: set[str] | None,
) -> SkillManifest:
    from .manifest import _VALID_KINDS, _VALID_STATUS

    sid = str(data.get("id") or path.stem).strip()

    def _err(msg: str) -> None:
        raise ValueError(f"skill {sid!r} ({path.name}): {msg}")

    for k in _REQUIRED:
        if not str(data.get(k) or "").strip():
            _err(f"missing/empty required field {k!r}")
    if known_ids is not None and sid in known_ids:
        _err(f"duplicate skill id {sid!r}")
    kind = str(data.get("kind")).strip()
    if kind not in _VALID_KINDS:
        _err(f"kind {kind!r} ∉ {list(_VALID_KINDS)}")
    region = str(data.get("region")).strip()
    if region_exists is not None and not region_exists(region):
        _err(f"region {region!r} not in regions registry")
    ref = str(data.get("ref") or "").strip()
    if kind == "provider":
        if not ref:
            _err("kind=provider requires non-empty ref (provider_name)")
        if provider_exists is not None and not provider_exists(ref):
            _err(f"provider {ref!r} not registered (kind=provider ref)")
    status = str(data.get("status") or "experimental").strip() or "experimental"
    if status not in _VALID_STATUS:
        _err(f"status {status!r} ∉ {list(_VALID_STATUS)}")
    extra = set(data.keys()) - _KNOWN_FIELDS
    if extra:
        _err(f"unknown field(s): {sorted(extra)}")
    return SkillManifest(
        id=sid,
        name=str(data.get("name") or sid).strip(),
        region=region,
        kind=kind,                                                   # type: ignore[arg-type]
        description=str(data.get("description") or "").strip(),
        tags=_as_tuple(data.get("tags")),
        version=dict(data.get("version") or {}),
        ref=ref,
        status=status,                                               # type: ignore[arg-type]
        metadata=dict(data.get("metadata") or {}),
    )


def load_skill(
    name: str, skills_dir: str | Path = SKILLS_DIR, *,
    region_exists: Callable[[str], bool] | None = None,
    provider_exists: Callable[[str], bool] | None = None,
    known_ids: set[str] | None = None,
) -> SkillManifest:
    """Load one skill manifest by id/name (fail-fast 校验)。"""
    d = Path(skills_dir)
    path = d / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown skill: {name!r}; available: {list_skills(d)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"skill YAML must be an object: {path}")
    return _manifest_from_dict(
        data, path, region_exists=region_exists,
        provider_exists=provider_exists, known_ids=known_ids,
    )


def list_skills(skills_dir: str | Path = SKILLS_DIR) -> list[str]:
    """List available skill ids (stems of *.yaml)。"""
    d = Path(skills_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_skills(
    skills_dir: str | Path = SKILLS_DIR, *,
    region_exists: Callable[[str], bool] | None = None,
    provider_exists: Callable[[str], bool] | None = None,
) -> list[SkillManifest]:
    """Load all skill manifests(dir 内每个 *.yaml),跨文件 id 唯一性校验。"""
    out: list[SkillManifest] = []
    known: set[str] = set()
    for name in list_skills(skills_dir):
        m = load_skill(
            name, skills_dir, region_exists=region_exists,
            provider_exists=provider_exists, known_ids=known,
        )
        known.add(m.id)
        out.append(m)
    return out
