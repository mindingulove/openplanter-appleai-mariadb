from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


VALID_REASONING_EFFORTS: set[str] = {"low", "medium", "high"}


def normalize_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid reasoning effort '{value}'. Expected one of: "
            f"{', '.join(sorted(VALID_REASONING_EFFORTS))}"
        )
    return cleaned


@dataclass(slots=True)
class PersistentSettings:
    default_model: str | None = None
    default_provider: str | None = None
    default_reasoning_effort: str | None = None
    default_model_openai: str | None = None
    default_model_anthropic: str | None = None
    default_model_openrouter: str | None = None
    default_model_cerebras: str | None = None
    default_model_apple: str | None = None
    openai_base_url: str | None = None
    anthropic_base_url: str | None = None
    openrouter_base_url: str | None = None
    cerebras_base_url: str | None = None
    apple_base_url: str | None = None
    mariadb_host: str | None = None
    mariadb_port: str | None = None
    mariadb_user: str | None = None
    mariadb_password: str | None = None
    mariadb_database: str | None = None
    use_mariadb: str | None = None

    def default_model_for_provider(self, provider: str) -> str | None:
        per_provider = {
            "openai": self.default_model_openai,
            "anthropic": self.default_model_anthropic,
            "openrouter": self.default_model_openrouter,
            "cerebras": self.default_model_cerebras,
            "apple": self.default_model_apple,
        }
        specific = per_provider.get(provider)
        if specific:
            return specific
        return self.default_model or None

    def normalized(self) -> "PersistentSettings":
        # Extract all fields dynamically from slots
        args = {}
        for field_name in self.__slots__:
            value = getattr(self, field_name)
            if isinstance(value, str):
                value = value.strip() or None
            args[field_name] = value
            
        # Specific normalization for reasoning effort
        if args.get("default_reasoning_effort"):
            try:
                args["default_reasoning_effort"] = normalize_reasoning_effort(args["default_reasoning_effort"])
            except ValueError:
                args["default_reasoning_effort"] = None
                
        return PersistentSettings(**args)

    def to_json(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        for field_name in self.__slots__:
            value = getattr(self, field_name)
            if value:
                payload[field_name] = str(value)
        return payload

    @classmethod
    def from_json(cls, payload: dict | None) -> "PersistentSettings":
        if not isinstance(payload, dict):
            return cls()
        
        args = {}
        for field_name in cls.__slots__:
            val = payload.get(field_name)
            args[field_name] = (str(val).strip() if val is not None else None) or None
            
        return cls(**args).normalized()


@dataclass(slots=True)
class SettingsStore:
    workspace: Path
    session_root_dir: str = ".openplanter"
    settings_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        root = self.workspace / self.session_root_dir
        root.mkdir(parents=True, exist_ok=True)
        self.settings_path = root / "settings.json"

    def load(self) -> PersistentSettings:
        if not self.settings_path.exists():
            return PersistentSettings()
        try:
            parsed = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return PersistentSettings()
        return PersistentSettings.from_json(parsed)

    def save(self, settings: PersistentSettings) -> None:
        normalized = settings.normalized()
        self.settings_path.write_text(
            json.dumps(normalized.to_json(), indent=2),
            encoding="utf-8",
        )
