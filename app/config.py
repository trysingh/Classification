"""
config.py
---------
Single source of truth for the product: paths, server and job behaviour, engine
choices, Ollama connection, taxonomy limits, DATASET PROFILES and every prompt.

Adapting the engine to a new dataset needs NO code change:
  1. Add a ProfileConfig to DEFAULT_PROFILES below (or supply OPENJEV_PROFILES as JSON).
  2. Pick it in the UI. Prompts, column auto-detection, the review threshold and the
     taxonomy file (data/taxonomies/<profile>.json) are all derived from the profile.

Any scalar can be overridden via `.env` / environment using the OPENJEV_ prefix,
e.g. OPENJEV_CLASSIFIER_BACKEND=keyword.

Prompt placeholders available everywhere: {domain} {item_name} {level1_name} {level2_name}
{others_label}. Extra ones per prompt are listed next to each template.
Literal braces inside a template must be doubled ({{ }}).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
BackendName = Literal["causal_lm", "laya", "keyword", "lexicon_sentiment", "reflex_decision"]


class ProfileConfig(BaseModel):
    """One dataset type. Holds everything that used to be hard-coded for 'ERP narrations in a bank'."""

    label: str                                   # name shown in the UI
    description: str                             # {domain}: what the data is about
    item_name: str = "record"                    # {item_name}: what one row is called
    level1_name: str = "Main category"           # wording for level 1 (UI + prompts)
    level2_name: str = "Sub-category"            # wording for level 2 (UI + prompts)
    taxonomy_hints: str = ""                     # optional examples handed to System 2
    text_column_hints: list[str] = ["description", "narration", "text", "remarks", "details"]
    amount_column_hints: list[str] = []          # optional numeric column -> spend analysis
    context_columns: list[str] = []              # extra columns appended to the classifier context
    use_supplier_context: bool = False           # add "Supplier: X" to the classifier context
    low_confidence_threshold: float = 0.5        # below this a row is flagged "needs review"
    backend: Optional[BackendName] = None        # None = global classifier_backend
    seed_taxonomy: dict[str, list[str]] = {}     # starter taxonomy when System 2 is unavailable
    # lexicon_sentiment only: relative path under /static/ the browser fetches to score locally (None = server-only).
    # negation_window is deliberately NOT here -- it lives once in the lexicon JSON so the Python
    # backend and the browser port can never tune it differently.
    lexicon_asset: Optional[str] = None
    neutral_band: float = 0.15                    # |avg score| at or below this reads as "Neutral"
    # reflex/decision profiles only (backend="reflex_decision" escalation modes, e.g. "dino"):
    # relative path under /static/ to the shared reflex-rules JSON, read identically by
    # ReflexDecisionBackend (server) and the matching JS adapter (client) -- same
    # never-disagree guarantee lexicon_asset gives the sentiment demo, for state instead of text.
    # None = ReflexDecisionBackend falls back to the static/data/<mode>_reflex_rules.json naming
    # convention (what dino_adapter.js's hardcoded fetch path also assumes).
    reflex_asset: Optional[str] = None
    # /decide/<mode> page only (not in the original patch notes -- added so the promoted Jinja
    # template stays mode-generic too, matching "a new mode needs no engine change": which JS
    # files the page includes (under /static/js/, in order) and the global factory function
    # name it calls to build the {reflex, applyAction, actions} adapter. Empty adapter_factory
    # means this profile isn't a playable /decide page.
    game_scripts: list[str] = []
    adapter_factory: str = ""
    # Per-profile overrides (None = inherit the global value)
    max_main_categories: Optional[int] = None
    max_sub_categories: Optional[int] = None
    debias_permutations: Optional[int] = None
    taxonomy_generation_prompt: Optional[str] = None
    main_category_question: Optional[str] = None
    sub_category_question: Optional[str] = None
    dynamic_label_prompt: Optional[str] = None


# ---------------------------------------------------------------------------------------
# Dataset profiles. "erp_costs" reproduces the original tuned prompts exactly; the other two
# show how little is needed to support a different dataset.
# ---------------------------------------------------------------------------------------
DEFAULT_PROFILES: dict[str, ProfileConfig] = {
    "erp_costs": ProfileConfig(
        label="ERP software costs (banking)",
        description="ERP narrations describing software-system costs in a banking enterprise",
        item_name="ERP narration",
        level1_name="Cost category",
        level2_name="Cost sub-category",
        taxonomy_hints="(e.g. infrastructure, network, licensing, operations, governance/compliance, "
                       "consulting/professional services, maintenance/support, and similar groupings "
                       "you infer from the sample)",
        text_column_hints=["narration", "description", "particulars", "remarks", "details", "text"],
        amount_column_hints=["amount", "net amount", "invoice amount", "value", "cost", "debit", "total"],
        use_supplier_context=True,
        main_category_question=(
            "What is the main cost category of this ERP narration for a banking software system? "
            "Choose '{others_label}' only if none of the other categories fit."),
        sub_category_question="Within '{main_category}', what is the specific sub-category of this ERP narration?",
        dynamic_label_prompt=(
            "ERP cost narration: {narration}\n"
            "In 2-4 words, name the most specific cost sub-category this belongs to:\n"
            "Sub-category:"),
        seed_taxonomy={
            "Infrastructure & Hosting": ["Cloud Hosting", "Data Centre", "Storage & Backup",
                                         "Virtualization & Compute", "Disaster Recovery"],
            "Network & Connectivity": ["WAN & Leased Lines", "Network Security", "Telecom & Bandwidth",
                                       "Load Balancing & CDN"],
            "Software Licensing": ["Perpetual License", "Subscription / SaaS", "License Renewal",
                                   "Database Licensing", "Security Software"],
            "Operations, Maintenance & Support": ["Annual Maintenance Contract", "Managed Services",
                                                  "Helpdesk & Support", "Monitoring & Observability",
                                                  "Patching & Upgrades"],
            "Consulting & Professional Services": ["Implementation Services", "Advisory & Consulting",
                                                   "Training", "Staff Augmentation"],
            "Governance & Compliance": ["Audit & Assurance", "Regulatory Compliance",
                                        "Risk & Security Assessment"],
            "Others": [],
        },
    ),
    "support_tickets": ProfileConfig(
        label="Support tickets",
        description="customer or IT support tickets raised with a service desk",
        item_name="support ticket",
        level1_name="Issue category",
        level2_name="Issue type",
        taxonomy_hints="(e.g. access & identity, application defects, infrastructure incidents, "
                       "service requests, billing, and similar groupings you infer from the sample)",
        text_column_hints=["summary", "description", "subject", "title", "ticket text", "message"],
        low_confidence_threshold=0.55,
    ),
    "generic": ProfileConfig(
        label="Generic text records",
        description="free-text business records",
        item_name="record",
    ),
    "live_sentiment": ProfileConfig(
        # Demo profile for the dynamic hybrid live-scoring page (see /live). Flat taxonomy, no
        # sub-categories, so HierarchicalClassifier also runs it unmodified through the normal
        # batch/single-file pipeline -- this profile is not special-cased anywhere else.
        label="Live sentiment (demo)",
        description="short pieces of free text such as chat messages, comments or feedback",
        item_name="message",
        level1_name="Sentiment",
        level2_name="Nuance",
        text_column_hints=["text", "message", "comment", "feedback", "review"],
        backend="lexicon_sentiment",
        lexicon_asset="data/live_sentiment_lexicon.json",
        neutral_band=0.15,
        seed_taxonomy={"Positive": [], "Negative": [], "Neutral": []},
    ),
    "dino": ProfileConfig(
        # NOT run through HierarchicalClassifier / the batch pipeline -- no text column makes
        # sense for a per-tick game state. Exists purely to hand /decide/dino its action set
        # (seed_taxonomy keys double as labels), reflex_asset, and default escalation backend.
        label="Chrome Dino (reflex + System-1 escalation demo)",
        description="real-time obstacle-avoidance decisions for an offline endless-runner",
        item_name="game tick",
        level1_name="Action",
        backend="laya",                          # ESCALATION backend only -- the reflex layer
                                                   # always runs regardless of this. Swap to
                                                   # "causal_lm" for better-but-slower quality,
                                                   # or override per-request (see decide.py).
        reflex_asset="data/dino_reflex_rules.json",
        game_scripts=["dino_mini_game.js", "dino_adapter.js"],
        adapter_factory="createMiniAdapter",
        main_category_question=(
            "An obstacle is approaching in a real-time endless-runner game. Given its type, "
            "distance, and the player's current speed, which single action should the player "
            "take right now?"),
        seed_taxonomy={"Jump": [], "Duck": [], "Run": []},   # doubles as the action label set
    ),
}


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPENJEV_", extra="ignore")

    # ---- Application ----
    app_name: str = "OpenJEV Classifier"
    debug: bool = False                  # True: auto-reload + stack traces for unexpected errors in the UI
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # ---- Storage (everything lives under data_dir unless overridden) ----
    data_dir: Path = BASE_DIR / "data"
    inbox_dir: Optional[Path] = None             # default batch-mode folder
    upload_dir: Optional[Path] = None            # single-file uploads
    output_dir: Optional[Path] = None            # per-file JSON / CSV results
    taxonomy_dir: Optional[Path] = None          # <profile>.json taxonomy files
    log_dir: Optional[Path] = None
    db_path: Optional[Path] = None
    allowed_folder_roots: list[Path] = []        # extra folders batch mode may read (inbox is always allowed)
    allowed_extensions: list[str] = [".csv", ".tsv", ".xlsx", ".xlsm", ".xls"]
    max_upload_mb: int = 100
    upload_retention_days: int = 14              # abandoned previews (never run) are deleted after this

    # ---- Jobs (async processing) ----
    single_workers: int = 1                      # on-demand runs get their own worker so they never queue behind batches
    batch_workers: int = 1
    commit_every_rows: int = 25                  # rows saved per checkpoint -> resume point after a crash/restart
    max_consecutive_row_errors: int = 20         # abort a file if the engine fails this many rows in a row
    dedupe_identical_rows: bool = True           # identical narrations are classified once per file
    store_source_row: bool = True                # keep the original row (all columns) in the DB and JSON
    require_supplier_single: bool = True         # single-file mode must name a supplier
    poll_interval_ms: int = 1500                 # browser polling interval for job progress
    page_size: int = 25                          # rows per page in the results table

    # ---- Engine: which System-1 backend runs ----
    # "causal_lm" = Qwen logit read-off (permutation debiased); "laya" = Convai Laya;
    # "keyword"   = dependency-free baseline for smoke tests / offline demos.
    classifier_backend: BackendName = "causal_lm"
    laya_model_name: str = "convaiinnovations/laya"
    system1_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- System 2: taxonomy design via Ollama ----
    system2_enabled: bool = True
    auto_generate_taxonomy: bool = True          # design a taxonomy on first use of a profile
    ollama_base_url: str = "http://localhost:11434"
    system2_model: str = "gemma4:31b-cloud"
    system2_timeout_sec: int = 120
    system2_seed: int | None = 42                # None = let Ollama sample freely

    # ---- Taxonomy shape ----
    max_main_categories: int = 7                 # includes the catch-all
    max_sub_categories: int = 10
    taxonomy_sample_size: int = 40               # narrations sampled to design the taxonomy
    others_label: str = "Others"
    enforce_taxonomy_caps: bool = False          # False: manual edits above the caps only warn
    # Small models show strong positional bias in MCQ-style logit read-off; each choice is run this
    # many times with shuffled option order and averaged per label to cancel it out.
    debias_permutations: int = 3

    default_profile: str = "erp_costs"
    profiles: dict[str, ProfileConfig] = DEFAULT_PROFILES

    # ---- Global prompt templates (a profile may override any of them) ----
    # extra placeholders: {n} {sample} {max_main} {max_sub}
    taxonomy_generation_prompt: str = (
        "You are designing a classification taxonomy for {domain}.\n\n"
        "Here is a sample of {n} real {item_name}s from the dataset:\n{sample}\n\n"
        "Design a taxonomy with AT MOST {max_main} main categories and AT MOST {max_sub} "
        "sub-categories inside each main category. Main categories must be broad enough to "
        "cover the whole sample{taxonomy_hints}. Always include exactly one main "
        "category literally named \"{others_label}\" as a catch-all, with an EMPTY "
        "sub-category list (its sub-categories are produced dynamically per {item_name} later, "
        "not fixed up front).\n\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences, in this exact shape:\n"
        '{{"taxonomy": {{"Main Category A": ["Sub 1", "Sub 2"], "Main Category B": ["Sub 1"], '
        '"{others_label}": []}}}}'
    )
    main_category_question: str = (
        "In the context of {domain}, what is the {level1_name} of this {item_name}? "
        "Choose '{others_label}' only if none of the other options fit."
    )
    # extra placeholder: {main_category}
    sub_category_question: str = "Within '{main_category}', what is the specific {level2_name} of this {item_name}?"
    # extra placeholder: {narration}
    dynamic_label_prompt: str = (
        "{item_name}: {narration}\n"
        "In 2-4 words, name the most specific {level2_name} this belongs to:\n"
        "{level2_name}:"
    )

    @model_validator(mode="after")
    def _derive_paths(self) -> "AppSettings":
        d = self.data_dir
        self.inbox_dir = self.inbox_dir or d / "inbox"
        self.upload_dir = self.upload_dir or d / "uploads"
        self.output_dir = self.output_dir or d / "outputs"
        self.taxonomy_dir = self.taxonomy_dir or d / "taxonomies"
        self.log_dir = self.log_dir or d / "logs"
        self.db_path = self.db_path or d / "openjev.db"
        for p in (d, self.inbox_dir, self.upload_dir, self.output_dir, self.taxonomy_dir, self.log_dir, self.db_path.parent):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def _check_system1_not_laya(self) -> "AppSettings":
        # Catches the collision whatever its source (code default, .env, OS env var): the Laya
        # 'Others' fallback loads system1_model via AutoModelForCausalLM, which Laya's repo can't satisfy.
        if self.system1_model == self.laya_model_name:
            raise ValueError(
                f"system1_model and laya_model_name are both '{self.system1_model}'. system1_model must be a "
                "causal-LM checkpoint (e.g. a Qwen2.5 repo id). Check OPENJEV_SYSTEM1_MODEL in .env / your shell.")
        return self

    @model_validator(mode="after")
    def _check_default_profile(self) -> "AppSettings":
        if self.default_profile not in self.profiles:
            raise ValueError(f"default_profile '{self.default_profile}' is not defined in profiles: {list(self.profiles)}")
        return self

    # ---- Resolution helpers: the only place that knows profile-vs-global precedence ----
    def profile(self, key: str | None = None) -> ProfileConfig:
        key = key or self.default_profile
        if key not in self.profiles:
            raise ConfigError(f"Unknown dataset profile '{key}'.",
                              hint=f"Available: {', '.join(self.profiles)}. Add new profiles in app/config.py.")
        return self.profiles[key]

    def eff(self, key: str, field: str) -> Any:
        """Profile override if set, else the global value."""
        v = getattr(self.profile(key), field, None)
        return v if v is not None else getattr(self, field)

    def backend_for(self, key: str, override: str | None = None) -> str:
        chosen = override or self.profile(key).backend or self.classifier_backend
        if chosen not in ("causal_lm", "laya", "keyword", "lexicon_sentiment", "reflex_decision"):
            raise ConfigError(f"Unknown engine '{chosen}'.",
                              hint="Use causal_lm, laya, keyword, lexicon_sentiment or reflex_decision.")
        return chosen

    def prompt(self, key: str, name: str, **kw: Any) -> str:
        """Render a prompt for a profile: profile override -> global template -> placeholders filled."""
        p = self.profile(key)
        template = getattr(p, name, None) or getattr(self, name)
        ctx = {
            "domain": p.description, "item_name": p.item_name,
            "level1_name": p.level1_name.lower(), "level2_name": p.level2_name.lower(),
            "others_label": self.others_label,
            "taxonomy_hints": f" {p.taxonomy_hints}" if p.taxonomy_hints else "",
            **kw,
        }
        try:
            return template.format(**ctx)
        except (KeyError, IndexError) as e:
            raise ConfigError(f"Prompt '{name}' (profile '{key}') uses an unknown placeholder {e}.",
                              hint=f"Allowed here: {', '.join(sorted(ctx))}. Literal braces must be doubled.")


settings = AppSettings()
