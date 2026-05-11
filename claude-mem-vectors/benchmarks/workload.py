"""Synthetic workload generator matching claude-mem's document mix.

Empirical claude-mem mix (from ChromaSync.formatObservationDocs / formatSummaryDocs
/ formatUserPromptDoc): roughly
    ~70% observation field-docs (1 narrative + ~3 facts per observation)
    ~20% session_summary field-docs (~5 fields per summary)
    ~10% user_prompt docs (1 per prompt)

Sampling is deterministic given the seed so benchmark runs are reproducible.

Each generated VectorDocument has:
    id          obs_{i}_narrative | obs_{i}_fact_{j} | summary_{i}_{field} | prompt_{i}
    text        synthetic prose, length ~ realistic for the doc type
    metadata    {doc_type, sqlite_id, project, field_type, created_at_epoch}
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

from store.vector_store import VectorDocument


SUMMARY_FIELDS = ("request", "investigated", "learned", "completed", "next_steps", "notes")
OBSERVATION_TYPES = ("research", "implementation", "debug", "review")


@dataclass
class WorkloadConfig:
    n_docs: int                                  # total docs to generate
    n_projects: int = 4                          # spread across this many projects
    facts_per_observation: int = 3
    summaries_share: float = 0.20                # fraction of total docs that are summary fields
    prompts_share: float = 0.10                  # fraction that are user_prompts
    seed: int = 42
    base_time: int | None = None                 # epoch ms; defaults to "now"


def generate(cfg: WorkloadConfig) -> list[VectorDocument]:
    rng = random.Random(cfg.seed)
    base_time = cfg.base_time if cfg.base_time is not None else int(time.time() * 1000)

    n_summary_docs = int(cfg.n_docs * cfg.summaries_share)
    n_prompt_docs = int(cfg.n_docs * cfg.prompts_share)
    n_obs_docs = cfg.n_docs - n_summary_docs - n_prompt_docs

    # Observations: 1 narrative + N facts per observation.
    docs_per_obs = 1 + cfg.facts_per_observation
    n_observations = max(1, n_obs_docs // docs_per_obs)
    n_summaries = max(1, n_summary_docs // len(SUMMARY_FIELDS))

    out: list[VectorDocument] = []

    # Observations.
    for obs_id in range(n_observations):
        project = f"project_{rng.randrange(cfg.n_projects)}"
        ts = base_time - rng.randrange(0, 90 * 86_400 * 1000)
        obs_type = rng.choice(OBSERVATION_TYPES)

        out.append(VectorDocument(
            id=f"obs_{obs_id}_narrative",
            text=_obs_narrative(rng, obs_id, obs_type),
            metadata={
                "doc_type": "observation",
                "sqlite_id": obs_id,
                "project": project,
                "field_type": "narrative",
                "obs_type": obs_type,
                "created_at_epoch": ts,
            },
        ))
        for f_idx in range(cfg.facts_per_observation):
            out.append(VectorDocument(
                id=f"obs_{obs_id}_fact_{f_idx}",
                text=_obs_fact(rng, obs_id, f_idx),
                metadata={
                    "doc_type": "observation",
                    "sqlite_id": obs_id,
                    "project": project,
                    "field_type": f"fact_{f_idx}",
                    "obs_type": obs_type,
                    "created_at_epoch": ts,
                },
            ))

    # Summaries.
    for sum_id in range(n_summaries):
        project = f"project_{rng.randrange(cfg.n_projects)}"
        ts = base_time - rng.randrange(0, 90 * 86_400 * 1000)
        for field in SUMMARY_FIELDS:
            out.append(VectorDocument(
                id=f"summary_{sum_id}_{field}",
                text=_summary_text(rng, sum_id, field),
                metadata={
                    "doc_type": "session_summary",
                    "sqlite_id": sum_id,
                    "project": project,
                    "field_type": field,
                    "created_at_epoch": ts,
                },
            ))

    # User prompts.
    for prompt_id in range(n_prompt_docs):
        project = f"project_{rng.randrange(cfg.n_projects)}"
        ts = base_time - rng.randrange(0, 90 * 86_400 * 1000)
        out.append(VectorDocument(
            id=f"prompt_{prompt_id}",
            text=_prompt_text(rng, prompt_id),
            metadata={
                "doc_type": "user_prompt",
                "sqlite_id": prompt_id,
                "project": project,
                "field_type": "prompt",
                "created_at_epoch": ts,
            },
        ))

    # Trim/pad to exact n_docs (we may be a few off due to integer division).
    if len(out) > cfg.n_docs:
        out = out[: cfg.n_docs]
    rng.shuffle(out)  # Realistic write-order: interleaved by time, not type.
    return out


def batched(docs: Sequence[VectorDocument], batch_size: int) -> Iterator[list[VectorDocument]]:
    for i in range(0, len(docs), batch_size):
        yield list(docs[i : i + batch_size])


# ---- text generators (cheap, ~realistic length) ----

_VERBS = ("found", "noticed", "implemented", "fixed", "refactored", "tested", "investigated")
_NOUNS = ("query", "endpoint", "table", "function", "module", "field", "schema", "test", "hook")
_ADJ = ("slow", "redundant", "missing", "broken", "deprecated", "legacy", "fragile")


def _obs_narrative(rng: random.Random, obs_id: int, obs_type: str) -> str:
    sents = []
    for _ in range(rng.randrange(3, 6)):
        sents.append(
            f"The agent {rng.choice(_VERBS)} a {rng.choice(_ADJ)} {rng.choice(_NOUNS)} "
            f"in the {obs_type} flow."
        )
    return f"Observation {obs_id}: " + " ".join(sents)


def _obs_fact(rng: random.Random, obs_id: int, f_idx: int) -> str:
    return (
        f"Fact {f_idx} for obs {obs_id}: the {rng.choice(_NOUNS)} "
        f"is {rng.choice(_ADJ)} and was {rng.choice(_VERBS)}."
    )


def _summary_text(rng: random.Random, sum_id: int, field: str) -> str:
    return (
        f"Summary {sum_id} {field}: the assistant {rng.choice(_VERBS)} "
        f"the {rng.choice(_NOUNS)} after the user requested {rng.choice(_NOUNS)} review."
    )


def _prompt_text(rng: random.Random, prompt_id: int) -> str:
    return (
        f"Prompt {prompt_id}: please {rng.choice(_VERBS)} the {rng.choice(_NOUNS)} "
        f"and explain why it is {rng.choice(_ADJ)}."
    )


# ---- query texts for benchmarks ----

def query_pool(seed: int = 99, n: int = 32) -> list[str]:
    """A small pool of realistic-ish queries for query-side benchmarks."""
    rng = random.Random(seed)
    return [
        f"why is {rng.choice(_NOUNS)} {rng.choice(_ADJ)}?"
        for _ in range(n)
    ]
