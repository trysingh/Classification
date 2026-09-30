"""
hierarchical.py
---------------
The one place that knows the classification algorithm:

    main category (closed set) -> sub-category (closed set)
    main == 'Others'           -> short generated label instead of a fixed sub-category

It is backend-agnostic (any ChoiceBackend) and profile-aware (prompts, debias count and labels
come from config), so a new dataset never needs a new classifier class.
"""
from __future__ import annotations

import time 

from app.core.errors import BackendError


class HierarchicalClassifier:
    def __init__(self, backend, taxonomy: dict[str, list[str]], settings, profile_key: str):
        self.backend, self.taxonomy, self.settings, self.profile_key = backend, taxonomy, settings, profile_key
        self.others = settings.others_label
        self.debias = settings.eff(profile_key, "debias_permutations")
        self.main_labels = list(taxonomy)
        self.main_q = settings.prompt(profile_key, "main_category_question")
        print(f"Main Category Question: {self.main_q}")
        # Prompts depend only on the main category, so render them once per taxonomy, not once per row.
        self.sub_q = {m: settings.prompt(profile_key, "sub_category_question", main_category=m)
                      for m in taxonomy if m != self.others}
        self.main_hints = {m: " ".join([m, *subs]) for m, subs in taxonomy.items()}

    @staticmethod
    def _top3(probs: dict[str, float]) -> str:
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        return " | ".join(f"{label} ({p:.1%})" for label, p in ranked)

    def _choose(self, context: str, question: str, labels: list[str], hints=None) -> dict:
        if len(labels) == 1:                                     # nothing to decide: skip the model call
            return {"choice": labels[0], "probabilities": {labels[0]: 1.0}}
        res = self.backend.choose(context, question, labels, debias=self.debias, hints=hints)
        if res["choice"] not in labels:
            raise BackendError("The engine returned a label that is not in the taxonomy.",
                               detail=f"choice={res['choice']!r}, expected one of {labels}")
        return res

    def classify(self, narration: str, context: str | None = None) -> dict:
        """`context` = narration plus optional supplier / extra columns; defaults to the narration alone."""
        t0 = time.perf_counter()
        context = context or narration
        main = self._choose(context, self.main_q, self.main_labels, self.main_hints)
        cat = main["choice"]
        out = {"main_category": cat,
               "main_confidence": round(float(main["probabilities"].get(cat, 0.0)), 4),
               "main_top3": self._top3(main["probabilities"])}
        subs = self.taxonomy.get(cat, [])

        if cat == self.others:
            label = self.backend.generate_label(
                narration, self.settings.prompt(self.profile_key, "dynamic_label_prompt", narration=narration))
            out |= {"sub_category": label, "sub_confidence": 0.0,
                    "sub_top3": "N/A (dynamically generated, not a closed-set probability)"}
        elif not subs:
            out |= {"sub_category": "", "sub_confidence": 0.0, "sub_top3": "N/A (no sub-categories defined)"}
        else:
            sub = self._choose(context, self.sub_q[cat], subs)
            c = sub["choice"]
            out |= {"sub_category": c, "sub_confidence": round(float(sub["probabilities"].get(c, 0.0)), 4),
                    "sub_top3": self._top3(sub["probabilities"])}

        out["time_taken_sec"] = round(time.perf_counter() - t0, 4)
        return out
