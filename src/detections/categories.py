"""Semantic class id -> name mapping from the habitat semantic scene."""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def build_category_names(sim) -> Dict[int, str]:
    """Map semantic class id -> category name from ``sim.semantic_scene.categories``.

    Returns an empty dict if the scene has no category lexicon; callers should
    fall back to ``str(class_id)`` for unmapped ids.
    """
    out: Dict[int, str] = {}
    ss = getattr(sim, "semantic_scene", None)
    cats = getattr(ss, "categories", None) if ss is not None else None
    if not cats:
        return out
    for c in cats:
        try:
            out[int(c.index())] = str(c.name())
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed semantic category %r: %s", c, exc)
            continue
    return out


def name_for(categories: Dict[int, str], class_id: int) -> str:
    """Category name for ``class_id``, falling back to the numeric id."""
    return categories.get(int(class_id), str(int(class_id)))
