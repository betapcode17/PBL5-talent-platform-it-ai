import pytest

from app.services.rag.score_and_boost.scoring import ScoreComponents, compose_hybrid_score


def test_compose_hybrid_score_include_fields_ignores_excluded_components():
    # Use moderate weights to avoid clamping to [-1,1] and make differences observable
    weights = {
        "semantic": 0.6,
        "bm25": 0.4,
        "title": 0.2,
        "description": 0.2,
        "skills": 0.2,
        "company": 0.2,
        "category": 0.2,
        "entity_bias": 0.0,
        "recency": 0.0,
        "fulltext": 0.0,
    }

    components = ScoreComponents(
        semantic=0.5,
        bm25=0.2,
        title=0.3,
        company=0.7,
        description=0.4,
        skills=0.9,
        category=0.6,
        recency=0.0,
        entity_bias=0.0,
        fulltext=0.0,
    )

    # Score when include_fields restricts to title and description only
    score_restricted = compose_hybrid_score(weights, components, fulltext_weight=0.0, include_fields=["title", "description"])

    # Changing excluded fields should NOT change the restricted score
    components_changed = ScoreComponents(
        semantic=0.9,
        bm25=0.8,
        title=0.3,
        company=0.0,
        description=0.4,
        skills=0.0,
        category=0.0,
        recency=0.0,
        entity_bias=1.0,
        fulltext=0.0,
    )

    score_restricted_changed = compose_hybrid_score(weights, components_changed, fulltext_weight=0.0, include_fields=["title", "description"])

    assert score_restricted == pytest.approx(score_restricted_changed, rel=1e-9)

    # But with no include_fields restriction, the same changes should affect score
    score_unrestricted = compose_hybrid_score(weights, components, fulltext_weight=0.0, include_fields=None)
    score_unrestricted_changed = compose_hybrid_score(weights, components_changed, fulltext_weight=0.0, include_fields=None)

    assert score_unrestricted != score_unrestricted_changed
