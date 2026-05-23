"""Tests for job query normalization and retrieval routing."""

from __future__ import annotations

from app.services.rag.retrieval import RAGRetrievalService


def test_java_query_routes_to_strict_job_search() -> None:
    profile = RAGRetrievalService._resolve_runtime_profile("java", None)

    assert profile.name == "strict-job-search"


def test_backend_java_query_routes_to_same_strict_profile() -> None:
    profile = RAGRetrievalService._resolve_runtime_profile("backend java", None)

    assert profile.name == "strict-job-search"


def test_job_query_plan_keeps_role_and_skill_intent() -> None:
    plan = RAGRetrievalService._build_job_query_plan("backend java", {"backend", "java"})

    assert plan["role_hint"] == "backend"
    assert "java" in plan["terms"]
    assert "backend" in plan["terms"]
    assert "backend" in plan["query"]
    assert "java" in plan["query"]


def test_city_alias_matches_ho_chi_minh_city() -> None:
    assert RAGRetrievalService._detect_city_hint("tim viec backend java hcm") == "hcm"
    assert RAGRetrievalService._matches_city_hint({"location": "Ho Chi Minh City"}, "hcm")


def test_minimum_relevance_uses_chunk_text_when_metadata_is_sparse() -> None:
    candidate = {
        "metadata": {"title": "Software Engineer"},
        "text": "We are hiring a Python backend engineer in Ho Chi Minh City.",
    }

    assert RAGRetrievalService._has_minimum_relevance(candidate, {"backend", "python", "hcm"})