"""
backend/services/analytics_service.py
=======================================
NLP analytics dashboard data aggregation.
All values computed from real DB records — zero hardcoded values.
"""

import json
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import Complaint, NlpMetric
from backend.utils.energy import detect_cpu_power_watts

# Detect CPU power once at module load (same logic as complaint_service).
ESTIMATED_CPU_POWER_WATTS, CPU_POWER_DETECTION_METHOD = detect_cpu_power_watts()


def build_analytics_dashboard(
    db: Session,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Return comprehensive NLP analytics. ALL values from real DB data."""
    from datetime import datetime

    base_q = db.query(NlpMetric)
    if start_date:
        try:
            sd = datetime.fromisoformat(start_date)
            base_q = base_q.filter(NlpMetric.created_at >= sd)
        except ValueError:
            pass
    if end_date:
        try:
            ed = datetime.fromisoformat(end_date)
            base_q = base_q.filter(NlpMetric.created_at <= ed)
        except ValueError:
            pass
    if language:
        base_q = base_q.filter(NlpMetric.source_language == language)

    total_nlp = base_q.count()

    if total_nlp == 0:
        return {
            "complaint_stats": {
                "total_complaints_processed": 0,
                "unique_complaints": db.query(Complaint).count(),
                "duplicate_complaints": 0,
                "total_votes": int(db.query(func.sum(Complaint.votes)).scalar() or 0),
                "average_votes_per_complaint": 0.0,
            },
            "nlp_stats": {
                "total_requests": 0,
                "avg_processing_time_seconds": 0.0,
                "avg_time_by_stage": {
                    "transcription": 0, "translation": 0, "classification": 0,
                    "ner": 0, "zero_shot": 0, "image_analysis": 0,
                },
                "zero_shot_fallback_rate": 0.0,
                "avg_classifier_confidence": 0.0,
                "avg_entity_count": 0.0,
                "entity_type_breakdown": [],
                "avg_word_count": 0.0,
                "avg_audio_duration": 0.0,
            },
            "energy_stats": {
                "total_energy_joules": 0.0,
                "avg_energy_per_complaint": 0.0,
                "energy_saved_by_dedup": 0.0,
                "energy_by_stage": {
                    "transcription": 0, "translation": 0, "classification": 0,
                    "ner": 0, "zero_shot": 0, "image_analysis": 0,
                },
                "calculation_method": CPU_POWER_DETECTION_METHOD,
            },
            "error_stats": {"total_errors": 0, "error_rate_percent": 0.0, "errors_by_stage": {}},
            "charts": {
                "energy_by_stage": [], "energy_over_time": [], "category_distribution": [],
                "duplicate_vs_unique": {"unique": 0, "duplicate": 0},
                "votes_per_complaint": [], "language_distribution": [],
                "confidence_histogram": [], "category_language_heatmap": [],
                "entity_count_histogram": [], "entity_type_breakdown": [],
                "stage_bottleneck_radar": {
                    "labels": ["Transcription", "Translation", "Classification", "NER", "Zero-shot", "Image Analysis"],
                    "avg_times": [0, 0, 0, 0, 0, 0],
                },
                "throughput_over_time": [], "audio_duration_vs_time": [],
                "duplicate_cluster_sizes": [], "error_rate_by_stage": [],
                "severity_distribution": [],
            },
            "data_sources": {
                "note": "No NLP metrics recorded yet. Submit complaints to populate analytics."
            },
        }

    # ── Complaint Stats ──────────────────────────────────────────
    unique_complaints = db.query(Complaint).count()
    dup_count = base_q.filter(NlpMetric.is_duplicate == True).count()
    total_votes = int(db.query(func.sum(Complaint.votes)).scalar() or 0)
    avg_votes = round(total_votes / unique_complaints, 2) if unique_complaints else 0.0

    # ── NLP Stats ────────────────────────────────────────────────
    _ids = base_q.with_entities(NlpMetric.id)

    def _avg(col):
        return float(db.query(func.avg(col)).filter(NlpMetric.id.in_(_ids)).scalar() or 0)

    avg_proc   = _avg(NlpMetric.total_processing_time)
    avg_trans  = _avg(NlpMetric.transcription_time)
    avg_transl = _avg(NlpMetric.translation_time)
    avg_clf    = _avg(NlpMetric.classification_time)
    avg_ner    = _avg(NlpMetric.ner_time)
    avg_zs     = _avg(NlpMetric.zero_shot_time)
    avg_img    = _avg(NlpMetric.image_analysis_time)

    zs_triggered = base_q.filter(NlpMetric.zero_shot_triggered == True).count()
    zs_rate = round((zs_triggered / total_nlp) * 100, 2) if total_nlp else 0.0

    avg_conf = float(
        db.query(func.avg(NlpMetric.classifier_confidence))
        .filter(NlpMetric.id.in_(_ids), NlpMetric.classifier_confidence.isnot(None))
        .scalar() or 0
    )
    avg_ent = _avg(NlpMetric.entity_count)
    avg_wc = _avg(NlpMetric.word_count)
    avg_audio = float(
        db.query(func.avg(NlpMetric.audio_duration_seconds))
        .filter(NlpMetric.id.in_(_ids), NlpMetric.audio_duration_seconds.isnot(None))
        .scalar() or 0
    )

    # ── Entity type breakdown ────────────────────────────────────
    all_entity_types_agg = {}
    for row in base_q.with_entities(NlpMetric.entity_types).all():
        if row[0]:
            try:
                et = json.loads(row[0])
                for k, v in et.items():
                    all_entity_types_agg[k] = all_entity_types_agg.get(k, 0) + v
            except (json.JSONDecodeError, TypeError):
                pass

    # ── Energy Stats ─────────────────────────────────────────────
    total_energy = float(
        db.query(func.sum(NlpMetric.total_energy_joules))
        .filter(NlpMetric.id.in_(_ids))
        .scalar() or 0
    )
    avg_energy = round(total_energy / total_nlp, 6) if total_nlp else 0.0
    energy_saved = round(avg_energy * dup_count, 6)

    energy_stage_agg = {
        "transcription": 0.0, "translation": 0.0, "classification": 0.0,
        "ner": 0.0, "zero_shot": 0.0, "image_analysis": 0.0,
    }
    for row in base_q.with_entities(NlpMetric.energy_by_stage).all():
        if row[0]:
            try:
                es = json.loads(row[0])
                for k, v in es.items():
                    energy_stage_agg[k] = energy_stage_agg.get(k, 0) + float(v)
            except (json.JSONDecodeError, TypeError):
                pass

    # ── Error Stats ──────────────────────────────────────────────
    total_errors = base_q.filter(NlpMetric.error_stage.isnot(None)).count()
    error_rate = round((total_errors / total_nlp) * 100, 2) if total_nlp else 0.0
    errors_by_stage_q = (
        db.query(NlpMetric.error_stage, func.count(NlpMetric.id))
        .filter(NlpMetric.id.in_(_ids), NlpMetric.error_stage.isnot(None))
        .group_by(NlpMetric.error_stage)
        .all()
    )
    errors_by_stage = {r[0]: r[1] for r in errors_by_stage_q}

    # ── Charts Data ──────────────────────────────────────────────
    energy_by_stage_chart = [
        {"stage": k, "joules": round(v, 4)} for k, v in energy_stage_agg.items()
    ]

    energy_over_time_q = (
        db.query(
            func.date(NlpMetric.created_at).label("date"),
            func.sum(NlpMetric.total_energy_joules).label("joules"),
            func.count(NlpMetric.id).label("count"),
        )
        .filter(NlpMetric.id.in_(_ids))
        .group_by(func.date(NlpMetric.created_at))
        .order_by(func.date(NlpMetric.created_at))
        .all()
    )
    energy_over_time = [
        {"date": str(r[0]), "joules": round(float(r[1] or 0), 4), "count": r[2]}
        for r in energy_over_time_q
    ]

    cat_dist_q = (
        db.query(NlpMetric.category, func.count(NlpMetric.id))
        .filter(NlpMetric.id.in_(_ids))
        .group_by(NlpMetric.category)
        .all()
    )
    category_distribution = [{"category": r[0] or "Unknown", "count": r[1]} for r in cat_dist_q]

    unique_count = base_q.filter(NlpMetric.is_duplicate == False).count()
    dup_vs_unique = {"unique": unique_count, "duplicate": dup_count}

    votes_q = (
        db.query(Complaint.id, Complaint.votes, Complaint.category)
        .filter(Complaint.votes > 0)
        .order_by(Complaint.votes.desc())
        .limit(20)
        .all()
    )
    votes_per_complaint = [{"complaint_id": r[0], "votes": r[1] or 0, "category": r[2] or ""} for r in votes_q]

    lang_dist_q = (
        db.query(
            NlpMetric.source_language,
            func.count(NlpMetric.id),
            func.avg(NlpMetric.total_processing_time),
        )
        .filter(NlpMetric.id.in_(_ids))
        .group_by(NlpMetric.source_language)
        .all()
    )
    language_distribution = [
        {"language": r[0] or "unknown", "count": r[1], "avg_processing_time": round(float(r[2] or 0), 4)}
        for r in lang_dist_q
    ]

    confidence_histogram = []
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        cnt = base_q.filter(
            NlpMetric.classifier_confidence >= lo,
            NlpMetric.classifier_confidence < hi if i < 9 else NlpMetric.classifier_confidence <= hi,
        ).count()
        confidence_histogram.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": cnt})

    severity_dist_q = (
        db.query(NlpMetric.pothole_severity, func.count(NlpMetric.id))
        .filter(NlpMetric.id.in_(_ids), NlpMetric.pothole_severity.isnot(None))
        .group_by(NlpMetric.pothole_severity)
        .all()
    )
    severity_distribution = [{"severity": r[0], "count": r[1]} for r in severity_dist_q]

    cat_lang_q = (
        db.query(NlpMetric.category, NlpMetric.source_language, func.count(NlpMetric.id))
        .filter(NlpMetric.id.in_(_ids))
        .group_by(NlpMetric.category, NlpMetric.source_language)
        .all()
    )
    category_language_heatmap = [
        {"category": r[0] or "Unknown", "language": r[1] or "unknown", "count": r[2]}
        for r in cat_lang_q
    ]

    ent_hist_q = (
        db.query(NlpMetric.entity_count, func.count(NlpMetric.id))
        .filter(NlpMetric.id.in_(_ids))
        .group_by(NlpMetric.entity_count)
        .order_by(NlpMetric.entity_count)
        .all()
    )
    entity_count_histogram = [{"entities": r[0] or 0, "count": r[1]} for r in ent_hist_q]

    entity_type_breakdown = [
        {"type": k, "count": v}
        for k, v in sorted(all_entity_types_agg.items(), key=lambda x: x[1], reverse=True)
    ]

    stage_bottleneck_radar = {
        "labels": ["Transcription", "Translation", "Classification", "NER", "Zero-shot", "Image Analysis"],
        "avg_times": [
            round(avg_trans, 4), round(avg_transl, 4), round(avg_clf, 4),
            round(avg_ner, 4), round(avg_zs, 4), round(avg_img, 4),
        ],
    }

    throughput_q = (
        db.query(func.date(NlpMetric.created_at).label("day"), func.count(NlpMetric.id).label("count"))
        .filter(NlpMetric.id.in_(_ids))
        .group_by(func.date(NlpMetric.created_at))
        .order_by(func.date(NlpMetric.created_at))
        .all()
    )
    throughput_over_time = [{"hour": str(r[0]), "count": r[1]} for r in throughput_q]

    audio_scatter_q = (
        base_q.filter(NlpMetric.audio_duration_seconds.isnot(None), NlpMetric.audio_duration_seconds > 0)
        .with_entities(NlpMetric.audio_duration_seconds, NlpMetric.total_processing_time)
        .limit(200)
        .all()
    )
    audio_duration_vs_time = [
        {"duration_s": round(float(r[0]), 2), "processing_time_s": round(float(r[1]), 4)}
        for r in audio_scatter_q
    ]

    dup_cluster_q = (
        db.query(Complaint.votes, func.count(Complaint.id))
        .filter(Complaint.votes > 1)
        .group_by(Complaint.votes)
        .order_by(Complaint.votes)
        .all()
    )
    duplicate_cluster_sizes = [{"cluster_size": r[0], "count": r[1]} for r in dup_cluster_q]

    stages = ["transcription", "translation", "classification", "ner", "zero_shot", "image_analysis"]
    error_rate_by_stage = []
    for stage in stages:
        stage_total = base_q.count()
        stage_errors = errors_by_stage.get(stage, 0)
        error_rate_by_stage.append({
            "stage": stage,
            "error_count": stage_errors,
            "total_count": stage_total,
            "rate_percent": round((stage_errors / stage_total) * 100, 2) if stage_total else 0,
        })

    return {
        "complaint_stats": {
            "total_complaints_processed": total_nlp,
            "unique_complaints": unique_complaints,
            "duplicate_complaints": dup_count,
            "total_votes": total_votes,
            "average_votes_per_complaint": avg_votes,
        },
        "nlp_stats": {
            "total_requests": total_nlp,
            "avg_processing_time_seconds": round(avg_proc, 4),
            "avg_time_by_stage": {
                "transcription": round(avg_trans, 4),
                "translation": round(avg_transl, 4),
                "classification": round(avg_clf, 4),
                "ner": round(avg_ner, 4),
                "zero_shot": round(avg_zs, 4),
                "image_analysis": round(avg_img, 4),
            },
            "zero_shot_fallback_rate": zs_rate,
            "avg_classifier_confidence": round(avg_conf, 4),
            "avg_entity_count": round(avg_ent, 2),
            "entity_type_breakdown": entity_type_breakdown,
            "avg_word_count": round(avg_wc, 1),
            "avg_audio_duration": round(avg_audio, 2),
        },
        "energy_stats": {
            "total_energy_joules": round(total_energy, 4),
            "avg_energy_per_complaint": round(avg_energy, 4),
            "energy_saved_by_dedup": round(energy_saved, 4),
            "energy_by_stage": {k: round(v, 4) for k, v in energy_stage_agg.items()},
            "calculation_method": CPU_POWER_DETECTION_METHOD,
        },
        "error_stats": {
            "total_errors": total_errors,
            "error_rate_percent": error_rate,
            "errors_by_stage": errors_by_stage,
        },
        "charts": {
            "energy_by_stage": energy_by_stage_chart,
            "energy_over_time": energy_over_time,
            "category_distribution": category_distribution,
            "duplicate_vs_unique": dup_vs_unique,
            "votes_per_complaint": votes_per_complaint,
            "language_distribution": language_distribution,
            "confidence_histogram": confidence_histogram,
            "category_language_heatmap": category_language_heatmap,
            "entity_count_histogram": entity_count_histogram,
            "entity_type_breakdown": entity_type_breakdown,
            "stage_bottleneck_radar": stage_bottleneck_radar,
            "throughput_over_time": throughput_over_time,
            "audio_duration_vs_time": audio_duration_vs_time,
            "duplicate_cluster_sizes": duplicate_cluster_sizes,
            "error_rate_by_stage": error_rate_by_stage,
            "severity_distribution": severity_distribution,
        },
        "data_sources": {
            "complaint_stats": "SELECT COUNT/SUM from complaints table",
            "nlp_metrics": "SELECT from nlp_metrics table (real per-request measurements via time.perf_counter())",
            "energy": f"Estimated TDP ({ESTIMATED_CPU_POWER_WATTS}W) × measured processing time. {CPU_POWER_DETECTION_METHOD}",
            "entities": "spaCy en_core_web_sm NER — entity_count and entity_types stored per request",
            "confidence": "sklearn predict_proba() stored as classifier_confidence per request",
            "audio_duration": "pydub AudioSegment.duration_seconds from uploaded audio",
            "errors": "Caught exceptions logged with stage name to nlp_metrics.error_stage",
            "note": "ALL values computed from database records and runtime logs. Zero hardcoded values.",
        },
    }
