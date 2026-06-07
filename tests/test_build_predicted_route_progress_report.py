from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "build_predicted_route_progress_report.py"
    spec = importlib.util.spec_from_file_location("build_predicted_route_progress_report", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_make_progress_rows_collects_claimable_and_diagnostic_rows(tmp_path):
    module = load_module()
    root = tmp_path / "outputs" / "oracle_route_memory"

    write_csv(
        root / "eval_smoke_20260606_072300" / "summary.csv",
        "mode,prefix_len,Recall@10,Recall@20,Recall@50,NDCG@50,MRR@50",
        [
            "metadata,,0.01,0.02,0.01,0.01,0.01",
            "oracle_route,1,0.02,0.03,0.05,0.02,0.01",
            "oracle_route,2,0.20,0.40,0.70,0.22,0.09",
        ],
    )
    write_csv(
        root / "official_comparison_20260607" / "comparison.csv",
        "method_key,display_name,selection_status,subset,domain,Recall@10,Recall@20,Recall@50,NDCG@50,MRR@50",
        [
            "predicted_route_validation_selected,Predicted Route Validation-Selected,claimable,cold,ALL,0.01,0.02,0.03,0.01,0.01",
            "predicted_route_diagnostic_fusion_lr_p1t4,Predicted Route Diagnostic Fusion LR P1T4,diagnostic_only,cold,ALL,0.02,0.03,0.04,0.02,0.01",
        ],
    )
    write_csv(
        root / "validation_fusion_locked_cold_gpu0" / "summary.csv",
        "query_source,subset,mode,Recall@10,Recall@20,Recall@50,NDCG@50,MRR@50",
        ["selected_policy,cold,validation_selected,0.01,0.02,0.03,0.01,0.01"],
    )
    write_csv(
        root / "fusion_domain_adaptive_eval_20260606_233954" / "summary.csv",
        "query_source,subset,mode,Recall@10,Recall@20,Recall@50,NDCG@50,MRR@50",
        ["domain_adaptive,cold,domain_prior_p1,0.02,0.03,0.05,0.02,0.01"],
    )
    write_csv(
        root / "prefix1_domain_adaptive_merge_eval_20260606_232037" / "summary.csv",
        "query_source,subset,mode,Recall@10,Recall@20,Recall@50,NDCG@50,MRR@50",
        ["domain_adaptive,cold,predicted_route_p1_top4_round_robin,0.01,0.02,0.04,0.02,0.01"],
    )
    write_csv(
        root / "validation_fusion_locked_cold_seed7_policy_gpu0" / "summary.csv",
        "query_source,subset,mode,Recall@10,Recall@20,Recall@50,NDCG@50,MRR@50",
        ["mean,cold,predicted_route_p1_top4_zscore,0.01,0.02,0.02,0.01,0.01"],
    )

    rows = module.make_progress_rows(root)
    labels = {row["label"] for row in rows}
    assert "Metadata Global" in labels
    assert "Predicted Route Validation-Selected" in labels
    assert "Domain Adaptive Fusion" in labels

    claimable = next(row for row in rows if row["label"] == "Validation Selected Policy")
    assert claimable["selection_status"] == "claimable"
    assert claimable["AboveTarget0.029"] is True

    report = module.render_markdown(rows)
    assert "Best claimable row" in report
    assert "Current target threshold" in report
