"""Pathway enrichment utilities migrated from classifier analysis notebook.

Notebook source:
  - demo/classifier_analysis.ipynb
    - PathwayEnrichmentRunner (cell 11)
    - run_cluster_enrichment_from_df (cell 20)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _lazy_gseapy() -> Any:
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError(
            "gseapy is required for enrichment runs. Install with: pip install gseapy"
        ) from e
    return gp


@dataclass
class EnrichmentConfig:
    gene_sets: str = "Reactome_2022"
    organism: str = "human"
    adj_p_cutoff: float = 0.05
    min_gene_n: int = 5
    output_dir: str = "EnrichmentResults"
    verbose: bool = True
    max_retries: int = 4
    retry_base_delay: float = 2.0
    retry_jitter: float = 0.5
    inter_request_delay: float = 1.0


class PathwayEnrichmentRunner:
    """Reusable wrapper for enrichment analysis on ranked gene lists."""

    def __init__(self, config: EnrichmentConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = EnrichmentConfig(**kwargs)
        self.cfg = config
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def min_gene_n(self) -> int:
        return int(self.cfg.min_gene_n)

    @property
    def gene_sets(self) -> str:
        return str(self.cfg.gene_sets)

    @staticmethod
    def _clean_gene_list(gene_list: list[str] | pd.Series | np.ndarray | None) -> list[str]:
        if gene_list is None:
            raise ValueError("gene_list is None")
        clean_genes = pd.Series(gene_list).dropna().astype(str).str.strip()
        clean_genes = clean_genes[clean_genes != ""].drop_duplicates().tolist()
        return clean_genes

    @staticmethod
    def _remove_reactome_id(term: str) -> str:
        return (
            pd.Series([term])
            .astype(str)
            .str.replace(r"\s*R-HSA-\d+\s*$", "", regex=True)
            .iloc[0]
        )

    @staticmethod
    def _parse_overlap_ratio(overlap: str) -> float:
        try:
            a, b = str(overlap).split("/")
            a_f, b_f = float(a), float(b)
            return a_f / b_f if b_f > 0 else np.nan
        except Exception:
            return np.nan

    @staticmethod
    def _prepare_enrichment_table(res: pd.DataFrame) -> pd.DataFrame:
        out = res.copy()
        if "Term" in out.columns:
            out["Term_clean"] = out["Term"].astype(str).map(
                PathwayEnrichmentRunner._remove_reactome_id
            )
        else:
            out["Term_clean"] = out.iloc[:, 0].astype(str)

        if "Adjusted P-value" not in out.columns:
            raise ValueError("Enrichment result missing 'Adjusted P-value' column")

        if "Overlap" in out.columns:
            out["Overlap_Ratio"] = out["Overlap"].map(
                PathwayEnrichmentRunner._parse_overlap_ratio
            )
        else:
            out["Overlap_Ratio"] = np.nan

        out["NegLog10_Adjusted_P"] = -np.log10(
            out["Adjusted P-value"].clip(lower=1e-300)
        )
        return out

    @staticmethod
    def extract_top_genes(
        df: pd.DataFrame,
        gene_col: str = "Gene",
        score_col: str = "MeanAbsDelta",
        top_n: int = 150,
        ascending: bool = False,
        min_freq_col: str | None = None,
        min_freq: float | None = None,
        extra_query: str | None = None,
        absolute_score: bool = False,
        drop_duplicates: bool = True,
    ) -> tuple[list[str], pd.DataFrame]:
        if gene_col not in df.columns:
            raise ValueError(f"Missing gene column: {gene_col}")
        if score_col not in df.columns:
            raise ValueError(f"Missing score column: {score_col}")

        sub = df.copy()
        if min_freq_col is not None and min_freq is not None:
            if min_freq_col not in sub.columns:
                raise ValueError(f"Missing frequency column: {min_freq_col}")
            sub = sub[sub[min_freq_col] >= min_freq].copy()

        if extra_query is not None:
            sub = sub.query(extra_query).copy()

        sub["_rank_score_"] = (
            sub[score_col].abs() if absolute_score else pd.to_numeric(sub[score_col], errors="coerce")
        )
        sub = sub.sort_values("_rank_score_", ascending=ascending)
        if drop_duplicates:
            sub = sub.drop_duplicates(subset=[gene_col], keep="first")

        top_df = sub.head(top_n).copy()
        top_genes = top_df[gene_col].dropna().astype(str).str.strip().tolist()
        return top_genes, top_df

    def run(self, gene_list: list[str], title: str | None = None) -> dict[str, Any]:
        clean_genes = self._clean_gene_list(gene_list)
        if len(clean_genes) < self.cfg.min_gene_n:
            raise ValueError(
                f"Not enough genes for enrichment: {len(clean_genes)} < {self.cfg.min_gene_n}"
            )

        gp = _lazy_gseapy()
        last_error: Exception | None = None
        enr = None

        for attempt in range(self.cfg.max_retries + 1):
            try:
                if attempt > 0 and self.cfg.verbose:
                    print(f"Retrying enrichment ({attempt}/{self.cfg.max_retries}) ...")
                enr = gp.enrichr(
                    gene_list=clean_genes,
                    gene_sets=self.cfg.gene_sets,
                    organism=self.cfg.organism,
                    outdir=None,
                )
                break
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                is_rate_limited = ("429" in msg) or ("rate limit" in msg)
                if (not is_rate_limited) or attempt >= self.cfg.max_retries:
                    raise
                delay = self.cfg.retry_base_delay * (2 ** attempt) + random.uniform(
                    0.0, self.cfg.retry_jitter
                )
                if self.cfg.verbose:
                    print(f"Rate limited by Enrichr. Waiting {delay:.1f}s before retry ...")
                time.sleep(delay)
        if enr is None and last_error is not None:
            raise last_error

        if self.cfg.inter_request_delay > 0:
            time.sleep(self.cfg.inter_request_delay)

        res = enr.results.copy()
        if res.empty:
            return {
                "title": title,
                "input_genes": clean_genes,
                "all_results": res,
                "sig_results": res,
            }

        res = self._prepare_enrichment_table(res)
        sig_res = res[res["Adjusted P-value"] < self.cfg.adj_p_cutoff].copy()
        if sig_res.empty:
            return {
                "title": title,
                "input_genes": clean_genes,
                "all_results": res,
                "sig_results": sig_res,
            }

        if "Combined Score" in sig_res.columns:
            sig_res = sig_res.sort_values(
                ["Adjusted P-value", "Combined Score"], ascending=[True, False]
            )
        else:
            sig_res = sig_res.sort_values("Adjusted P-value", ascending=True)

        return {
            "title": title,
            "input_genes": clean_genes,
            "all_results": res,
            "sig_results": sig_res,
        }

    def run_from_df(
        self,
        df: pd.DataFrame,
        gene_col: str = "Gene",
        score_col: str = "MeanAbsDelta",
        top_n: int = 150,
        ascending: bool = False,
        min_freq_col: str | None = None,
        min_freq: float | None = None,
        extra_query: str | None = None,
        absolute_score: bool = False,
        title: str | None = None,
    ) -> dict[str, Any]:
        top_genes, top_df = self.extract_top_genes(
            df=df,
            gene_col=gene_col,
            score_col=score_col,
            top_n=top_n,
            ascending=ascending,
            min_freq_col=min_freq_col,
            min_freq=min_freq,
            extra_query=extra_query,
            absolute_score=absolute_score,
        )
        enrich_res = self.run(gene_list=top_genes, title=title)
        enrich_res["top_gene_table"] = top_df
        return enrich_res

    def save_results(
        self,
        enrich_res: dict[str, Any],
        filename: str = "significant_enrichment_results.csv",
    ) -> str:
        if "sig_results" not in enrich_res:
            raise ValueError("enrich_res must contain 'sig_results'")
        save_path = Path(self.cfg.output_dir) / filename
        enrich_res["sig_results"].to_csv(save_path, index=False)
        return str(save_path)


def run_cluster_enrichment_from_df(
    runner: PathwayEnrichmentRunner,
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    gene_col: str = "Gene",
    score_col: str = "Abs_Mean_IG_to_PV",
    top_n: int = 100,
    ascending: bool = False,
    min_freq_col: str | None = "Frequency",
    min_freq: float | None = 0.05,
    extra_query: str | None = None,
    absolute_score: bool = False,
    cluster_order: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Run enrichment per cluster from a cluster-level scored gene table."""
    if cluster_col not in df.columns:
        raise ValueError(f"Missing cluster column: {cluster_col}")

    work_df = df.copy()
    work_df[cluster_col] = work_df[cluster_col].astype(str).str.strip()
    clusters = (
        [str(x) for x in cluster_order]
        if cluster_order is not None
        else sorted(work_df[cluster_col].dropna().unique().tolist())
    )

    all_rows: list[pd.DataFrame] = []
    cluster_gene_dict: dict[str, list[str]] = {}

    for cl in clusters:
        sub = work_df[work_df[cluster_col] == str(cl)].copy()
        if sub.empty:
            continue

        top_genes, _top_df = runner.extract_top_genes(
            df=sub,
            gene_col=gene_col,
            score_col=score_col,
            top_n=top_n,
            ascending=ascending,
            min_freq_col=min_freq_col,
            min_freq=min_freq,
            extra_query=extra_query,
            absolute_score=absolute_score,
        )
        cluster_gene_dict[str(cl)] = top_genes

        if verbose:
            print(f"Cluster {cl}: {len(top_genes)} genes for enrichment")

        if len(top_genes) < runner.min_gene_n:
            if verbose:
                print(f"Skip cluster {cl}: too few genes after filtering")
            continue

        try:
            enrich_res = runner.run(
                gene_list=top_genes,
                title=f"{runner.gene_sets} enrichment: cluster {cl}",
            )
        except Exception as e:
            if verbose:
                print(f"Skip cluster {cl}: enrichment failed ({e})")
            continue

        sig_res = enrich_res.get("sig_results", pd.DataFrame()).copy()
        if sig_res.empty:
            continue
        sig_res[cluster_col] = str(cl)
        sig_res["TopN_Genes_Used"] = len(top_genes)
        all_rows.append(sig_res)

    if not all_rows:
        return pd.DataFrame(), cluster_gene_dict
    return pd.concat(all_rows, axis=0, ignore_index=True), cluster_gene_dict

