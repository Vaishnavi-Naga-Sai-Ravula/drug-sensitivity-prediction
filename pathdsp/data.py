from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE_ARCHIVE = ROOT / "app_data" / "pathdsp_core.zip"

DATA_FILES = {
    "GDSC": {
        "response": "GDSC/drug_response/GDSC.RESP.negLog10IC50.txt",
        "drug_chem": "GDSC/morgan_fingerprint/GDSC.CHEM.MF.txt",
        "drug_pathway": "GDSC/pathway_enrichment/DGNet/GDSC.DGNet.NetPEA.txt",
        "cell_exp": "GDSC/pathway_enrichment/EXP/GDSC.EXP.ssGSEA.txt",
        "cell_mut": "GDSC/pathway_enrichment/MUT/GDSC.MUT.NetPEA.txt",
        "cell_cnv": "GDSC/pathway_enrichment/CNV/GDSC.CNV.NetPEA.txt",
    },
    "CCLE": {
        "response": "CCLE/drug_response/CCLE.RESP.negLog10IC50.txt",
        "drug_chem": "CCLE/morgan_fingerprint/CCLE..CHEM.MF.txt",
        "drug_pathway": "CCLE/pathway_enrichment/DGNet/CCLE.DGNet.NetPEA.txt",
        "cell_exp": "CCLE/pathway_enrichment/EXP/CCLE.EXP.ssGSEA.txt",
        "cell_mut": "CCLE/pathway_enrichment/MUT/CCLE.MUT.NetPEA.txt",
        "cell_cnv": "CCLE/pathway_enrichment/CNV/CCLE.CNV.NetPEA.txt",
    },
}

FEATURE_SOURCES = (
    ("CHEM", "drug_chem", "drug"),
    ("DGNet", "drug_pathway", "drug"),
    ("EXP", "cell_exp", "cell"),
    ("MUT-Net", "cell_mut", "cell"),
    ("CNV-Net", "cell_cnv", "cell"),
)


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray | None
    y: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]
    feature_groups: dict[str, list[int]]
    drug_features: np.ndarray
    cell_features: np.ndarray
    drug_ids: list[str]
    cell_ids: list[str]
    drug_index: dict[str, int]
    cell_index: dict[str, int]
    response: pd.DataFrame

    def pair_features(self, drug: str, cell: str) -> np.ndarray:
        dkey, ckey = normalize_id(drug), normalize_id(cell)
        if dkey not in self.drug_index:
            raise KeyError(f"Drug '{drug}' does not have all required PathDSP features in {self.name}.")
        if ckey not in self.cell_index:
            raise KeyError(f"Cell line '{cell}' does not have all required PathDSP features in {self.name}.")
        drow = self.drug_features[self.drug_index[dkey]]
        crow = self.cell_features[self.cell_index[ckey]]
        return np.concatenate((drow, crow)).astype(np.float32, copy=False)

    def observed_response(self, drug: str, cell: str) -> float | None:
        dkey, ckey = normalize_id(drug), normalize_id(cell)
        row_map = {normalize_id(v): i for i, v in enumerate(self.response.index)}
        col_map = {normalize_id(v): i for i, v in enumerate(self.response.columns)}
        if dkey not in row_map or ckey not in col_map:
            return None
        value = self.response.iat[row_map[dkey], col_map[ckey]]
        return float(value) if pd.notna(value) else None


def normalize_id(value: object) -> str:
    return str(value).strip().casefold()


def _open_table(relative_path: str) -> BinaryIO:
    # A single versioned source prevents stale local tables changing predictions.
    if CORE_ARCHIVE.is_file():
        with zipfile.ZipFile(CORE_ARCHIVE) as archive:
            member = "processed_data/" + relative_path
            try:
                return BytesIO(archive.read(member))
            except KeyError as exc:
                raise FileNotFoundError(f"{relative_path} is missing from {CORE_ARCHIVE}") from exc
    raise FileNotFoundError(
        "Bundled PathDSP data were not found. Restore app_data/pathdsp_core.zip from the repository."
    )


def read_matrix(relative_path: str) -> pd.DataFrame:
    with _open_table(relative_path) as stream:
        frame = pd.read_csv(stream, sep="\t", index_col=0, low_memory=False)
    frame.index = frame.index.astype(str).str.strip()
    frame.columns = frame.columns.astype(str).str.strip()
    if pd.Index(frame.index.str.casefold()).duplicated().any() or frame.columns.duplicated().any():
        raise ValueError(f"Duplicate identifiers or feature columns in {relative_path}.")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if (frame.notna() & numeric.isna()).any().any() or np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"Nonnumeric or infinite values in {relative_path}.")
    return numeric


def _combine_features(tables: dict[str, pd.DataFrame], domain: str) -> tuple[np.ndarray, list[str], dict[str, list[int]], list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[int]] = {}
    ids: list[str] | None = None
    for label, source_key, source_domain in FEATURE_SOURCES:
        if source_domain != domain:
            continue
        table = tables[source_key]
        if ids is None:
            ids = list(table.index)
        else:
            common = set(map(normalize_id, ids)) & set(map(normalize_id, table.index))
            ids = [v for v in ids if normalize_id(v) in common]
        prefix = label.replace("-", "")
        start = len(names)
        names.extend(f"{prefix}__{col}" for col in table.columns)
        groups[label] = list(range(start, len(names)))
        blocks.append(table)
    assert ids is not None

    # Align every block to the same ordered identifiers after intersection.
    aligned: list[np.ndarray] = []
    for (_, source_key, source_domain), _ in zip(
        [spec for spec in FEATURE_SOURCES if spec[2] == domain], blocks
    ):
        table = tables[source_key]
        row_map = {normalize_id(v): i for i, v in enumerate(table.index)}
        aligned.append(table.to_numpy(dtype=np.float32)[[row_map[normalize_id(v)] for v in ids]])
    return np.concatenate(aligned, axis=1), names, groups, ids


def build_dataset(name: str = "GDSC", *, include_examples: bool = True) -> DatasetBundle:
    """Join the supplied PathDSP matrices into drug-cell examples."""
    dataset_name = name.upper()
    if dataset_name not in DATA_FILES:
        raise ValueError(f"Dataset must be one of {', '.join(DATA_FILES)}")
    files = DATA_FILES[dataset_name]
    tables = {key: read_matrix(path) for key, path in files.items()}

    drug_features, drug_names, drug_groups, drug_ids = _combine_features(tables, "drug")
    cell_features, cell_names, cell_groups, cell_ids = _combine_features(tables, "cell")
    if drug_names + cell_names != list(dict.fromkeys(drug_names + cell_names)):
        raise ValueError("Feature names are duplicated after grouping; resolve naming collisions before training.")

    feature_names = drug_names + cell_names
    groups: dict[str, list[int]] = dict(drug_groups)
    for label, indices in cell_groups.items():
        groups[label] = [i + len(drug_names) for i in indices]

    response = tables["response"]
    response.index.name = "drug"
    response.columns.name = "cell"
    target = response.stack(future_stack=True).dropna().rename("target").reset_index()
    target.columns = ["drug", "cell", "target"]

    drug_lookup = {normalize_id(v): i for i, v in enumerate(drug_ids)}
    cell_lookup = {normalize_id(v): i for i, v in enumerate(cell_ids)}
    target["drug_row"] = target["drug"].map(lambda v: drug_lookup.get(normalize_id(v)))
    target["cell_row"] = target["cell"].map(lambda v: cell_lookup.get(normalize_id(v)))
    target["target"] = pd.to_numeric(target["target"], errors="coerce")
    target = target.dropna(subset=["drug_row", "cell_row", "target"])
    target = target[np.isfinite(target["target"].to_numpy(dtype=float))].reset_index(drop=True)

    drug_rows = target["drug_row"].to_numpy(dtype=np.int64)
    cell_rows = target["cell_row"].to_numpy(dtype=np.int64)
    X = None
    if include_examples:
        X = np.concatenate((drug_features[drug_rows], cell_features[cell_rows]), axis=1).astype(np.float32, copy=False)
    y = target["target"].to_numpy(dtype=np.float32)
    meta = target[["drug", "cell"]].copy()
    meta["dataset"] = dataset_name

    return DatasetBundle(
        name=dataset_name,
        X=X,
        y=y,
        meta=meta,
        feature_names=feature_names,
        feature_groups=groups,
        drug_features=drug_features,
        cell_features=cell_features,
        drug_ids=drug_ids,
        cell_ids=cell_ids,
        drug_index=drug_lookup,
        cell_index=cell_lookup,
        response=response,
    )

