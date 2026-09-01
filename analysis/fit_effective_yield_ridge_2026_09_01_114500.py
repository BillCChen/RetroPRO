"""Fit a target-grouped structural ridge baseline for fragment yield."""

import argparse
import glob
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr

from mlp_retrosyn.css_effective_yield import enumerate_effective_yield_candidates


FEATURES = (
    "fragment_atoms",
    "atom_fraction",
    "atom_fraction_squared",
    "fragment_bonds",
    "bond_fraction",
    "rings",
    "aromatic_fraction",
    "hetero_fraction",
    "formal_charge_abs",
    "rotatable_bonds",
    "molecular_weight",
    "tpsa",
    "family_r3",
    "family_r7",
    "family_pair",
    "family_triple",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=10.0)
    return parser.parse_args()


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def family_map(target):
    output = {}
    for record in enumerate_effective_yield_candidates(target, max_triples=256):
        output[canonical(record["smiles"])] = set(record["families"])
    return output


def structural_features(target, fragment, families):
    target_mol = Chem.MolFromSmiles(target)
    fragment_mol = Chem.MolFromSmiles(fragment)
    target_atoms = max(target_mol.GetNumAtoms(), 1)
    target_bonds = max(target_mol.GetNumBonds(), 1)
    atoms = fragment_mol.GetNumAtoms()
    bonds = fragment_mol.GetNumBonds()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in fragment_mol.GetAtoms())
    hetero_atoms = sum(atom.GetAtomicNum() not in (1, 6) for atom in fragment_mol.GetAtoms())
    atom_fraction = atoms / float(target_atoms)
    values = (
        atoms,
        atom_fraction,
        atom_fraction * atom_fraction,
        bonds,
        bonds / float(target_bonds),
        rdMolDescriptors.CalcNumRings(fragment_mol),
        aromatic_atoms / float(max(atoms, 1)),
        hetero_atoms / float(max(atoms, 1)),
        abs(sum(atom.GetFormalCharge() for atom in fragment_mol.GetAtoms())),
        rdMolDescriptors.CalcNumRotatableBonds(fragment_mol),
        Descriptors.MolWt(fragment_mol),
        rdMolDescriptors.CalcTPSA(fragment_mol),
        int("r3" in families),
        int("r7" in families),
        int("pair" in families),
        int("triple" in families),
    )
    return np.asarray(values, dtype=float)


def load_rows(directory):
    raw = []
    for path in sorted(glob.glob(str(directory / "*_fragment_yield_*.jsonl"))):
        arm = Path(path).name.split("_fragment_yield_", 1)[0]
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                row["arm"] = arm
                row["fragment"] = canonical(row["fragment"])
                raw.append(row)

    grouped = defaultdict(list)
    arm_membership = defaultdict(set)
    for row in raw:
        key = (int(row["task_id"]), row["target"], row["fragment"])
        grouped[key].append(float(row["full_product_unique_reaction_count"]))
        arm_membership[key].add(row["arm"])

    family_cache = {}
    rows = []
    for key, labels in grouped.items():
        target_id, target, fragment = key
        if target_id not in family_cache:
            family_cache[target_id] = family_map(target)
        families = family_cache[target_id].get(fragment, set())
        rows.append({
            "target_id": target_id,
            "target": target,
            "fragment": fragment,
            "families": sorted(families),
            "arms": sorted(arm_membership[key]),
            "label": sum(labels) / len(labels),
            "observation_count": len(labels),
            "features": structural_features(target, fragment, families),
        })
    return rows


def fit_ridge(x, y, ridge):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T.dot(design) + penalty,
        design.T.dot(y),
    )
    return mean, scale, coefficients


def predict(x, model):
    mean, scale, coefficients = model
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    return np.maximum(design.dot(coefficients), 0.0)


def pairwise_accuracy(rows, predictions):
    correct = 0.0
    total = 0
    by_target = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_target[row["target_id"]].append((row["label"], prediction))
    for values in by_target.values():
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                left_y, left_p = values[i]
                right_y, right_p = values[j]
                if left_y == right_y:
                    continue
                total += 1
                delta = (left_y - right_y) * (left_p - right_p)
                correct += 1.0 if delta > 0 else (0.5 if delta == 0 else 0.0)
    return correct / total if total else None


def topk_capture(rows, predictions, topk=8):
    by_target = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_target[row["target_id"]].append((row["label"], prediction))
    metrics = []
    for target_id, values in sorted(by_target.items()):
        selected = sorted(values, key=lambda item: item[1], reverse=True)[:topk]
        oracle = sorted(values, key=lambda item: item[0], reverse=True)[:topk]
        selected_credit = sum(item[0] for item in selected)
        oracle_credit = sum(item[0] for item in oracle)
        metrics.append({
            "target_id": target_id,
            "candidate_count": len(values),
            "predicted_topk_credit": selected_credit,
            "oracle_topk_credit": oracle_credit,
            "capture": selected_credit / oracle_credit if oracle_credit else None,
        })
    return metrics


def main():
    args = parse_args()
    telemetry_dir = Path(args.telemetry_dir).resolve()
    output_path = Path(args.output).resolve()
    rows = load_rows(telemetry_dir)
    targets = sorted({row["target_id"] for row in rows})
    predictions = np.zeros(len(rows), dtype=float)
    fold_records = []
    for fold in range(args.folds):
        test_targets = {target for target in targets if target % args.folds == fold}
        train_indices = [
            index for index, row in enumerate(rows)
            if row["target_id"] not in test_targets
        ]
        test_indices = [
            index for index, row in enumerate(rows)
            if row["target_id"] in test_targets
        ]
        x_train = np.vstack([rows[index]["features"] for index in train_indices])
        y_train = np.asarray([rows[index]["label"] for index in train_indices])
        x_test = np.vstack([rows[index]["features"] for index in test_indices])
        model = fit_ridge(x_train, y_train, args.ridge)
        predictions[test_indices] = predict(x_test, model)
        fold_records.append({
            "fold": fold,
            "train_target_count": len(set(targets) - test_targets),
            "test_targets": sorted(test_targets),
            "train_row_count": len(train_indices),
            "test_row_count": len(test_indices),
        })

    labels = np.asarray([row["label"] for row in rows])
    spearman = spearmanr(labels, predictions)
    per_target_spearman = []
    for target_id in targets:
        indices = [
            index for index, row in enumerate(rows) if row["target_id"] == target_id
        ]
        value = spearmanr(labels[indices], predictions[indices]).statistic
        if not math.isnan(value):
            per_target_spearman.append(value)
    capture = topk_capture(rows, predictions)

    full_x = np.vstack([row["features"] for row in rows])
    full_model = fit_ridge(full_x, labels, args.ridge)
    report = {
        "as_of": datetime.now().astimezone().isoformat(),
        "analysis_version": "1.0.0",
        "protocol": {
            "telemetry_dir": str(telemetry_dir),
            "folds": args.folds,
            "fold_rule": "target_id modulo folds",
            "ridge": args.ridge,
            "features": FEATURES,
            "label": "mean full_product_unique_reaction_count per target-fragment",
            "candidate_boundary": "union of fragments observed by six development arms",
        },
        "data": {
            "target_count": len(targets),
            "unique_target_fragment_rows": len(rows),
            "positive_row_count": int((labels > 0).sum()),
        },
        "cross_validation": {
            "folds": fold_records,
            "mae": float(np.mean(np.abs(labels - predictions))),
            "spearman": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "mean_target_spearman": float(np.mean(per_target_spearman)),
            "pairwise_accuracy": pairwise_accuracy(rows, predictions),
            "mean_top8_oracle_capture": float(np.mean([
                row["capture"] for row in capture if row["capture"] is not None
            ])),
            "top8_capture_by_target": capture,
        },
        "full_fit": {
            "feature_mean": full_model[0].tolist(),
            "feature_scale": full_model[1].tolist(),
            "coefficients_with_intercept": full_model[2].tolist(),
        },
        "interpretation_boundary": (
            "Target-grouped development diagnostic over observed fragments; "
            "not a prospective route-success estimate."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "data": report["data"],
        "cross_validation": {
            key: value for key, value in report["cross_validation"].items()
            if key != "top8_capture_by_target"
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
