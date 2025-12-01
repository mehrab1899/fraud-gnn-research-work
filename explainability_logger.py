"""
Explainability Logging Module for Fraud Detection GNN Thesis

This module provides structured logging for two-level interpretability:
1. GRAPH-LEVEL EXPLANATIONS: Structural importance (which edges/relationships matter)
2. FEATURE-LEVEL EXPLANATIONS: Financial importance (which features matter)

Usage:
    logger = ExplainabilityLogger(output_dir="./explanations", experiment_name="run_1")
    logger.log_graph_explanation(node_id, edge_mask, top_edges)
    logger.log_feature_explanation(transaction_id, shap_values, feature_names)
    logger.save_summary_report()
"""

import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class ExplainabilityLogger:
    """
    Centralized logging system for fraud detection model explainability.
    Stores graph-level (structural) and feature-level (financial) explanations.
    """

    def __init__(
        self, output_dir: str = "./explanations", experiment_name: str = "default"
    ):
        """
        Initialize the logger.

        Args:
            output_dir: Root directory for all explanation outputs
            experiment_name: Name of the experiment (e.g., "run_1", "production_v1")
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment directory
        self.exp_dir = os.path.join(output_dir, f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Create subdirectories for different explanation types
        self.graph_dir = os.path.join(self.exp_dir, "graph_explanations")
        self.feature_dir = os.path.join(self.exp_dir, "feature_explanations")
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

        # Storage for aggregated data
        self.graph_explanations = []  # List of dicts
        self.feature_explanations = []  # List of dicts
        self.model_metadata = {}

        print(f"[OK] ExplainabilityLogger initialized: {self.exp_dir}")

    def log_model_metadata(self, metadata: Dict) -> None:
        """
        Log model configuration and training parameters.

        Args:
            metadata: Dict with keys like hidden_dim, epochs, loss_type, device, etc.
        """
        self.model_metadata = metadata
        metadata_file = os.path.join(self.exp_dir, "model_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  [OK] Model metadata saved to: {metadata_file}")

    def log_graph_explanation(
        self,
        node_id: int,
        edge_mask: np.ndarray,
        edge_index: Tuple[List, List],
        label: str = "fraud",
        top_k: int = 20,
    ) -> None:
        """
        Log a graph-level explanation (structural/relational importance).

        This explains WHICH EDGES in the transaction graph influenced predictions.

        Args:
            node_id: Transaction node ID being explained
            edge_mask: Edge importance scores (1D array, one per edge)
            edge_index: Tuple of (source_nodes, target_nodes) for all edges
            label: Classification label (e.g., "fraud", "normal")
            top_k: Number of top edges to record
        """
        edge_mask_np = (
            edge_mask.cpu().numpy() if hasattr(edge_mask, "cpu") else edge_mask
        )
        edge_mask_np = edge_mask_np.flatten()

        # Find top-k important edges
        if edge_mask_np.size > 0:
            top_indices = np.argsort(edge_mask_np)[-top_k:][::-1]
            top_edges = [
                {
                    "src": int(edge_index[0][i]),
                    "dst": int(edge_index[1][i]),
                    "importance": float(edge_mask_np[i]),
                }
                for i in top_indices
            ]
        else:
            top_edges = []

        explanation = {
            "timestamp": datetime.now().isoformat(),
            "node_id": int(node_id),
            "label": label,
            "num_edges": len(edge_mask_np),
            "top_k": top_k,
            "top_edges": top_edges,
            "max_importance": float(edge_mask_np.max()) if edge_mask_np.size > 0 else 0,
            "min_importance": float(edge_mask_np.min()) if edge_mask_np.size > 0 else 0,
            "mean_importance": (
                float(edge_mask_np.mean()) if edge_mask_np.size > 0 else 0
            ),
        }

        self.graph_explanations.append(explanation)

        # Save individual explanation file
        node_file = os.path.join(self.graph_dir, f"node_{node_id}_explanation.json")
        with open(node_file, "w") as f:
            json.dump(explanation, f, indent=2)

        print(
            f"  [OK] Graph explanation logged for node {node_id} | Top-{top_k} edges importance: {top_edges[0]['importance']:.4f}"
        )

    def log_feature_explanation(
        self,
        transaction_id: int,
        shap_values: Optional[np.ndarray],
        feature_names: List[str],
        feature_values: Optional[np.ndarray],
        base_value: Optional[float] = None,
        label: str = "fraud",
        top_k: int = 10,
    ) -> None:
        """
        Log a feature-level explanation (financial importance).

        This explains WHICH FEATURES drove the fraud prediction.

        Args:
            transaction_id: Transaction ID being explained
            shap_values: SHAP values for each feature (1D array)
            feature_names: Names of all features
            feature_values: Actual feature values for this transaction
            base_value: SHAP base value (expected model output)
            label: Classification label (e.g., "fraud", "normal")
            top_k: Number of top features to record
        """
        if shap_values is None:
            print(
                f"  ⚠ SHAP values unavailable for transaction {transaction_id}, skipping feature explanation"
            )
            return

        shap_values_np = shap_values.flatten()
        feature_values_np = (
            feature_values.flatten() if feature_values is not None else None
        )

        # Find top-k important features (by absolute SHAP value)
        feature_importance = np.abs(shap_values_np)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]

        top_features = []
        for idx in top_indices:
            feature_info = {
                "name": (
                    feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                ),
                "shap_value": float(shap_values_np[idx]),
                "abs_shap": float(feature_importance[idx]),
                "feature_value": (
                    float(feature_values_np[idx])
                    if feature_values_np is not None
                    else None
                ),
                "direction": (
                    "increases_fraud" if shap_values_np[idx] > 0 else "decreases_fraud"
                ),
            }
            top_features.append(feature_info)

        # Separate features by direction (fraud-pushing vs fraud-reducing)
        fraud_pushing = [f for f in top_features if f["direction"] == "increases_fraud"]
        fraud_reducing = [
            f for f in top_features if f["direction"] == "decreases_fraud"
        ]

        explanation = {
            "timestamp": datetime.now().isoformat(),
            "transaction_id": int(transaction_id),
            "label": label,
            "base_value": base_value,
            "num_features": len(feature_names),
            "top_k": top_k,
            "fraud_pushing_features": fraud_pushing,
            "fraud_reducing_features": fraud_reducing,
            "max_shap_positive": float(
                max([f["shap_value"] for f in fraud_pushing], default=0)
            ),
            "max_shap_negative": float(
                min([f["shap_value"] for f in fraud_reducing], default=0)
            ),
        }

        self.feature_explanations.append(explanation)

        # Save individual explanation file
        txn_file = os.path.join(
            self.feature_dir, f"txn_{transaction_id}_explanation.json"
        )
        with open(txn_file, "w") as f:
            json.dump(explanation, f, indent=2)

        print(
            f"  [OK] Feature explanation logged for txn {transaction_id} | Fraud-pushing: {len(fraud_pushing)}, Fraud-reducing: {len(fraud_reducing)}"
        )

    def save_aggregated_csv(self) -> None:
        """
        Save aggregated graph and feature explanations to CSV for easy thesis analysis.
        """
        # Graph explanations CSV
        if self.graph_explanations:
            graph_csv = os.path.join(self.exp_dir, "graph_explanations_summary.csv")
            with open(graph_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "node_id",
                        "label",
                        "num_edges",
                        "top_edge_importance",
                        "mean_importance",
                    ],
                )
                writer.writeheader()
                for exp in self.graph_explanations:
                    writer.writerow(
                        {
                            "timestamp": exp["timestamp"],
                            "node_id": exp["node_id"],
                            "label": exp["label"],
                            "num_edges": exp["num_edges"],
                            "top_edge_importance": (
                                exp["top_edges"][0]["importance"]
                                if exp["top_edges"]
                                else 0
                            ),
                            "mean_importance": exp["mean_importance"],
                        }
                    )
            print(f"  [OK] Graph explanations CSV: {graph_csv}")

        # Feature explanations CSV
        if self.feature_explanations:
            feature_csv = os.path.join(self.exp_dir, "feature_explanations_summary.csv")
            with open(feature_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "transaction_id",
                        "label",
                        "fraud_pushing_count",
                        "fraud_reducing_count",
                        "max_fraud_push_shap",
                    ],
                )
                writer.writeheader()
                for exp in self.feature_explanations:
                    writer.writerow(
                        {
                            "timestamp": exp["timestamp"],
                            "transaction_id": exp["transaction_id"],
                            "label": exp["label"],
                            "fraud_pushing_count": len(exp["fraud_pushing_features"]),
                            "fraud_reducing_count": len(exp["fraud_reducing_features"]),
                            "max_fraud_push_shap": exp["max_shap_positive"],
                        }
                    )
            print(f"  [OK] Feature explanations CSV: {feature_csv}")

    def save_summary_report(self) -> None:
        """
        Generate a comprehensive markdown report summarizing all explanations.
        Suitable for thesis documentation.
        """
        report_file = os.path.join(self.exp_dir, "EXPLAINABILITY_REPORT.md")

        with open(report_file, "w") as f:
            f.write("# Fraud Detection Model: Explainability Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Experiment:** {self.experiment_name}\n\n")

            # Model metadata section
            f.write("## 1. Model Configuration\n\n")
            if self.model_metadata:
                f.write("```json\n")
                f.write(json.dumps(self.model_metadata, indent=2))
                f.write("\n```\n\n")

            # Graph-level explanations section
            f.write("## 2. Graph-Level Explanations (Structural Importance)\n\n")
            f.write(
                "These explanations show **which edges/relationships** in the transaction graph were important for predictions.\n\n"
            )
            f.write("### Purpose\n")
            f.write(
                "- Understand relational patterns (who transacted with whom, in what order)\n"
            )
            f.write(
                "- Validate that the GNN captures meaningful transaction sequences\n"
            )
            f.write("- Identify suspicious transaction chains\n\n")

            if self.graph_explanations:
                f.write(f"### Summary Statistics\n\n")
                f.write(f"- **Total explanations:** {len(self.graph_explanations)}\n")
                mean_max_importance = np.mean(
                    [e["max_importance"] for e in self.graph_explanations]
                )
                f.write(
                    f"- **Mean max edge importance:** {mean_max_importance:.4f}\n\n"
                )

                f.write("### Example Explanations\n\n")
                for exp in self.graph_explanations[:5]:  # Show first 5
                    f.write(f"#### Node {exp['node_id']} (Label: {exp['label']})\n\n")
                    f.write("**Top Important Edges:**\n\n")
                    f.write("| Source | Destination | Importance |\n")
                    f.write("|--------|-------------|------------|\n")
                    for edge in exp["top_edges"][:5]:
                        f.write(
                            f"| {edge['src']} | {edge['dst']} | {edge['importance']:.4f} |\n"
                        )
                    f.write("\n")
            else:
                f.write("⚠ No graph explanations available.\n\n")

            # Feature-level explanations section
            f.write("## 3. Feature-Level Explanations (Financial Importance)\n\n")
            f.write(
                "These explanations show **which features** drove the fraud predictions.\n\n"
            )
            f.write("### Purpose\n")
            f.write(
                "- Understand financial patterns (amount, velocity, location anomalies)\n"
            )
            f.write("- Validate domain knowledge (expected features matter)\n")
            f.write("- Build trust in model predictions\n\n")

            if self.feature_explanations:
                f.write(f"### Summary Statistics\n\n")
                f.write(f"- **Total explanations:** {len(self.feature_explanations)}\n")
                avg_fraud_pushing = np.mean(
                    [
                        len(e["fraud_pushing_features"])
                        for e in self.feature_explanations
                    ]
                )
                f.write(
                    f"- **Average fraud-pushing features per transaction:** {avg_fraud_pushing:.1f}\n\n"
                )

                f.write("### Example Explanations\n\n")
                for exp in self.feature_explanations[:5]:  # Show first 5
                    f.write(
                        f"#### Transaction {exp['transaction_id']} (Label: {exp['label']})\n\n"
                    )
                    f.write("**Fraud-Pushing Features:**\n\n")
                    if exp["fraud_pushing_features"]:
                        f.write("| Feature | Value | SHAP Impact |\n")
                        f.write("|---------|-------|-------------|\n")
                        for feat in exp["fraud_pushing_features"][:5]:
                            f.write(
                                f"| {feat['name']} | {feat['feature_value']:.4f} | {feat['shap_value']:+.4f} |\n"
                            )
                    else:
                        f.write("*(None)*\n")
                    f.write("\n")

                    f.write("**Fraud-Reducing Features:**\n\n")
                    if exp["fraud_reducing_features"]:
                        f.write("| Feature | Value | SHAP Impact |\n")
                        f.write("|---------|-------|-------------|\n")
                        for feat in exp["fraud_reducing_features"][:5]:
                            f.write(
                                f"| {feat['name']} | {feat['feature_value']:.4f} | {feat['shap_value']:+.4f} |\n"
                            )
                    else:
                        f.write("*(None)*\n")
                    f.write("\n")
            else:
                f.write("⚠ No feature explanations available.\n\n")

            # Thesis integration guidance
            f.write("## 4. Thesis Integration Guide\n\n")
            f.write(
                """
### For Chapter: Model Interpretability & Explainability

**Graph-Level (Structural) Findings:**
- Present example explanations showing important transaction chains
- Discuss how the GNN learns relational patterns
- Compare edge importance patterns between fraudulent and normal transactions
- Validate that discovered patterns match domain expectations

**Feature-Level (Financial) Findings:**
- Show which features have highest SHAP impact on predictions
- Discuss whether model respects financial domain knowledge
- Identify unexpected feature combinations that trigger fraud flags
- Compare feature importance across fraud vs. normal transactions

**Combined Narrative:**
- Graph-level explains *relational context* (the web of transactions)
- Feature-level explains *transaction properties* (amounts, anomalies)
- Together = complete interpretability of fraud detection decisions
"""
            )

        print(f"  [OK] Summary report saved: {report_file}")

    def get_statistics(self) -> Dict:
        """Return aggregated statistics for quick reference."""
        stats = {
            "num_graph_explanations": len(self.graph_explanations),
            "num_feature_explanations": len(self.feature_explanations),
            "avg_top_edge_importance": np.mean(
                [
                    e["top_edges"][0]["importance"]
                    for e in self.graph_explanations
                    if e["top_edges"]
                ]
            ),
            "total_features_analyzed": sum(
                [e["num_features"] for e in self.feature_explanations]
            ),
        }
        return stats
