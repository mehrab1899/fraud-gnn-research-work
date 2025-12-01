import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import networkx as nx
import itertools

# Import explainability logger
from explainability_logger import ExplainabilityLogger

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 16
MAX_TRX_PER_COMPANY = 8
LIMIT_PER_ENTITY = 8
MAKE_CLIQUES = False
EPOCHS = 8
LR = 0.001
WEIGHT_DECAY = 1e-4
# Quick-run flags to speed up prototyping and reduce resource usage
QUICK_RUN = True
QUICK_RUN_SUBSAMPLE = 0.05  # fraction of transactions to keep when QUICK_RUN=True
QUICK_SHAP_SAMPLES = 10  # reduce SHAP background samples for quick runs

# Explainability logging setup
EXPLANATION_OUTPUT_DIR = r"D:\Thesis\explanations"
EXPERIMENT_NAME = "fraud_detection_gnn"

print(f"Device: {DEVICE}")

# Initialize explainability logger
try:
    explainability_logger = ExplainabilityLogger(
        output_dir=EXPLANATION_OUTPUT_DIR, experiment_name=EXPERIMENT_NAME
    )
except Exception as e:
    print(f"[WARN] ExplainabilityLogger initialization failed: {e}")
    explainability_logger = None

# ============================================================================
# 1. LOAD AND PREPROCESS DATASETS
# ============================================================================

# Dataset 1: Financial Fraud Detection
file_path_1 = r"D:\Thesis\Dataset\financial_fraud_detection_dataset.csv"
df = pd.read_csv(file_path_1)
print("Dataset 1 loaded:", df.shape)

# Handle missing values
df["fraud_type"] = df["fraud_type"].fillna("unknown")
df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(
    df["time_since_last_transaction"].median()
)

# Categorical encoding
df_encoded = pd.get_dummies(
    df,
    columns=[
        "transaction_type",
        "merchant_category",
        "location",
        "device_used",
        "payment_channel",
    ],
    drop_first=True,
)

# Feature scaling
scaler = StandardScaler()
numerical_columns = [
    "amount",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# Dataset 2: German Credit Data
file_path_2 = r"D:\Thesis\Dataset\german_credit_data.csv"
df_loan = pd.read_csv(file_path_2)
print("Dataset 2 loaded:", df_loan.shape)

df_loan = df_loan.drop(columns=["Unnamed: 0"])
df_loan["Saving accounts"] = df_loan["Saving accounts"].fillna("unknown")
df_loan["Checking account"] = df_loan["Checking account"].fillna("unknown")

df_loan_encoded = pd.get_dummies(
    df_loan,
    columns=["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"],
    drop_first=True,
)
scaler = StandardScaler()
df_loan_encoded[["Credit amount", "Duration", "Age"]] = scaler.fit_transform(
    df_loan_encoded[["Credit amount", "Duration", "Age"]]
)

# Dataset 3: Credit Risk Data
file_path_3 = r"D:\Thesis\Dataset\nz_bank_loans_synthetic_with_dates.csv"
df_credit = pd.read_csv(file_path_3)
print("Dataset 3 loaded:", df_credit.shape)

df_credit["employment_status"].fillna("unknown", inplace=True)
df_credit["home_ownership"].fillna("unknown", inplace=True)
df_credit["stated_income"].fillna(df_credit["stated_income"].median(), inplace=True)

df_credit["application_date"] = pd.to_datetime(df_credit["application_date"])
df_credit["issue_date"] = pd.to_datetime(df_credit["issue_date"])
df_credit["first_payment_date"] = pd.to_datetime(df_credit["first_payment_date"])

df_credit["application_year"] = df_credit["application_date"].dt.year
df_credit["issue_year"] = df_credit["issue_date"].dt.year
df_credit["days_to_issue"] = (
    df_credit["issue_date"] - df_credit["application_date"]
).dt.days

df_credit_encoded = pd.get_dummies(
    df_credit,
    columns=["gender", "region", "employment_status", "home_ownership"],
    drop_first=True,
)
scaler = StandardScaler()
numerical_columns = [
    "age",
    "employment_length",
    "employer_tenure_years",
    "annual_income",
    "ip_risk_score",
    "income_mismatch_ratio",
    "days_to_fund",
    "days_to_issue",
]
df_credit_encoded[numerical_columns] = scaler.fit_transform(
    df_credit_encoded[numerical_columns]
)

# Dataset 4: Transaction Data
file_path_4 = r"D:\Thesis\Dataset\PS_20174392719_1491204439457_log.csv"
df_transaction = pd.read_csv(file_path_4)
print("Dataset 4 loaded:", df_transaction.shape)

df_transaction["isFraud"].fillna(0, inplace=True)

# Quick-run subsampling to speed up development and reduce memory use
if QUICK_RUN and QUICK_RUN_SUBSAMPLE is not None and QUICK_RUN_SUBSAMPLE < 1.0:
    orig_rows = len(df_transaction)
    df_transaction = df_transaction.sample(
        frac=QUICK_RUN_SUBSAMPLE, random_state=42
    ).reset_index(drop=True)
    print(
        f"Quick-run subsampled transactions: {orig_rows} -> {len(df_transaction)} rows"
    )

encoder = LabelEncoder()
df_transaction["nameOrig"] = encoder.fit_transform(df_transaction["nameOrig"])
df_transaction["nameDest"] = encoder.fit_transform(df_transaction["nameDest"])

df_transaction_encoded = pd.get_dummies(
    df_transaction, columns=["type"], drop_first=True
)
scaler = StandardScaler()
numerical_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
df_transaction_encoded[numerical_columns] = scaler.fit_transform(
    df_transaction_encoded[numerical_columns]
)

# ============================================================================
# 2. BUILD HETEROGENEOUS GRAPH
# ============================================================================


def force_numeric(df):
    """Convert all columns to numeric and fill NaNs with 0."""
    df_copy = df.copy()
    for col in df_copy.columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    df_copy = df_copy.fillna(0)
    return df_copy.astype(np.float32)


company_feat = force_numeric(df_loan_encoded)
transaction_feat = force_numeric(df_transaction_encoded.drop(["isFraud"], axis=1))
loan_feat = force_numeric(df_credit_encoded)

hetero_data = HeteroData()
hetero_data["company"].x = torch.from_numpy(company_feat.values)
hetero_data["transaction"].x = torch.from_numpy(transaction_feat.values)
hetero_data["loan"].x = torch.from_numpy(loan_feat.values)

num_companies = hetero_data["company"].num_nodes
num_transactions = hetero_data["transaction"].num_nodes
num_loans = hetero_data["loan"].num_nodes

# Bipartite edges (company -> transaction)
hetero_data["company", "performs", "transaction"].edge_index = torch.stack(
    [
        torch.randint(0, num_companies, (num_transactions,)),
        torch.arange(num_transactions),
    ],
    dim=0,
)

hetero_data["transaction", "funds", "loan"].edge_index = torch.stack(
    [torch.arange(num_transactions), torch.randint(0, num_loans, (num_transactions,))],
    dim=0,
)

transaction_labels = torch.tensor(
    df_transaction_encoded["isFraud"].values, dtype=torch.long
)

train_ratio = 0.8
train_size = int(train_ratio * num_transactions)
perm = torch.randperm(num_transactions)
train_idx = perm[:train_size]
test_idx = perm[train_size:]

train_mask = torch.zeros(num_transactions, dtype=torch.bool)
train_mask[train_idx] = True

test_mask = torch.zeros(num_transactions, dtype=torch.bool)
test_mask[test_idx] = True

print(
    f"Graph created: {num_companies} companies, {num_transactions} transactions, {num_loans} loans"
)

# ============================================================================
# 3. CLASS IMBALANCE TECHNIQUES
# ============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard negative examples by down-weighting easy examples.
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def apply_smote_to_transactions(transaction_feat, transaction_labels, random_state=42):
    """
    Apply SMOTE oversampling to balance transaction dataset.
    Only applied to training set to avoid data leakage.
    """
    print(f"Class distribution before SMOTE:")
    print(f"  Non-fraud: {(transaction_labels == 0).sum()}")
    print(f"  Fraud: {(transaction_labels == 1).sum()}")

    try:
        # Only apply SMOTE if we have both classes and enough samples
        if (transaction_labels == 1).sum() > 1:
            smote = SMOTE(
                random_state=random_state,
                k_neighbors=min(5, (transaction_labels == 1).sum() - 1),
            )
            transaction_feat_balanced, transaction_labels_balanced = smote.fit_resample(
                transaction_feat, transaction_labels
            )
            print(f"\nClass distribution after SMOTE:")
            print(f"  Non-fraud: {(transaction_labels_balanced == 0).sum()}")
            print(f"  Fraud: {(transaction_labels_balanced == 1).sum()}")
            return (
                torch.from_numpy(transaction_feat_balanced).float(),
                torch.from_numpy(transaction_labels_balanced).long(),
            )
        else:
            print("Insufficient fraud samples for SMOTE, skipping.")
            return torch.from_numpy(transaction_feat).float(), transaction_labels
    except Exception as e:
        print(f"SMOTE failed ({e}), using original data.")
        return torch.from_numpy(transaction_feat).float(), transaction_labels


# ============================================================================
# 3. DEFINE MODEL COMPONENTS
# ============================================================================


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.view(-1, 1) * x[col])
        return out


class AttnBiLSTM(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.attn = nn.Linear(hidden, 1)

    def forward(self, seq_embs, lengths=None):
        H, _ = self.lstm(seq_embs)
        a = torch.softmax(self.attn(H).squeeze(-1), dim=-1)
        z = (a.unsqueeze(-1) * H).sum(dim=1)
        return z, a


class MultiFraudNet(nn.Module):
    def __init__(
        self,
        company_feat_dim,
        transaction_feat_dim,
        companies_index,
        transactions_index,
        hidden_dim=HIDDEN_DIM,
        edge_type_number=1,
    ):
        super().__init__()
        self.companies_index = companies_index
        self.transactions_index = transactions_index
        self.hidden_dim = hidden_dim
        self.edge_type_number = edge_type_number

        self.company_ffn = nn.Linear(company_feat_dim, hidden_dim)
        self.company_convs = nn.ModuleList(
            [GraphConvLayer(hidden_dim, hidden_dim) for _ in range(edge_type_number)]
        )
        self.company_post = nn.Linear(hidden_dim, hidden_dim)

        self.trans_ffn = nn.Linear(transaction_feat_dim, hidden_dim)
        self.trans_conv1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.trans_conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.trans_post = nn.Linear(hidden_dim, hidden_dim)

        self.attn_seq = AttnBiLSTM(hidden_dim, hidden=hidden_dim)
        self.fc_company = nn.Linear(hidden_dim, 2)
        self.fc_trans = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        company_x,
        company_edge_index_h,
        transaction_x,
        transaction_edge_index_h,
        companies_length=None,
        single=False,
    ):
        x_c = F.gelu(self.company_ffn(company_x))
        if len(self.company_convs) > 0:
            conv_outs = []
            x_tmp = x_c
            for conv in self.company_convs:
                h = F.gelu(conv(x_tmp, company_edge_index_h))
                x_tmp = x_tmp + h
                conv_outs.append(h)
            # Avoid creating an extra stacked tensor to reduce peak memory usage
            x_sum = x_c
            for h in conv_outs:
                x_sum = x_sum + h
            x_c = x_sum / (1 + len(conv_outs))
        x_c = F.gelu(self.company_post(x_c))

        x_t = F.gelu(self.trans_ffn(transaction_x))
        x1_t = F.gelu(self.trans_conv1(x_t, transaction_edge_index_h))
        x_t = x_t + x1_t
        x2_t = F.gelu(self.trans_conv2(x_t, transaction_edge_index_h))
        x_t = x_t + x2_t
        x_avg = (x_t + x1_t + x2_t) / 3.0
        x_t = F.gelu(self.trans_post(x_avg))

        seqs = []
        for comp in self.companies_index:
            valid_idxs = [i for i in comp if (i is not None) and (0 <= i < x_t.size(0))]
            if len(valid_idxs) == 0:
                seqs.append(torch.zeros(1, x_t.size(1), device=x_t.device))
            else:
                seqs.append(x_t[valid_idxs])
        padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        lengths = torch.tensor([len(s) for s in seqs], device=x_t.device)
        att_out, _ = self.attn_seq(padded, lengths)
        x_c_shared = x_c + att_out

        out_c = self.fc_company(x_c_shared)
        out_t = self.fc_trans(x_t)
        return out_c, out_t


class SimpleGNNExplainer:
    def __init__(self, model, epochs=200, lr=0.01):
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def explain_node(self, node_idx, x, edge_index):
        self.model.eval()
        num_edges = edge_index.size(1)
        edge_mask = torch.randn(num_edges, requires_grad=True, device=x.device)
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            x_proj = F.gelu(self.model.trans_ffn(x))
            masked_weight = torch.sigmoid(edge_mask)
            row, col = edge_index
            out = torch.zeros_like(x_proj)
            out.index_add_(0, row, masked_weight.view(-1, 1) * x_proj[col])
            pred = out[node_idx].mean()
            loss = -pred
            loss.backward()
            optimizer.step()

        return torch.sigmoid(edge_mask).detach()

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None, top_k=20):
        top_k = min(top_k, edge_mask.numel())
        threshold = edge_mask.topk(top_k).values.min().item()
        important = edge_mask >= threshold
        sub_edge_index = edge_index[:, important]

        G = nx.Graph()
        edges = sub_edge_index.cpu().T.numpy()
        G.add_edges_from([tuple(e) for e in edges])

        if node_idx not in G.nodes:
            G.add_node(node_idx)

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_color=["red" if n == node_idx else "gray" for n in G.nodes()],
            edge_color="blue",
            node_size=200,
            width=1.2,
        )
        plt.title(f"Explanation for Node {node_idx}")
        plt.show()


class SHAPExplainer:
    """
    SHAP-based feature-level explainability for fraud detection GNN.
    Explains which transaction features drive fraud predictions.
    """

    def __init__(self, model, transaction_features, feature_names, device="cpu"):
        self.model = model
        self.transaction_features = transaction_features
        self.feature_names = feature_names
        self.device = device
        self.model.eval()

    def predict_fn(self, x):
        """
        Wrapper function for SHAP that takes feature matrix and returns fraud probabilities.
        """
        x_tensor = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            logits = self.model.trans_ffn(x_tensor)  # Get raw embeddings
            # Get fraud probability from final layer
            proba = torch.softmax(
                self.model.fc_trans(F.gelu(logits)), dim=1
            )  # (batch, 2)
            return proba[:, 1].cpu().numpy()  # Return fraud probability only

    def explain_instance(self, instance_idx, num_samples=10):
        """
        Generate SHAP explanations for a specific transaction.

        Args:
            instance_idx: Index of transaction to explain
            num_samples: Number of background samples for SHAP

        Returns:
            shap_values: SHAP values for each feature
            base_value: Expected model output
        """
        if not HAS_SHAP:
            print("SHAP not available. Install with: pip install shap")
            return None, None

        try:
            # Use a sample of background data for faster computation
            background_indices = np.random.choice(
                len(self.transaction_features),
                min(num_samples, len(self.transaction_features)),
                replace=False,
            )
            background_data = self.transaction_features[background_indices]

            # Create SHAP explainer
            explainer = shap.KernelExplainer(
                self.predict_fn, background_data, link="identity"
            )

            # Get instance to explain
            instance = self.transaction_features[[instance_idx]]

            # Compute SHAP values
            shap_values = explainer.shap_values(instance, check_additivity=False)

            return shap_values, explainer.expected_value

        except Exception as e:
            print(f"Error generating SHAP values: {e}")
            return None, None

    def plot_explanation(self, instance_idx, shap_values, base_value, top_k=10):
        """
        Visualize SHAP explanation using force plot and bar plot.
        """
        if shap_values is None or not HAS_SHAP:
            print("Cannot plot: SHAP values not available")
            return

        try:
            # Get feature values for this instance
            instance_features = self.transaction_features[instance_idx]

            # Create force plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)

            # Bar plot of top features
            feature_importance = np.abs(shap_values[0])
            top_indices = np.argsort(feature_importance)[-top_k:]

            colors = ["red" if shap_values[0, i] > 0 else "blue" for i in top_indices]
            plt.barh(
                [self.feature_names[i] for i in top_indices],
                shap_values[0, top_indices],
                color=colors,
            )
            plt.xlabel("SHAP Value (Impact on Fraud Prediction)")
            plt.title(f"Top {top_k} Features Explaining Transaction {instance_idx}")
            plt.tight_layout()

            # Summary plot
            plt.subplot(1, 2, 2)
            feature_vals = instance_features[top_indices]
            plt.scatter(
                feature_vals, shap_values[0, top_indices], c=colors, alpha=0.6, s=100
            )
            plt.xlabel("Feature Value")
            plt.ylabel("SHAP Value")
            plt.title("Feature Value vs Impact")
            plt.tight_layout()
            plt.show()

            print(f"\n=== SHAP Feature Explanation for Transaction {instance_idx} ===")
            print(f"Base value (expected output): {base_value:.4f}")
            print(f"\nTop features pushing towards FRAUD (positive SHAP):")
            for i in top_indices[::-1]:
                if shap_values[0, i] > 0:
                    print(
                        f"  {self.feature_names[i]:30s} = {instance_features[i]:8.4f} | SHAP = {shap_values[0, i]:8.4f}"
                    )

            print(f"\nTop features pushing towards NON-FRAUD (negative SHAP):")
            for i in top_indices:
                if shap_values[0, i] < 0:
                    print(
                        f"  {self.feature_names[i]:30s} = {instance_features[i]:8.4f} | SHAP = {shap_values[0, i]:8.4f}"
                    )

        except Exception as e:
            print(f"Error plotting explanation: {e}")


# ============================================================================
# 4. PREPARE TRAINING DATA
# ============================================================================

company_x = hetero_data["company"].x.to(DEVICE)
transaction_x = hetero_data["transaction"].x.to(DEVICE)
edge_company_bip = hetero_data["company", "performs", "transaction"].edge_index.to(
    DEVICE
)

y = transaction_labels.to(DEVICE)
train_idx = train_mask.nonzero(as_tuple=False).squeeze()
test_idx = test_mask.nonzero(as_tuple=False).squeeze()

# Build company -> transaction mapping
companies_index_full = [[] for _ in range(num_companies)]
src_cpu = edge_company_bip[0].detach().cpu().numpy()
dst_cpu = edge_company_bip[1].detach().cpu().numpy()
for s, t in zip(src_cpu, dst_cpu):
    if 0 <= s < num_companies and 0 <= t < num_transactions:
        companies_index_full[int(s)].append(int(t))

companies_index = []
for L in companies_index_full:
    if len(L) > MAX_TRX_PER_COMPANY:
        companies_index.append(L[-MAX_TRX_PER_COMPANY:])
    else:
        companies_index.append(L)

companies_length = torch.tensor([len(c) for c in companies_index])
transactions_index = [src_cpu, dst_cpu]

# Company graph edges
edge_company_h = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

# Transaction graph edges
src_list, dst_list = [], []


def add_group_edges(txn_ids):
    if len(txn_ids) <= 1:
        return
    if MAKE_CLIQUES:
        for a, b in itertools.combinations(txn_ids, 2):
            src_list.extend([a, b])
            dst_list.extend([b, a])
    else:
        for i in range(len(txn_ids) - 1):
            a, b = txn_ids[i], txn_ids[i + 1]
            src_list.extend([a, b])
            dst_list.extend([b, a])


grouped_by_sender = df_transaction.groupby("nameOrig").indices
for _, idxs in grouped_by_sender.items():
    txn_ids = list(idxs)[-LIMIT_PER_ENTITY:]
    add_group_edges(txn_ids)

grouped_by_receiver = df_transaction.groupby("nameDest").indices
for _, idxs in grouped_by_receiver.items():
    txn_ids = list(idxs)[-LIMIT_PER_ENTITY:]
    add_group_edges(txn_ids)

if "step" in df_transaction.columns:
    grouped_by_day = df_transaction.groupby("step").indices
    for _, idxs in grouped_by_day.items():
        txn_ids = list(idxs)[-LIMIT_PER_ENTITY:]
        add_group_edges(txn_ids)

edge_trx_h = (
    torch.tensor([src_list, dst_list], dtype=torch.long, device=DEVICE)
    if len(src_list) > 0
    else torch.empty((2, 0), dtype=torch.long, device=DEVICE)
)
print(f"Transaction-to-transaction edges: {edge_trx_h.shape[1]:,}")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

model = MultiFraudNet(
    company_feat_dim=company_x.size(1),
    transaction_feat_dim=transaction_x.size(1),
    companies_index=companies_index,
    transactions_index=transactions_index,
    hidden_dim=HIDDEN_DIM,
    edge_type_number=1,
).to(DEVICE)

num_classes = 2
counts = torch.bincount(y)
total = counts.sum().float()
class_weights = total / (num_classes * counts.float())

print(f"\n=== CLASS IMBALANCE ANALYSIS ===")
print(f"Original class distribution:")
print(f"  Non-fraud (0): {counts[0].item()} ({counts[0].item()/total*100:.2f}%)")
print(f"  Fraud (1): {counts[1].item()} ({counts[1].item()/total*100:.2f}%)")
print(f"  Imbalance ratio: {counts[0].item()/max(counts[1].item(), 1):.2f}:1")
print(f"  Class weights: {class_weights}")

# Choose loss function: 'weighted_ce', 'focal', or 'combined'
LOSS_TYPE = "focal"  # Options: 'weighted_ce', 'focal', 'combined'

if LOSS_TYPE == "weighted_ce":
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    print(f"\nUsing: Weighted Cross-Entropy Loss")
elif LOSS_TYPE == "focal":
    # Focal loss with alpha balancing
    alpha_focal = class_weights.to(DEVICE)
    criterion = FocalLoss(alpha=alpha_focal, gamma=2.0, reduction="mean")
    print(f"\nUsing: Focal Loss (gamma=2.0)")
elif LOSS_TYPE == "combined":
    # Use both techniques
    criterion = FocalLoss(alpha=class_weights.to(DEVICE), gamma=2.0, reduction="mean")
    print(f"\nUsing: Combined (Focal Loss + Class Weights)")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Log model metadata for explainability report
if explainability_logger:
    model_metadata = {
        "model_type": "MultiFraudNet (Heterogeneous GNN)",
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_type": LOSS_TYPE,
        "device": str(DEVICE),
        "num_companies": num_companies,
        "num_transactions": num_transactions,
        "num_loans": num_loans,
        "max_trx_per_company": MAX_TRX_PER_COMPANY,
        "limit_per_entity": LIMIT_PER_ENTITY,
        "quick_run": QUICK_RUN,
        "quick_run_subsample": QUICK_RUN_SUBSAMPLE,
        "class_weights": {
            "non_fraud": float(class_weights[0]),
            "fraud": float(class_weights[1]),
        },
    }
    explainability_logger.log_model_metadata(model_metadata)

# Optional: Apply SMOTE to training data (uncomment to use)
# transaction_feat_balanced, transaction_labels_balanced = apply_smote_to_transactions(
#     transaction_feat.cpu().numpy(),
#     y.cpu().numpy()
# )
# transaction_x_balanced = transaction_feat_balanced.to(DEVICE)
# y_balanced = transaction_labels_balanced.to(DEVICE)

print(f"\n=== STARTING TRAINING ===")
best_val_loss = float("inf")
patience = 10
patience_counter = 0

# Use AMP scaler when CUDA is available to reduce memory usage
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        out_c, out_t = model(
            company_x, edge_company_h, transaction_x, edge_trx_h, companies_length
        )

        company_labels = torch.zeros(
            len(companies_index), dtype=torch.long, device=DEVICE
        )
        for i, txns in enumerate(companies_index):
            if any(y[t].item() == 1 for t in txns if t < len(y)):
                company_labels[i] = 1

        loss_company = criterion(out_c, company_labels)
        loss_transaction = criterion(out_t[train_idx], y[train_idx])
        loss = 0.5 * (loss_company + loss_transaction)

    # Backprop with AMP if available
    if DEVICE.type == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Free CUDA cache to reduce fragmentation and peak memory
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Evaluate less frequently to save time (every 2 epochs or final epoch)
    if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            out_c_eval, out_t_eval = model(
                company_x, edge_company_h, transaction_x, edge_trx_h, companies_length
            )
            preds = out_t_eval[test_idx].argmax(dim=1)
            acc = (preds == y[test_idx]).float().mean().item()

            # Calculate F1 score (better metric for imbalanced data)
            y_test_np = y[test_idx].cpu().numpy()
            preds_np = preds.cpu().numpy()
            f1 = f1_score(y_test_np, preds_np, zero_division=0)

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f} | F1-Score: {f1:.4f}"
        )

    # Early stopping based on validation loss
    if loss.item() < best_val_loss:
        best_val_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n=== TRAINING COMPLETED ===")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================

model.eval()
with torch.no_grad():
    out_c, out_t = model(
        company_x, edge_company_h, transaction_x, edge_trx_h, companies_length
    )
    preds_logits = out_t[test_idx]
    preds = preds_logits.argmax(dim=1).cpu()
    preds_proba = F.softmax(preds_logits, dim=1)[:, 1].cpu()  # Probability of fraud
    true = y[test_idx].cpu()

print("\n=== EVALUATION RESULTS ===")
print(classification_report(true, preds, digits=4, target_names=["Non-Fraud", "Fraud"]))
print("\nConfusion matrix:")
print(confusion_matrix(true, preds))

# ROC-AUC and PR-AUC scores
try:
    fpr, tpr, thresholds = roc_curve(true, preds_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    precision, recall, pr_thresholds = precision_recall_curve(true, preds_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc:.4f}")

    # Find optimal threshold (F1-based)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = (
        pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
    )
    print(f"\nOptimal threshold (F1-based): {optimal_threshold:.4f}")

    # Predictions with optimal threshold
    preds_optimized = (preds_proba >= optimal_threshold).astype(int)
    f1_optimized = f1_score(true, preds_optimized)
    print(f"F1-Score with optimal threshold: {f1_optimized:.4f}")
    print(f"\nOptimized Classification Report:")
    print(
        classification_report(
            true, preds_optimized, digits=4, target_names=["Non-Fraud", "Fraud"]
        )
    )

except Exception as e:
    print(f"Error computing advanced metrics: {e}")

# ============================================================================
# 7. EXPLAIN FRAUD DETECTION (OPTIONAL)
# ============================================================================

print("\n" + "=" * 80)
print("EXPLAINABILITY PHASE")
print("=" * 80)

# 7A. Graph-Level Explanations (structural importance)
print("\n7A. GRAPH-LEVEL EXPLANATION (Edge/Node Importance)")
print("-" * 80)

fraud_indices = (y == 1).nonzero(as_tuple=True)[0]
if len(fraud_indices) > 0:
    node_to_explain = int(fraud_indices[0].item())
    print(f"Explaining fraudulent transaction node: {node_to_explain}")
    print("This explains WHICH EDGES in the graph influenced the fraud prediction")
    print("(i.e., relational/structural importance)\n")

    try:
        explainer_epochs = 300 if QUICK_RUN else 1200
        explainer_gnn = SimpleGNNExplainer(model, epochs=explainer_epochs, lr=0.01)
        edge_mask = explainer_gnn.explain_node(
            node_to_explain, transaction_x, edge_trx_h
        )
        explainer_gnn.visualize_subgraph(
            node_to_explain, edge_trx_h.cpu(), edge_mask.cpu(), y.cpu(), top_k=50
        )
        print(f"[OK] Graph-level explanation completed for node {node_to_explain}")

        # Log graph explanation
        if explainability_logger:
            edge_index_tuple = (
                edge_trx_h[0].cpu().numpy(),
                edge_trx_h[1].cpu().numpy(),
            )
            explainability_logger.log_graph_explanation(
                node_id=node_to_explain,
                edge_mask=edge_mask,
                edge_index=edge_index_tuple,
                label="fraud",
                top_k=20,
            )
    except Exception as e:
        print(f"✗ Graph-level explanation failed: {e}")
else:
    print("⚠ No fraudulent transactions in test set to explain")

# 7B. Feature-Level Explanations (SHAP)
print("\n7B. FEATURE-LEVEL EXPLANATION (SHAP - Feature Importance)")
print("-" * 80)

if HAS_SHAP and len(fraud_indices) > 0:
    try:
        # Get transaction features and feature names
        transaction_feat_np = transaction_feat.cpu().numpy()
        feature_names = list(df_transaction_encoded.drop(["isFraud"], axis=1).columns)

        print(f"Number of features: {len(feature_names)}")
        print("This explains WHICH FEATURES drove the fraud prediction")
        print("(i.e., feature-level importance)\n")

        # Create SHAP explainer
        shap_explainer = SHAPExplainer(
            model, transaction_feat_np, feature_names, device=DEVICE
        )

        # Explain a few fraudulent transactions
        num_to_explain = min(3, len(fraud_indices))
        for i, fraud_idx in enumerate(fraud_indices[:num_to_explain]):
            fraud_idx_int = int(fraud_idx.item())
            print(f"\n{'='*80}")
            print(
                f"SHAP Explanation {i+1}/{num_to_explain} - Transaction {fraud_idx_int}"
            )
            print(f"{'='*80}")

            shap_values, base_value = shap_explainer.explain_instance(
                fraud_idx_int, num_samples=QUICK_SHAP_SAMPLES
            )

            if shap_values is not None:
                shap_explainer.plot_explanation(
                    fraud_idx_int, shap_values, base_value, top_k=10
                )
                print(f"[OK] SHAP explanation completed")

                # Log feature explanation to logger
                if explainability_logger:
                    feature_values = transaction_feat_np[fraud_idx_int]
                    explainability_logger.log_feature_explanation(
                        transaction_id=fraud_idx_int,
                        shap_values=shap_values[0],
                        feature_names=feature_names,
                        feature_values=feature_values,
                        base_value=base_value,
                        label="fraud",
                        top_k=10,
                    )
            else:
                print(f"✗ SHAP explanation failed for transaction {fraud_idx_int}")

    except Exception as e:
        print(f"✗ SHAP explainability error: {e}")
        print("Tip: Install SHAP with: pip install shap")
else:
    if not HAS_SHAP:
        print("⚠ SHAP not installed")
        print("Install with: pip install shap")
    if len(fraud_indices) == 0:
        print("⚠ No fraudulent transactions available for SHAP explanation")

# Save explainability reports
if explainability_logger:
    print("\n" + "=" * 80)
    print("SAVING EXPLAINABILITY REPORTS")
    print("=" * 80)
    try:
        explainability_logger.save_aggregated_csv()
        explainability_logger.save_summary_report()
        print(f"\n[OK] All explanations saved to: {explainability_logger.exp_dir}")
    except Exception as e:
        print(f"[ERROR] Error saving reports: {e}")

print("\n" + "=" * 80)
print("SUMMARY: Two-Level Explainability")
print("=" * 80)
print(
    """
Your fraud detection model now has TWO complementary explanation levels:

1. GRAPH-LEVEL (Structural):
   - Shows which edges (relationships) in the transaction graph were important
   - Helps understand RELATIONAL patterns (who transacted with whom, in what order)
   - Implemented via: SimpleGNNExplainer (edge masking)

2. FEATURE-LEVEL (Financial):
   - Shows which transaction features (amount, velocity, location, etc.) were important
   - Helps understand FINANCIAL patterns (what characteristics flagged the transaction)
   - Implemented via: SHAP (Shapley values from game theory)

For your thesis:
- Use Graph-Level to explain the GNN architecture's relational learning
- Use Feature-Level to validate that the model respects financial domain knowledge
- Together, they provide full end-to-end interpretability
"""
)

print("\n--- Script Completed ---")
