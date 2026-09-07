# src/data_preprocessing.py
"""
Data preprocessing pipeline for NSL-KDD dataset.
Handles loading, encoding, scaling, and label creation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_numpy, save_model, print_header, print_subheader


# ─── Column Names ───────────────────────────────────────────────
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# ─── Attack Mappings ────────────────────────────────────────────
ATTACK_MAP = {
    # DoS
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos',
    'udpstorm': 'dos', 'processtable': 'dos', 'worm': 'dos',
    'mailbomb': 'dos',
    # Probe
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe',
    'portsweep': 'probe', 'mscan': 'probe', 'saint': 'probe',
    # R2L
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l',
    'phf': 'r2l', 'multihop': 'r2l', 'warezmaster': 'r2l',
    'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l',
    'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
    'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    # U2R
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
    'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r',
    # Normal
    'normal': 'normal'
}

SEVERITY_MAP = {
    'normal': 0.0,
    'probe': 0.25,
    'r2l': 0.50,
    'dos': 0.75,
    'u2r': 1.00
}


def load_nsl_kdd(train_path, test_path):
    """Load raw NSL-KDD data files."""
    print_header("Loading NSL-KDD Dataset")

    train_df = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES, header=None)

    # Drop difficulty_level
    train_df = train_df.drop('difficulty_level', axis=1)
    test_df = test_df.drop('difficulty_level', axis=1)

    print(f"  Training set: {train_df.shape}")
    print(f"  Testing set:  {test_df.shape}")
    print(f"  Features:     {train_df.shape[1] - 1}")
    print(f"  Unique labels: {train_df['label'].nunique()}")

    return train_df, test_df


def create_labels(df):
    """Create binary, multiclass, and severity labels."""
    print_subheader("Creating Labels")

    # Multiclass labels
    df['attack_category'] = df['label'].map(ATTACK_MAP)

    # Handle any unmapped labels
    unmapped = df[df['attack_category'].isna()]['label'].unique()
    if len(unmapped) > 0:
        print(f"  ⚠️  Unmapped labels (treating as r2l): {unmapped}")
        df['attack_category'] = df['attack_category'].fillna('r2l')

    # Binary labels: normal=0, attack=1
    df['binary_label'] = (df['attack_category'] != 'normal').astype(int)

    # Severity scores
    df['severity_score'] = df['attack_category'].map(SEVERITY_MAP)

    print(f"  Binary distribution:")
    print(f"    Normal (0): {(df['binary_label'] == 0).sum()}")
    print(f"    Attack (1): {(df['binary_label'] == 1).sum()}")
    print(f"\n  Multiclass distribution:")
    print(df['attack_category'].value_counts().to_string())

    return df


def encode_and_scale(train_df, test_df, output_dir):
    """One-hot encode categorical features and scale numerical features."""
    print_header("Encoding & Scaling")

    # Separate features and labels
    feature_cols = [c for c in train_df.columns
                    if c not in ['label', 'attack_category',
                                 'binary_label', 'severity_score']]

    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    # Identify numerical columns
    numerical_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_COLS)
        ]
    )

    print("  Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Get feature names after encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS)
    all_feature_names = numerical_cols + list(cat_feature_names)

    print(f"  Features before encoding: {len(feature_cols)}")
    print(f"  Features after encoding:  {X_train_processed.shape[1]}")
    print(f"    Numerical: {len(numerical_cols)}")
    print(f"    One-hot:   {len(cat_feature_names)}")

    # Save preprocessor
    save_model(preprocessor, os.path.join(output_dir, 'encoders',
                                          'preprocessor.pkl'))

    # Save feature names
    np.save(os.path.join(output_dir, 'processed', 'feature_names.npy'),
            all_feature_names)

    return X_train_processed, X_test_processed, all_feature_names


def run_full_pipeline(train_path, test_path, output_dir):
    """Run the complete preprocessing pipeline."""
    print_header("NSL-KDD PREPROCESSING PIPELINE")

    # Step 1: Load
    train_df, test_df = load_nsl_kdd(train_path, test_path)

    # Step 2: Create labels
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)

    # Step 3: Encode and scale
    X_train, X_test, feature_names = encode_and_scale(
        train_df, test_df, output_dir)

    # Step 4: Extract label arrays
    y_train_binary = train_df['binary_label'].values
    y_test_binary = test_df['binary_label'].values

    le_multi = LabelEncoder()
    y_train_multi = le_multi.fit_transform(train_df['attack_category'])
    y_test_multi = le_multi.transform(test_df['attack_category'])

    y_train_severity = train_df['severity_score'].values
    y_test_severity = test_df['severity_score'].values

    # Step 5: Save everything
    print_header("Saving Processed Data")

    processed_dir = os.path.join(output_dir, 'processed')
    encoders_dir = os.path.join(output_dir, 'encoders')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(encoders_dir, exist_ok=True)

    # Save numpy arrays
    save_numpy(X_train, os.path.join(processed_dir, 'X_train.npy'))
    save_numpy(X_test, os.path.join(processed_dir, 'X_test.npy'))
    save_numpy(y_train_binary, os.path.join(processed_dir, 'y_train_binary.npy'))
    save_numpy(y_test_binary, os.path.join(processed_dir, 'y_test_binary.npy'))
    save_numpy(y_train_multi, os.path.join(processed_dir, 'y_train_multi.npy'))
    save_numpy(y_test_multi, os.path.join(processed_dir, 'y_test_multi.npy'))
    save_numpy(y_train_severity, os.path.join(processed_dir, 'y_train_severity.npy'))
    save_numpy(y_test_severity, os.path.join(processed_dir, 'y_test_severity.npy'))

    # Save label encoder
    save_model(le_multi, os.path.join(encoders_dir, 'label_encoder_multi.pkl'))

    # Save processed CSVs (for reference)
    train_df.to_csv(os.path.join(processed_dir, 'train_preprocessed.csv'),
                    index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test_preprocessed.csv'),
                   index=False)

    print_header("PIPELINE COMPLETE ✅")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Binary classes: {np.unique(y_train_binary)}")
    print(f"  Multi classes:  {np.unique(y_train_multi)} → "
          f"{list(le_multi.classes_)}")
    print(f"  Severity range: [{y_train_severity.min()}, "
          f"{y_train_severity.max()}]")

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train_binary': y_train_binary, 'y_test_binary': y_test_binary,
        'y_train_multi': y_train_multi, 'y_test_multi': y_test_multi,
        'y_train_severity': y_train_severity, 'y_test_severity': y_test_severity,
        'feature_names': feature_names,
        'label_encoder_multi': le_multi
    }
