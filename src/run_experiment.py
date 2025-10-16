import pandas as pd
from pathlib import Path
import warnings
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


BASE_PATH = Path(__file__).parent.parent
PROCESSED_PATH = BASE_PATH / "data" / "processed"
SYNTHETIC_PATH = BASE_PATH / "data" / "synthetic"
RESULTS_PATH = BASE_PATH / "results"
TABLES_PATH = RESULTS_PATH / "tables"
FIGURES_PATH = RESULTS_PATH / "figures"
MODELS_PATH = BASE_PATH / "models"

TARGET_FEATURE = "Severity"
RANDOM_STATE = 42
N_REPEATS = 3  # Number of times to repeat each experiment

GENERATORS_TO_TEST = [
    "ddpm",
    "nflow",
    "ctgan",
    "tvae",
]

SYNTHETIC_PATH.mkdir(exist_ok=True)
TABLES_PATH.mkdir(exist_ok=True)
FIGURES_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

def load_data(train_path: Path, test_path: Path):
    print(f"Loading training data from: {train_path.name}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_and_evaluate_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns=[TARGET_FEATURE])
    y_train = train_df[TARGET_FEATURE]
    X_test = test_df.drop(columns=[TARGET_FEATURE])
    y_test = test_df[TARGET_FEATURE]

    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # For binary classification, the minority class (malignant) is labeled as 1
    pos_label_idx = 1 if 1 in model.classes_ else 0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # In mammographic mass, 1 = Malignant (minority), 0 = Benign (majority)
    minority_label_str = "1"

    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_minority": report[minority_label_str]["f1-score"],
        "precision_minority": report[minority_label_str]["precision"],
        "recall_minority": report[minority_label_str]["recall"],
        "roc_auc": roc_auc_score(y_test, y_prob[:, pos_label_idx])
    }
    return metrics


def run_single_experiment(train_path: Path, test_path: Path, generator_name: str, strategy: str, run_id: int):
    train_df, test_df = load_data(train_path, test_path)
    
    if generator_name.lower() == 'baseline':
        print("Running BASELINE experiment...")
        results = train_and_evaluate_classifier(train_df, test_df)
    
    else:
        print(f"Running SYNTHETIC experiment with generator: {generator_name}")

        loader = GenericDataLoader(train_df, target_column=TARGET_FEATURE)
        generator = Plugins().get(generator_name)
        generator.fit(loader, cond=train_df[TARGET_FEATURE])

        value_counts = train_df[TARGET_FEATURE].value_counts()
        # 0 = Benign (majority), 1 = Malignant (minority)
        n_benign = value_counts.get(0, 0)
        n_malignant = value_counts.get(1, 0)
        
        if n_benign > n_malignant:
            n_to_generate = n_benign - n_malignant
            minority_class_label = 1
            print(f"Class imbalance detected. Generating {n_to_generate} minority (Malignant) samples...")
            
            conditions = [minority_class_label] * n_to_generate
            
            synth_data = generator.generate(count=n_to_generate, cond=conditions).dataframe()
            
            balanced_train_df = pd.concat([train_df, synth_data], ignore_index=True)
            
            synth_filename = f"{train_path.stem}_balanced_by_{generator_name}_run{run_id}.csv"
            balanced_train_df.to_csv(SYNTHETIC_PATH / synth_filename, index=False)
        else:
            print("Data is already balanced or has minority as majority. Using as is.")
            balanced_train_df = train_df

        results = train_and_evaluate_classifier(balanced_train_df, test_df)

    # Extract dataset information from filename
    imbalance_info = train_path.stem.split('_')
    dataset_type = imbalance_info[1]  # 'imbalanced' or 'control'
    imbalance_ratio = imbalance_info[3] if len(imbalance_info) > 3 else "original"

    result_log = {
        "run_id": run_id,
        "dataset_type": dataset_type,
        "imbalance_ratio": f"{imbalance_ratio}:1",
        "model": generator_name,
        "strategy": strategy,
        "train_set_size": len(train_df),
        **results 
    }
    return result_log


def main():
    print("\n")
    print(" Starting Mammographic Mass Experimental Pipeline ")
    print("\n")
    
    all_results = []
    test_path = PROCESSED_PATH / "test.csv"
    
    train_paths = sorted(glob.glob(str(PROCESSED_PATH / "train_*.csv")))
    
    if not train_paths:
        print("ERROR: No training datasets found in", PROCESSED_PATH)
        return
    
    print(f"\nFound {len(train_paths)} training datasets to process.")
    print(f"Each will be run {N_REPEATS} times with {len(GENERATORS_TO_TEST) + 1} methods (baseline + {len(GENERATORS_TO_TEST)} generators).")
    print(f"Total experiments: {len(train_paths) * N_REPEATS * (len(GENERATORS_TO_TEST) + 1)}\n")
    
    for train_path in train_paths:
        train_path = Path(train_path)
        
        for i in range(1, N_REPEATS + 1):
            print("\n")
            print(f">>> Dataset: {train_path.name} | Run: {i}/{N_REPEATS} <<<")
            print("\n")

            baseline_result = run_single_experiment(train_path, test_path, "baseline", "N/A", i)
            all_results.append(baseline_result)
            print(f"✓ Baseline - F1-Minority: {baseline_result['f1_minority']:.4f}, ROC-AUC: {baseline_result['roc_auc']:.4f}")

            for generator in GENERATORS_TO_TEST:
                synthetic_result = run_single_experiment(train_path, test_path, generator, "Naive Oversampling", i)
                all_results.append(synthetic_result)
                print(f"✓ {generator.upper()} - F1-Minority: {synthetic_result['f1_minority']:.4f}, ROC-AUC: {synthetic_result['roc_auc']:.4f}")
    

    print(" All experiments complete. Saving results... ")
    print("\n")
    
    results_df = pd.DataFrame(all_results)

    summary_df = results_df.groupby(['dataset_type', 'imbalance_ratio', 'model', 'strategy']).agg(
        {
            'f1_minority': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'recall_minority': ['mean', 'std'],
            'precision_minority': ['mean', 'std']
        }
    ).reset_index()

    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.rename(columns={
        'dataset_type_': 'dataset_type',
        'imbalance_ratio_': 'imbalance_ratio',
        'model_': 'model',
        'strategy_': 'strategy'
    })

    results_df.to_csv(TABLES_PATH / "detailed_experiment_results.csv", index=False)
    summary_df.to_csv(TABLES_PATH / "summary_experiment_results.csv", index=False)
    
    print(f"\n✓ Detailed results saved to: {TABLES_PATH / 'detailed_experiment_results.csv'}")
    print(f"✓ Summary results saved to: {TABLES_PATH / 'summary_experiment_results.csv'}")
    

    print(" FINAL SUMMARY REPORT ")
    print("\n")
    print(summary_df.to_string(index=False))
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()