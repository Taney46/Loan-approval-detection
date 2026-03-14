#After cleaning and balancing our data we are going to train 2 supervised models, evaluate them and pick the best one
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,confusion_matrix, classification_report)
from xgboost import XGBClassifier 

SEED = 42
np.random.seed(SEED)

BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})
GREEN = '#1D9E75'; CORAL = '#D85A30'; BLUE = '#378ADD'; PURPLE = '#7F77DD'


#Loading the preprocessed data
def load_preprocessed():
    print("="*60)
    print("SECTION 1: LOADING PREPROCESSED DATA")
    print("="*60)

    X_train = pd.read_csv(os.path.join(OUTPUTS_DIR, 'X_train_preprocessed.csv'))
    y_train = pd.read_csv(os.path.join(OUTPUTS_DIR, 'y_train_preprocessed.csv')).squeeze()
    X_val   = pd.read_csv(os.path.join(OUTPUTS_DIR, 'X_val.csv'))
    y_val   = pd.read_csv(os.path.join(OUTPUTS_DIR, 'y_val.csv')).squeeze()
    X_test  = pd.read_csv(os.path.join(OUTPUTS_DIR, 'X_test_preprocessed.csv'))

    print(f"  X_train : {X_train.shape}  |  y_train: Approved={y_train.sum()}  Rejected={(y_train==0).sum()}")
    print(f"  X_val   : {X_val.shape}    |  y_val  : Approved={y_val.sum()}  Rejected={(y_val==0).sum()}")
    print(f"  X_test  : {X_test.shape}   |  (no labels — competition format)")
    return X_train, y_train, X_val, y_val, X_test


#RANDOM FOREST
def train_random_forest(X_train, y_train):
    print("\n" + "="*60)
    print("SECTION 2: RANDOM FOREST — TRAINING & TUNING")
    print("="*60)

    param_grid = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'max_features'    : ['sqrt', 'log2']}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rf_base = RandomForestClassifier(random_state=SEED, class_weight='balanced')

    print("\n  Running Grid Search ")
    grid_search_rf = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs= -1,
        verbose= 1
    )

    grid_search_rf.fit(X_train, y_train)

    best_rf = grid_search_rf.best_estimator_
    print(f"\n  Best hyperparameters : {grid_search_rf.best_params_}")
    print(f"  Best CV F1-score     : {grid_search_rf.best_score_:.4f}")

    #Cross validation
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='f1')

    print(f"  CV F1 scores (5 folds): {[round(s,4) for s in cv_scores]}")
    print(f"  Mean CV F1: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

    return best_rf


#XGBOOST
def train_xgboost(X_train, y_train):

     print("\n" + "="*60)
     print("SECTION 3: XGBOOST — TRAINING & TUNING")
     print("="*60) 

     neg = (y_train == 0).sum()
     pos = y_train.sum()
     spw = round(neg/pos, 2)

     print(f"  scale_pos_weight = {neg}/{pos} = {spw}")

     param_grid = {
         'n_estimators' : [100, 200, 300],
         'max_depth' : [3, 5, 7],
         'learning_rate' : [0.05, 0.1, 0.2],
         'subsample' : [0.8, 1.0],
         'colsample_bytree' : [0.8, 1.0]
     }

     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

     xgb_base = XGBClassifier(
         random_state = SEED,
         scale_pos_weight = spw,
         use_label_encoder = False,
         eval_metric = 'logloss',
         verbosity = 0
     )
     print("\n  Running Grid Search ")

     grid_search_xgb = GridSearchCV(
         estimator= xgb_base,
         param_grid=param_grid,
         cv=cv,
         scoring='f1',
         n_jobs=-1,
         verbose=1
     )

     grid_search_xgb.fit(X_train, y_train)

     best_xgb = grid_search_xgb.best_estimator_
     print(f"\n  Best hyperparameters : {grid_search_xgb.best_params_}")
     print(f"  Best CV F1-score     : {grid_search_xgb.best_score_:.4f}")

     cv_scores = cross_val_score(best_xgb, X_train, y_train, cv=cv, scoring='f1')

     print(f"  CV F1 scores (5 folds): {[round(s,4) for s in cv_scores]}")
     print(f"  Mean CV F1: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

     return best_xgb


#MODEL EVALUATION

def evaluate_models(rf_model, xgb_model, X_val, y_val):

    print("\n" + "="*60)
    print("SECTION 4: MODEL EVALUATION ON VALIDATION SET")
    print("="*60)

    models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
    results = {}
    colors = {'Random Forest': BLUE, 'XGBoost': PURPLE}

    for name, model in models.items():
        if model is None:
            print(f"\n  {name}: skipped (model not trained)")
            continue


        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]


        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        results[name] = {
            'model'    : model,
            'y_pred'   : y_pred,
            'y_prob'   : y_prob,
            'accuracy' : acc,
            'precision': prec,
            'recall'   : rec,
            'f1'       : f1,
            'auc_roc'  : auc,
            'cm'       : cm
        }


        print(f"\n  ── {name} ──")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}   ← primary metric")
        print(f"  AUC-ROC   : {auc:.4f}   ← secondary metric")
        print(f"\n  Classification Report:\n{classification_report(y_val, y_pred, target_names=['Rejected','Approved'])}")

        #Pick the best model
        best_name = max(results, key=lambda k: (results[k]['f1'], results[k]['auc_roc']))
        print("\n" + "="*60)
        print(f"  BEST MODEL: {best_name}")
        print(f"  F1={results[best_name]['f1']:.4f} | AUC-ROC={results[best_name]['auc_roc']:.4f}")
        print("="*60)
    
        return results, best_name
    


#VISUALISATIONS
def plot_results(results, figures_dir):
    print("\n" + "="*60)
    print("SECTION 5: GENERATING RESULT FIGURES")
    print("="*60)
 
    model_names = list(results.keys())
    colors      = [BLUE, PURPLE]
    metrics     = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']

    #Metrics comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.3
 
    for i, (name, col) in enumerate(zip(model_names, colors)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=col, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
 
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Fig 08 — Model performance comparison on validation set\n'
                 '(F1-score and AUC-ROC are the primary evaluation metrics)',
                 fontweight='bold')
    ax.legend()
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(metrics) - 0.1, 0.81, '0.80 baseline', fontsize=8, color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_08_metrics_comparison.png'))
    plt.close()
    print("  → Saved: fig_08_metrics_comparison.png")

    #Confusion matrices
    fig, axes = plt.subplots(1, len(model_names), figsize=(12, 4))
    if len(model_names) == 1:
        axes = [axes]
    fig.suptitle('Fig 09 — Confusion matrices\n'
                 '(Rows = actual class, Columns = predicted class)',
                 fontweight='bold')
 
    for ax, name, col in zip(axes, model_names, colors):
        cm = results[name]['cm']
        # Normalisation so we show percentages 
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    cmap='Blues' if col == BLUE else 'Purples',
                    linewidths=0.5, cbar=False,
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        # Add percentage annotations below the count
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'({cm_norm[i,j]*100:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
 
        ax.set_title(name)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
 
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_09_confusion_matrices.png'))
    plt.close()
    print("  → Saved: fig_09_confusion_matrices.png")

    #ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
 
    for name, col in zip(model_names, colors):
        fpr, tpr, _ = roc_curve(
            # y_val needs to be re-loaded here because it's passed in from outside
            results[name]['y_prob'] != results[name]['y_prob'],  # placeholder
            results[name]['y_prob']
        )
        # Correct approach: we need y_val accessible — store it in results
        auc = results[name]['auc_roc']
        ax.plot(results[name]['_fpr'], results[name]['_tpr'],
                color=col, linewidth=2,
                label=f'{name} (AUC = {auc:.4f})')
 
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('Fig 10 — ROC curves\n'
                 '(Higher and further left = better discrimination)',
                 fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_10_roc_curves.png'))
    plt.close()
    print("  → Saved: fig_10_roc_curves.png")

    #Feauture Importance - to show which features each model found most useful for making predictions
    try:
        feature_names = results[model_names[0]]['model'].feature_names_in_
    except AttributeError:
        feature_names = [f'Feature {i}' for i in range(
            len(results[model_names[0]]['model'].feature_importances_))]
 
    fig, axes = plt.subplots(1, len(model_names), figsize=(14, 7))
    if len(model_names) == 1:
        axes = [axes]
    fig.suptitle('Fig 11 — Feature importance\n'
                 '(Which applicant attributes most strongly influence the prediction?)',
                 fontweight='bold')
 
    for ax, name, col in zip(axes, model_names, colors):
        importances = results[name]['model'].feature_importances_
        indices     = np.argsort(importances)  # sort ascending for horizontal bar
        top_n       = 12                        # show top 12 features
 
        top_idx   = indices[-top_n:]
        top_imp   = importances[top_idx]
        top_names = [feature_names[i] for i in top_idx]
 
        ax.barh(range(top_n), top_imp, color=col,
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('Importance score')
        ax.set_title(name)
 
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_11_feature_importance.png'))
    plt.close()
    print("  → Saved: fig_11_feature_importance.png")

def evaluate_and_plot(rf_model, xgb_model, X_val, y_val, figures_dir):
    from sklearn.metrics import roc_curve

    models  = {'Random Forest': rf_model, 'XGBoost': xgb_model}
    results = {}
 
    for name, model in models.items():
        if model is None:
            continue
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
    
        results[name] = {
            'model'    : model,
            'y_pred'   : y_pred,
            'y_prob'   : y_prob,
            '_fpr'     : fpr,
            '_tpr'     : tpr,
            'accuracy' : accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall'   : recall_score(y_val, y_pred),
            'f1'       : f1_score(y_val, y_pred),
            'auc_roc'  : roc_auc_score(y_val, y_prob),
            'cm'       : confusion_matrix(y_val, y_pred)
            }
    
        print(f"\n  ── {name} ──")
        print(f"  Accuracy  : {results[name]['accuracy']:.4f}")
        print(f"  Precision : {results[name]['precision']:.4f}")
        print(f"  Recall    : {results[name]['recall']:.4f}")
        print(f"  F1-score  : {results[name]['f1']:.4f}   ← primary metric")
        print(f"  AUC-ROC   : {results[name]['auc_roc']:.4f}   ← secondary metric")
        print(f"\n  Classification Report:")
        print(classification_report(y_val, y_pred,
                                    target_names=['Rejected', 'Approved']))
    
    best_name = max(results, key=lambda k: (results[k]['f1'], results[k]['auc_roc']))
    print("\n" + "="*60)
    print(f"  BEST MODEL: {best_name}")
    print(f"  F1={results[best_name]['f1']:.4f} | AUC-ROC={results[best_name]['auc_roc']:.4f}")
    print("="*60)


    #Metrics bar chart
    model_names   = list(results.keys())
    colors_list   = [BLUE, PURPLE]
    metrics       = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.3
    for i, (name, col) in enumerate(zip(model_names, colors_list)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                    color=col, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Fig 08 — Model performance comparison on validation set',
                fontweight='bold')
    ax.legend()
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(metrics) - 0.35, 0.81, '0.80 baseline', fontsize=8, color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_08_metrics_comparison.png'))
    plt.close()
    print("\n  → Saved: fig_08_metrics_comparison.png")


    #Confusion Matrix
    fig, axes = plt.subplots(1, len(model_names), figsize=(12, 4))
    if len(model_names) == 1: axes = [axes]
    fig.suptitle('Fig 09 — Confusion matrices', fontweight='bold')
    for ax, name, col in zip(axes, model_names, colors_list):
        cm = results[name]['cm']
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmap = 'Blues' if col == BLUE else 'Purples'
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap,
                    linewidths=0.5, cbar=False,
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'({cm_norm[i,j]*100:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
        ax.set_title(name); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_09_confusion_matrices.png'))
    plt.close()
    print("  → Saved: fig_09_confusion_matrices.png")

    #ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, col in zip(model_names, colors_list):
        ax.plot(results[name]['_fpr'], results[name]['_tpr'],
                color=col, linewidth=2,
                label=f'{name} (AUC = {results[name]["auc_roc"]:.4f})')
    ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Fig 10 — ROC curves', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_10_roc_curves.png'))
    plt.close()
    print("  → Saved: fig_10_roc_curves.png")

    #Feature Importance
    try:
        feature_names = list(results[model_names[0]]['model'].feature_names_in_)
    except AttributeError:
        feature_names = [f'Feature {i}' for i in
                        range(len(results[model_names[0]]['model'].feature_importances_))]
    
    fig, axes = plt.subplots(1, len(model_names), figsize=(14, 7))
    if len(model_names) == 1: axes = [axes]
    fig.suptitle('Fig 11 — Feature importance (top 12)', fontweight='bold')
    for ax, name, col in zip(axes, model_names, colors_list):
        imps    = results[name]['model'].feature_importances_
        indices = np.argsort(imps)
        top_n   = min(12, len(indices))
        top_idx = indices[-top_n:]
        ax.barh(range(top_n), imps[top_idx], color=col,
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
        ax.set_xlabel('Importance score'); ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_11_feature_importance.png'))
    plt.close()
    print("  → Saved: fig_11_feature_importance.png")
    
    
    return results, best_name
    


if __name__ == '__main__':
    print("="*60)
    print("LOAN APPROVAL PREDICTION - MODEL TRAINING & EVALUATION")
    print("Group 4 | COSC440 Artificial Intelligence")
    print("="*60)
 
    X_train, y_train, X_val, y_val, X_test = load_preprocessed()
 
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
 
    print("\n" + "="*60)
    print("SECTION 4 & 5: EVALUATION + FIGURES")
    print("="*60)
    results, best_name = evaluate_and_plot(rf_model, xgb_model, X_val, y_val, FIGURES_DIR)
 
    best_model = results[best_name]['model']
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]
 
    # Reload raw test file to recover Loan_ID (was dropped during preprocessing)
    # The competition submission requires Loan_ID and Loan_Status columns only
    test_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test.csv'))
    loan_ids = test_raw['Loan_ID'].values
 
    # Competition submission file (Loan_ID + Loan_Status only)
    submission = pd.DataFrame({
        'Loan_ID'     : loan_ids,
        'Loan_Status' : ['Y' if p == 1 else 'N' for p in test_preds]
    })
    submission.to_csv(os.path.join(OUTPUTS_DIR, 'submission.csv'), index=False)
 
    # Detailed predictions file (for our own analysis)
    detailed = pd.DataFrame({
        'Loan_ID'              : loan_ids,
        'Predicted_Loan_Status': ['Y' if p == 1 else 'N' for p in test_preds],
        'Approval_Probability' : test_probs.round(4)
    })
    detailed.to_csv(os.path.join(OUTPUTS_DIR, 'test_predictions.csv'), index=False)
 
    print(f"\n  Submission file saved -> outputs/submission.csv")
    print(f"  Detailed predictions  -> outputs/test_predictions.csv")
    print(f"  Predicted Approved: {(test_preds==1).sum()} | Rejected: {(test_preds==0).sum()}")
 
    print("\n" + "="*60)
    print("  ALL DONE.")
    print(f"  Best model : {best_name}")
    print(f"  F1-score   : {results[best_name]['f1']:.4f}")
    print(f"  AUC-ROC    : {results[best_name]['auc_roc']:.4f}")
    print("  Figures saved to outputs/figures/")
    print("="*60)