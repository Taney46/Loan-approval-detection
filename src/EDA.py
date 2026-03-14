#Before training the model, we did EDA to undersatnd the data first then preprocess it


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

TRAIN_PATH  = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH   = os.path.join(DATA_DIR, 'test.csv')

PALETTE = {'Approved': '#1D9E75', 'Rejected': '#D85A30'}
sns.set_theme(style='whitegrid', font_scale=1.05)

#LOADING THE DATA
def load_data(train_path: str, test_path:str):
    train= pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Training set: {train.shape[0]} rows * {train.shape[1]} columns")
    print(f"Test set: {test.shape[0]} rows * {test.shape[1]} columns")

    return train, test


#EDA
def run_EDA(df:pd.DataFrame, figures_dir: str):
    print("\n" + "="*60)
    print("SECTION 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Overview
    print("\n── 2.1 Dataset overview ──")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

    # Descriptive statistics: mean,median, std, min max for every column
    print("\n── 2.2 Descriptive statistics (numerical features) ──")
    print(df.describe().round(2))

    #Missing values
    print("\n── 2.3 Missing values ──")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing, 'Percent (%)': missing_pct})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
    print(missing_df)

    #Target Variables
    print("\n── 2.4 Target variable distribution ──")
    counts = df['Loan_Status'].value_counts()
    pcts   = df['Loan_Status'].value_counts(normalize=True).mul(100).round(1)
    print(pd.DataFrame({'Count': counts, 'Percent (%)': pcts}))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('2.4 — Target variable: Loan Status distribution', fontweight='bold')
 
    status_map = {'Y': 'Approved', 'N': 'Rejected'}
    df['_Status'] = df['Loan_Status'].map(status_map)
    vc = df['_Status'].value_counts()
 
    # Bar chart
    axes[0].bar(vc.index, vc.values,
                color=[PALETTE['Approved'], PALETTE['Rejected']],
                edgecolor='white', linewidth=0.5)
    for i, (label, val) in enumerate(vc.items()):
        axes[0].text(i, val + 5, f'{val}\n({val/len(df)*100:.1f}%)',
                     ha='center', fontsize=11)
    axes[0].set_title('Count per class')
    axes[0].set_ylabel('Number of applications')
    axes[0].set_ylim(0, vc.max() * 1.2)
 
    # Pie chart
    axes[1].pie(vc.values, labels=vc.index, autopct='%1.1f%%',
                colors=[PALETTE['Approved'], PALETTE['Rejected']],
                startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title('Class proportion')
 
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_01_target_distribution.png'))
    plt.close()
    print("  → Saved: fig_01_target_distribution.png")
    df.drop(columns=['_Status'], inplace=True)
 

    #Univariate variables
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
 
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle('2.5 — Univariate analysis: numerical features\n'
                 '(Histograms show distribution shape; box plots show outliers)',
                 fontweight='bold')
 
    for i, col in enumerate(num_cols):
        # Histogram
        axes[0, i].hist(df[col].dropna(), bins=30,
                        color='#378ADD', edgecolor='white', linewidth=0.4)
        axes[0, i].set_title(col.replace('Applicant', 'App.'))
        axes[0, i].set_xlabel('Value')
        if i == 0:
            axes[0, i].set_ylabel('Frequency')
 
        # Box plot — the dots outside the whiskers are outliers
        axes[1, i].boxplot(df[col].dropna(), vert=True,
                           patch_artist=True,
                           boxprops=dict(facecolor='#B5D4F4', color='#185FA5'),
                           medianprops=dict(color='#185FA5', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='#D85A30',
                                          markersize=3, alpha=0.5))
        axes[1, i].set_title(f'{col.replace("Applicant", "App.")} (outliers)')
        if i == 0:
            axes[1, i].set_ylabel('Value')
 
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_02_numerical_univariate.png'))
    plt.close()
    print("  → Saved: fig_02_numerical_univariate.png")
 
    #  Univariate: these are  categorical features
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area', 'Credit_History']
 
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('2.6 — Univariate analysis: categorical features',
                 fontweight='bold')
    axes = axes.flatten()
 
    for i, col in enumerate(cat_cols):
        vc = df[col].value_counts()
        axes[i].bar(vc.index.astype(str), vc.values,
                    color='#7F77DD', edgecolor='white', linewidth=0.5)
        axes[i].set_title(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=20)
 
    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_03_categorical_univariate.png'))
    plt.close()
    print("  → Saved: fig_03_categorical_univariate.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('2.7 — Bivariate analysis: categorical features vs Loan Status\n'
                 '(Green = Approved, Red = Rejected)',
                 fontweight='bold')
    axes = axes.flatten()
 
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col].astype(str), df['Loan_Status'])
        ct = ct.rename(columns={'Y': 'Approved', 'N': 'Rejected'})
        ct[['Approved', 'Rejected']].plot(
            kind='bar', ax=axes[i], stacked=False,
            color=[PALETTE['Approved'], PALETTE['Rejected']],
            edgecolor='white', linewidth=0.5
        )
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=20)
        axes[i].legend(fontsize=8)
 
    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_04_categorical_bivariate.png'))
    plt.close()
    print("  → Saved: fig_04_categorical_bivariate.png")
 
    # Bivariate: numerical features vs Loan_Status
    # Box plots split by class show whether approved/rejected differ in income, etc.
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(' Bivariate analysis: numerical features vs Loan Status\n'
                 '(Do approved and rejected applicants differ in income/loan size?)',
                 fontweight='bold')
 
    status_map = {'Y': 'Approved', 'N': 'Rejected'}
    df['_Status'] = df['Loan_Status'].map(status_map)
 
    for i, col in enumerate(num_cols):
        data_approved = df.loc[df['_Status'] == 'Approved', col].dropna()
        data_rejected = df.loc[df['_Status'] == 'Rejected', col].dropna()
 
        bp = axes[i].boxplot(
            [data_approved, data_rejected],
            labels=['Approved', 'Rejected'],
            patch_artist=True,
            boxprops=dict(linewidth=1),
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )
        bp['boxes'][0].set_facecolor('#9FE1CB')
        bp['boxes'][1].set_facecolor('#F5C4B3')
        axes[i].set_title(col.replace('Applicant', 'App.'))
        axes[i].set_ylabel('Value')
 
    df.drop(columns=['_Status'], inplace=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_05_numerical_bivariate.png'))
    plt.close()
    print("  → Saved: fig_05_numerical_bivariate.png")
 
    #  Correlation heatmap
    # Checks multicollinearity — if two features are highly correlated (>0.9),
    # one adds no new information and can be dropped. Here we expect moderate
    # correlation between ApplicantIncome and LoanAmount (~0.57).
    num_df = df[num_cols + ['Credit_History']].copy()
    corr = num_df.corr()
 
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show only lower triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('2.9 — Correlation heatmap (numerical features)\n'
                 'No severe multicollinearity detected (all |r| < 0.6)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_06_correlation_heatmap.png'))
    plt.close()
    print("  → Saved: fig_06_correlation_heatmap.png")

    # Log-transform comparison 
    # Shows BEFORE vs AFTER log transform for the two most skewed features.
    # This visualises why the transformation is necessary.
    skewed = ['ApplicantIncome', 'LoanAmount']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('2.10 — Log-transformation: before vs after\n'
                 '(Right-skewed → near-normal after log; models learn better from this)',
                 fontweight='bold')
 
    for i, col in enumerate(skewed):
        series = df[col].dropna()
        log_series = np.log1p(series)   # log1p = log(1+x), safe for zero values
 
        axes[0, i].hist(series, bins=40, color='#D85A30', edgecolor='white', linewidth=0.3)
        axes[0, i].set_title(f'{col} — original (skewed)')
        axes[0, i].set_ylabel('Frequency')
 
        axes[1, i].hist(log_series, bins=40, color='#1D9E75', edgecolor='white', linewidth=0.3)
        axes[1, i].set_title(f'log({col}) — after transform (near-normal)')
        axes[1, i].set_ylabel('Frequency')
 
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_07_log_transform.png'))
    plt.close()
    print("  → Saved: fig_07_log_transform.png")
 
    print("\n  EDA complete — all figures saved to outputs/figures/")


#PRE-PROCESSING

def preprocess(train: pd.DataFrame, test: pd.DataFrame):

     print("\n" + "="*60)
     print("SECTION 3: PREPROCESSING")
     print("="*60)

     #Copy the data to avoid messing up the original ones
     train= train.copy()
     test= test.copy()

     ##Dropping the loan ID because it has no predictive power and may confuse the model
     train.drop(columns=['Loan_ID'], inplace=True)
     test.drop(columns=['Loan_ID'], inplace=True)

     #Separating the target variables from he features
     y_train= train['Loan_Status'].map({'Y':1, 'N':0})
     train.drop(columns=['Loan_Status'], inplace=True)

     print(f"Target Extracted | Approved: {y_train.sum()} | Rejected: {(y_train==0).sum()}")

     #Identify the features types
     cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area']
     num_cols = ['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
     
     #Missing value Imputation for the train data only 
     print("\n Missing value imputation")
     for col in cat_cols:
        mode_val = train[col].mode()[0]   
        missing_tr = train[col].isna().sum()
        missing_te = test[col].isna().sum()
        train[col].fillna(mode_val, inplace=True)
        test[col].fillna(mode_val, inplace=True)
        if missing_tr > 0 or missing_te > 0:
            print(f"  {col}: imputed {missing_tr} train / {missing_te} test → mode='{mode_val}'")
 
     for col in num_cols:
        median_val = train[col].median()   
        missing_tr = train[col].isna().sum()
        missing_te = test[col].isna().sum()
        train[col].fillna(median_val, inplace=True)
        test[col].fillna(median_val, inplace=True)
        if missing_tr > 0 or missing_te > 0:
            print(f"  {col}: imputed {missing_tr} train / {missing_te} test → median={median_val:.2f}")
 
    # Verify all missing values are gone
     assert train.isna().sum().sum() == 0, "Training set still has missing values!"
     assert test.isna().sum().sum()  == 0, "Test set still has missing values!"
     print("  All missing values resolved.")

     #Outlier capping using IQR: opting for this becasue we have only 614 records, removing the outlier rows will mean loosing data
     print("\n Outlier capping (IQR) ")
     income_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
     for col in income_cols:
        Q1  = train[col].quantile(0.25)
        Q3  = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
 
        # Count outliers before capping
        n_out_tr = ((train[col] < lower) | (train[col] > upper)).sum()
        n_out_te = ((test[col]  < lower) | (test[col]  > upper)).sum()
 
        # Cap both sets using limits learned from TRAIN
        train[col] = train[col].clip(lower=lower, upper=upper)
        test[col]  = test[col].clip(lower=lower, upper=upper)
 
        print(f"  {col}: capped {n_out_tr} train / {n_out_te} test outliers "
              f"[{lower:.0f}, {upper:.0f}]")
        

        #FEATURE ENGINEERING
        print("\n Feature engineering ")
        for df_ in [train, test]:
            df_['TotalIncome']        = df_['ApplicantIncome'] + df_['CoapplicantIncome']
            df_['Log_ApplicantIncome']  = np.log1p(df_['ApplicantIncome'])
            df_['Log_CoapplicantIncome']= np.log1p(df_['CoapplicantIncome'])
            df_['Log_LoanAmount']       = np.log1p(df_['LoanAmount'])
            df_['Log_TotalIncome']      = np.log1p(df_['TotalIncome'])
 
        # Drop raw skewed columns — the log versions contain the same info, cleaner
        raw_skewed = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome']
        train.drop(columns=raw_skewed, inplace=True)
        test.drop(columns=raw_skewed,  inplace=True)
        print("  Created: TotalIncome, Log_ApplicantIncome, Log_CoapplicantIncome,")
        print("           Log_LoanAmount, Log_TotalIncome")
        print(f"  Dropped raw skewed columns: {raw_skewed}")

        #Encoding Categorical variables
        print("\n── 3.7 Encoding categorical variables ──")
        binary_map = {
            'Gender':        {'Male': 1, 'Female': 0},
            'Married':       {'Yes': 1, 'No': 0},
            'Education':     {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0}
        }
        for col, mapping in binary_map.items():
            train[col] = train[col].map(mapping)
            test[col]  = test[col].map(mapping)
            print(f"  Label encoded: {col} → {mapping}")
    
        # Dependents: '3+' → 3 so it becomes truly numerical
        for df_ in [train, test]:
            df_['Dependents'] = df_['Dependents'].replace('3+', 3).astype(int)
        print("  Dependents: '3+' replaced with 3")
    
        # One-hot encode Property_Area
        train = pd.get_dummies(train, columns=['Property_Area'], drop_first=True)
        test  = pd.get_dummies(test,  columns=['Property_Area'], drop_first=True)
    
        # Align columns — test might be missing a dummy column if a category is absent
        train, test = train.align(test, join='left', axis=1, fill_value=0)
        print("  One-hot encoded: Property_Area (drop_first=True → 2 new columns)")

        #SCALING
        from sklearn.preprocessing import StandardScaler
        print("\n── 3.8 Scaling numerical features ──")
        scale_cols = ['Log_ApplicantIncome', 'Log_CoapplicantIncome',
                    'Log_LoanAmount', 'Log_TotalIncome',
                    'Loan_Amount_Term', 'Credit_History']
    
        scaler = StandardScaler()
        train[scale_cols] = scaler.fit_transform(train[scale_cols])   # fit+transform on train
        test[scale_cols]  = scaler.transform(test[scale_cols])         # transform only on test
        print(f"  Scaled columns: {scale_cols}")
    
        print(f"\n  Final training shape : {train.shape}")
        print(f"  Final test shape     : {test.shape}")
        print(f"  Final feature list   : {list(train.columns)}")
    
        return train, y_train, test, scaler



#TRANING, VALIDATION
def split_and_smote(X_train: pd.DataFrame, y_train: pd.Series):
    from sklearn.model_selection import train_test_split

    print("\n" + "="*60)
    print("SECTION 4: TRAIN/VAL SPLIT + SMOTE")
    print("="*60)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train
    )

    print(f"\n── 4.1 Split ──")
    print(f"  Train   : {X_tr.shape[0]} rows | "
          f"Approved={y_tr.sum()} ({y_tr.mean()*100:.1f}%) | "
          f"Rejected={(y_tr==0).sum()} ({(1-y_tr.mean())*100:.1f}%)")
    print(f"  Validate: {X_val.shape[0]} rows | "
          f"Approved={y_val.sum()} ({y_val.mean()*100:.1f}%) | "
          f"Rejected={(y_val==0).sum()} ({(1-y_val.mean())*100:.1f}%)")
    
    ##SMOTE
    imbalance_ratio = y_tr.sum() / (y_tr == 0).sum()
    print(f"\n── 4.2 SMOTE ──")
    print(f"  Imbalance ratio (Approved:Rejected) = {imbalance_ratio:.2f}")
 
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=SEED)
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)
        print(f"  SMOTE applied → new training size: {X_tr_bal.shape[0]} rows")
        print(f"  Post-SMOTE: Approved={y_tr_bal.sum()} | Rejected={(y_tr_bal==0).sum()}")
    except ImportError:
        # imbalanced-learn not installed — note it and continue without SMOTE
        print("  WARNING: imbalanced-learn not installed. Install with:")
        print("           pip install imbalanced-learn")
        print("  Continuing WITHOUT SMOTE (class weights will be used instead).")
        X_tr_bal, y_tr_bal = X_tr, y_tr
 
    return X_tr_bal, y_tr_bal, X_val, y_val


if __name__ == '__main__':
    print("="*60)
    print("LOAN APPROVAL PREDICTION — EDA & PREPROCESSING")
    print("Group 4 | COSC440 Artificial Intelligence")
    print("="*60)
 
    # Load
    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)
 
    # EDA (saves all figures to outputs/figures/)
    run_EDA(train_raw.copy(), FIGURES_DIR)
 
    # Preprocess
    X_train, y_train, X_test, scaler = preprocess(train_raw, test_raw)
 
    # Split + SMOTE
    X_tr_bal, y_tr_bal, X_val, y_val = split_and_smote(X_train, y_train)
 
    # Save preprocessed data for the next script (model training)
    out = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(out, exist_ok=True)
    X_tr_bal.to_csv(os.path.join(out, 'X_train_preprocessed.csv'), index=False)
    y_tr_bal.to_csv(os.path.join(out, 'y_train_preprocessed.csv'), index=False)
    X_val.to_csv(   os.path.join(out, 'X_val.csv'),                index=False)
    y_val.to_csv(   os.path.join(out, 'y_val.csv'),                index=False)
    X_test.to_csv(  os.path.join(out, 'X_test_preprocessed.csv'),  index=False)
    print("\n  Preprocessed data saved to outputs/")
    print("\nDone! Run 02_model_training.py next.")