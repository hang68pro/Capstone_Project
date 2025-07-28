# LSC568 CAPSTONE PROJECT
# GROUP 3
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA

# Define the file path
file_path = '/Users/hangtran/Desktop/Capstone_Project/Datasets/'
# Load cell lines data
cell_lines = pd.read_excel(file_path + 'Cell_Lines_Details.xlsx')
# Load gene expression data
gene_expression = pd.read_csv(file_path + 'Cell_line_RMA_proc_basalExp.txt', delimiter='\t')
# Load dose response data
gdsc2 = pd.read_excel(file_path + 'GDSC2_fitted_dose_response_27Oct23.xlsx')
# Load compound data
compounds_df = pd.read_excel(file_path + 'screened_compounds_rel_8.5.xlsx')
# Load mutation data
mutations = pd.read_csv(file_path + 'mutations_all_20230202.csv')
# See the first few row of cell lines data
cell_lines.head(2)
# Gene expression shape
print(f'Gene expression shape: {cell_lines.shape}')
# See the first few rows of gene expression data
gene_expression.head(3)
# See the first few lines of dose response
gdsc2.head(3)
# Shape of dose response
print(f'Dose response (GDSC2) rows and columns: {gdsc2.shape}')

#FILTER, CLEAN AND MERGE DATASETS
# Rename COSMIC Identifier to COSMIC_ID
cell_lines = cell_lines.rename(columns={'COSMIC identifier':'COSMIC_ID'})
# Filter LUAD cancer type
luad_cell_lines = cell_lines[cell_lines.iloc[:,9] == 'LUAD']
# Number of LUAD cell lines
print('Number of LUAD cell lines:', len(luad_cell_lines['COSMIC_ID']))
# Filter LUAD cell lines in dose response uisng COSMIC_ID
gdsc2_luad = gdsc2[gdsc2['COSMIC_ID'].isin(luad_cell_lines['COSMIC_ID'])]
# Shape of dose response dataset after filtering for LUAD
print(f'Dose response (GDSC2) shape after filtered for LUAD: {gdsc2_luad.shape}')
# Gene experssion first column is gene symbol and the rest are expression value in format "DATA.<COSMIC_ID>"
# Filter for columns corresponding to luad_cell_lines using COSMIC_ID
# Generate LUAD cell lines into a list
# Ensure luad_cosmicID is a set of strings
luad_cosmicID_set = set(map(str, luad_cell_lines['COSMIC_ID'].astype(int).tolist()))
# Find matching columns
luad_col = ["GENE_SYMBOLS"] + [col for col in gene_expression.columns if col.startswith('DATA.') and col.split(".")[1] in luad_cosmicID_set]
# Filter the gene expression dataframe for LUAD cell lines
cell_line_exp = gene_expression[luad_col]
# Shape of gene expression after filter for LUAD
print("Gene expression shape after filtered for LUAD:", cell_line_exp.shape)
# Drop missing values and duplicated values in Filtered Gene Expression
cell_line_exp = cell_line_exp.drop_duplicates(subset=['GENE_SYMBOLS'])
# Check the shape of gene expression after dropping missing and duplicated values
print('Gene expression shape after dropping missing and duplicated values', cell_line_exp.shape)
# Clean columns name in gene expression so sample columns only contain COSMIC_ID
new_cols = {cell_line_exp.columns[0]: 'GENE_SYMBOLS'}
for col in cell_line_exp.columns[1:]:
    cosmic_str = col.replace('DATA.', '')
    if '.' in cosmic_str:
        cosmic_str = cosmic_str.split('.')[0]
    try:
        cid = int(cosmic_str)
        new_cols[col] = str(cid)
    except ValueError:
        new_cols[col] = col
cell_line_exp.rename(columns=new_cols, inplace=True)
# Tranpose the columns and row in gene expression data
gene_exp_transposed = cell_line_exp.set_index('GENE_SYMBOLS').T
# Convert the index name to COSMIC_ID
gene_exp_transposed.index.name = 'COSMIC_ID'
# Convert index to integer type
gene_exp_transposed.index = gene_exp_transposed.index.astype(int)
gene_exp_transposed.reset_index(inplace=True)
# Check the transposed gene expression shape
print('Transposed gene expression shape', gene_exp_transposed.shape)
# Merge dose response with cell lines using COSMIC_ID
merged_df = pd.merge(gdsc2_luad, luad_cell_lines, on='COSMIC_ID', suffixes=('_drug', '_cell'))
print('Merged dataframe of drug (dose) response and cell lines details: ', merged_df.shape)
# Continue to merge in gene experssion using COSMIC_ID
merged_df = pd.merge(merged_df, gene_exp_transposed, on = 'COSMIC_ID', how = 'inner')
print('Merged dataframe after merging gene expression data', merged_df.shape)
merged_df = pd.merge(merged_df, compounds_df, on='DRUG_ID', how='left', suffixes=('', '_comp'))
print('Merged dataframe after merging compound data', merged_df.shape)
# Filter mutaion data to only include the mutation causing cancer
mutations = mutations[mutations['cancer_driver'] == True]
# Remove rows with unvalid model_id and gen_symbols
mutations = mutations.dropna(subset=['model_id','gene_symbol'])
# Remove duplicates
mutations = mutations.drop_duplicates(subset=['model_id', 'gene_symbol'])
# Binary matrix rows = model_id, columns = genes
mutation_matrix = (
    mutations
    .assign(present=1)
    .pivot(index='model_id', columns='gene_symbol', values='present')
    .fillna(0)
    .astype(int)
)
# Reset index to reprare for merging
mutation_matrix.reset_index(inplace=True)
print('Mutation matrix shape with cancer driver gene', mutation_matrix.shape)
# Merge the mutation dataset with merged_df on model_id and SANGER_MODEL_ID
# Ensure they are both string
merged_df['SANGER_MODEL_ID'] = merged_df['SANGER_MODEL_ID'].astype(str)
mutation_matrix['model_id'] = mutation_matrix['model_id'].astype(str)
merged_df = pd.merge(merged_df, mutation_matrix, how='left',left_on='SANGER_MODEL_ID', right_on='model_id')
print('Shape of the merged dataset with mutation data', merged_df.shape)

#DEFINE FEATURES & TARGET
merged_df = merged_df.dropna(subset=['LN_IC50'])
y_class = (merged_df['LN_IC50']< 0).astype(int)

categorical_cols = [
    'DATASET', 'CELL_LINE_NAME', 'SANGER_MODEL_ID', 'TCGA_DESC',
    'DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'WEBRELEASE',
    'Sample Name', 'Whole Exome Sequencing (WES)', 'Copy Number Alterations (CNA)',
    'Gene Expression', 'Methylation', 'Drug\nResponse',
    'GDSC\nTissue descriptor 1', 'GDSC\nTissue\ndescriptor 2',
    'Cancer Type\n(matching TCGA label)', 'Microsatellite \ninstability Status (MSI)', 'Screen Medium',
    'Growth Properties', 'DRUG_NAME_comp', 'SYNONYMS', 'TARGET', 'TARGET_PATHWAY',
    'SCREENING_SITE'
]

exclude_cols = [
    'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID', 'DRUG_ID',
    'COMPANY_ID', 'MIN_CONC', 'MAX_CONC',
    'AUC', 'RMSE', 'Z_SCORE', 'LN_IC50'
]

# Fix known mixed-type columns
mixed_type_cols = ['DRUG_NAME', 'PUTATIVE_TARGET', 'DRUG_NAME_comp', 'SYNONYMS', 'TARGET']
for col in mixed_type_cols:
    merged_df[col] = merged_df[col].astype(str)

numeric_cols = [
    col for col in merged_df.select_dtypes(include=[np.number]).columns
    if col not in exclude_cols
]

print(f"Number of categorical features: {len(categorical_cols)}")
print(f"Number of numeric features: {len(numeric_cols)}")

# STEP 3: PREPROCESS DATA
missing_rate = merged_df[numeric_cols].isnull().mean()
valid_numeric_features = missing_rate[missing_rate < 0.5].index.tolist()
print(f"Numeric features after removing high-missing columns: {len(valid_numeric_features)}")

for col in categorical_cols:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna('Unknown')

categorical_df = pd.get_dummies(merged_df[categorical_cols], drop_first=True)
print(f"One-hot encoded categorical features: {categorical_df.shape[1]}")

X_numeric = merged_df[valid_numeric_features].copy().fillna(0).astype(np.float32)
X = pd.concat([X_numeric, categorical_df], axis=1)
X = X.astype(np.float32)
print(f"Final feature matrix shape: {X.shape}")
# Train/test split
X.columns = X.columns.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost with manually chosen parameters
clf = XGBClassifier(
    n_estimators=160,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.87,
    colsample_bytree=0.61,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    tree_method='hist'  # optimized for large datasets
)

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Classifier")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot Top 20 Feature Importances for XGBoost Classifier
importances = clf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1][:20]  # Top 20 features

plt.figure(figsize=(10, 6))
plt.bar(range(20), importances[sorted_idx], align='center')
plt.xticks(range(20), feature_names[sorted_idx], rotation=90)
plt.xlabel("Feature Importance")
plt.ylabel("Importance Score")
plt.title("Top 20 Features in XGBoost Classifier")
plt.tight_layout()
plt.show()