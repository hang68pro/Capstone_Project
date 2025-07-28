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

#Show the first 5 rows of the final merged dataset
merged_df.head()
#Get info on each column type and non-null values
merged_df.info()
#Distribution of natural log of IC50
plt.hist(merged_df['LN_IC50'].dropna(), bins=50, color='skyblue')
plt.title('Distribution of LN_IC50')
plt.xlabel('LN_IC50')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Mutation columns are new ones added, and are binary values (0/1)
# Get a list of the added mutation gene columns by checking those not in the original columns
original_columns = [
    'DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID',
    'CELL_LINE_NAME', 'SANGER_MODEL_ID', 'LN_IC50', 'AUC', 'RMSE', 'Z_SCORE'
]
mutation_columns = [col for col in merged_df.columns if col not in original_columns and merged_df[col].dropna().isin([0, 1]).all()]
print(f"Number of binary mutation columns: {len(mutation_columns)}")
print("Example mutation columns:", mutation_columns[:10])
# What % of cell lines have a mutation in each gene
mutation_presence = merged_df[mutation_columns].mean().sort_values(ascending=False)
print(mutation_presence.head(10))  # Most common driver gene mutations
# Visualize top 10 most mutated driver genes
mutation_presence.head(10).plot(kind='bar', color='coral')
plt.title('Top 10 Most Frequently Mutated Driver Genes')
plt.ylabel('% of LUAD Cell Lines with Mutation')
plt.grid(True)
plt.show()
# Filter for mutation columns ending in _y
mutation_cols = [col for col in merged_df.columns if col.endswith('_y') and merged_df[col].nunique() == 2]
# Store correlations in a list
correlations = []
for gene in mutation_cols:
    try:
        corr, pval = pointbiserialr(merged_df[gene], merged_df['LN_IC50'])
        correlations.append((gene, corr, pval))
    except Exception as e:
        print(f"Error processing {gene}: {e}")

# Convert to DataFrame
corr_df = pd.DataFrame(correlations, columns=['Gene', 'Correlation', 'P-value'])
# Sort by absolute correlation strength
corr_df = corr_df.sort_values(by='Correlation', key=lambda x: abs(x), ascending=False)
# Show top 15 genes with strongest correlations (positive or negative)
top_corr = corr_df.head(15)
print(top_corr)
# Mutated Genes Correlated with Drug Sensitivity
plt.figure(figsize=(10, 5))
plt.bar(top_corr['Gene'], top_corr['Correlation'], color='coral')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Point-Biserial Correlation with LN_IC50')
plt.title('Top 15 Mutated Genes Correlated with Drug Sensitivity')
plt.tight_layout()
plt.show()

# Drugs most sensitive to LUAD
# Group by drug name and calculate average sensitivity
drug_effectiveness = merged_df.groupby('DRUG_NAME')['LN_IC50'].mean().sort_values()
# Show top 5 most effective drugs (lowest LN_IC50)
top_effective_drugs = drug_effectiveness.head(5)
print(top_effective_drugs)
# Bar plot of the top 5 most effective drugs in LUAD cell lines
top_effective_drugs.plot(kind='barh', figsize=(8,6), color='teal')
plt.xlabel('Average LN_IC50 (Lower = More Effective)')
plt.title('Top 5 Most Effective Drugs in LUAD Cell Lines')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#Plot for correlation between top 20 pathways in two separate groups based on LN_IC50 values
# Split data into two groups based on LN_IC50 values
group_a = merged_df[merged_df['LN_IC50'] < 0] #sensitive
group_b = merged_df[merged_df['LN_IC50'] > 0] #resistant
# Mean LN_IC50 per pathway for each group
group_a_mean = group_a.groupby('TARGET_PATHWAY')['LN_IC50'].mean().sort_values().head(20)
group_b_mean = group_b.groupby('TARGET_PATHWAY')['LN_IC50'].mean().sort_values().head(20)
# Barplot of the top 20 pathways in each group
fig, axs = plt.subplots(2, 1, figsize = (10,10))
sns.barplot(x=group_a_mean.values,y= group_a_mean.index, ax=axs[0])
axs[0].set_title('Top 20 Pathways In Sensitive Cell Line (LN_IC50<0)')
axs[0].set_xlabel('Mean LN_IC50')
axs[0].set_ylabel('Target Pathway')

sns.barplot(x=group_b_mean.values, y=group_b_mean.index, ax=axs[1])
axs[1].set_title('Top 20 Pathways in Resistant Cell Line (LN_IC50 > 0)')
axs[1].set_xlabel('Mean LN_IC50')
axs[1].set_ylabel('Target Pathway')
plt.tight_layout()
plt.show()