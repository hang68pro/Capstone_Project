# Capstone Project May 2025
Drug Sensitivity in Lung Adenocarcinoma Using Machine Learning 

This project focuses on predicting drug sensitivity in Lung Adenocarcinoma (LUAD) cell lines using machine learning models

**Key Aspects:**
* **Objective:** To predict the sensitivity of LUAD cancer cell lines to drugs using genetic data. The study explores both regression models (predicting half-maximal inhibitory concentration, IC50 values) and classification models (categorizing samples as sensitive or resistant). It also investigates the impact of incorporating somatic mutation data on predictive performance and interpretability.
* **Data Sources:** Publicly available datasets from the Genomics of Drug Sensitivity in Cancer (GDSC) project were used, including drug sensitivity measures (LN IC50 values), cell line metadata, gene expression values, binary mutation statuses for cancer driver genes, and information on screened compounds.
* **Methodology:**
    * **Data Merging:** Various datasets were combined, filtering for LUAD cell lines, transposing gene expression data, merging drug-level data, and processing mutation data to create a comprehensive multi-omic resource.
    * **Preprocessing and Feature Engineering:** Steps included removing features with high proportions of missing values, one-hot encoding categorical features, and standardizing numerical data using StandardScaler. Principal Component Analysis (PCA) was applied in selected pipelines for dimensionality reduction.
    * **Exploratory Data Analysis (EDA):**
        * Analysis of natural logarithm of IC50 (LN_IC50) values showed a slightly left-skewed distribution, justifying the use of log-transformed values and tree-based models like XGBoost. A threshold of LN_IC50 < 0 was set for the "sensitive" class and LN_IC50 ≥ 0 for the "resistant" class.
        * Analysis of driver gene mutations revealed TP53 as the most frequently mutated (81%), followed by KRAS (34%).
        * Point-biserial correlation analysis between mutation status and LN_IC50 identified genes correlated with drug resistance (e.g., ASXL2, PALB2) and drug sensitivity (e.g., STK11, KEAP1).
        * Pathway-level enrichment showed that chromatin histone acetylation, metabolism, and mitosis pathways were highly active in sensitive cell lines, while protein stability and degradation, hormone-related pathways, and genome integrity mechanisms were associated with resistant cell lines.
        * Drug-level efficacy analysis identified Romidepsin, Sepantronium bromide, Bortezomib, Dactinomycin, and SN-38 as the most effective compounds in LUAD cell lines.
* **Machine Learning Models:**
    * **Regression Models (XGBoost):**
        * Models were trained with and without PCA and mutation data.
        * Without PCA and mutation data: Top features included RNA polymerase targets and drugs like Sepantronium bromide.
        * Without PCA and with mutation data: ASXL1 mutation emerged as an important predictor alongside expression-based features.
        * With PCA and without mutation data: Achieved an R² of 0.7943, RMSE of 1.2418, and MAE of 0.9345, showing good performance with reduced dimensionality.
        * With PCA and with mutation data: Showed marginal improvement in performance (R² = 0.7976, RMSE = 1.2316, MAE = 0.9321).
        * Lasso regression underperformed compared to XGBoost.
    * **Classification Models (XGBoost):**
        * Models were trained with and without PCA and mutation data[.
        * Without PCA and mutation data: Achieved a high ROC AUC of 0.97 and 96% accuracy. RNA polymerase was the most important feature.
        * Without PCA and with mutation data: Similar performance (ROC AUC = 0.9708) with the addition of features related to BCL-2 family proteins.
        * PCA severely impaired interpretability and lowered performance in classification models.
* **Results Summary:** Gene expression profiles were found to be more important than mutation status alone for predicting treatment response. Mutation features contributed minimally to performance but provided additional biological insight. PCA improved speed for regression but reduced interpretability and dramatically lowered classification performance.
* **Limitations and Future Work:** The study was limited to LUAD-specific cell lines from the GDSC database and did not include other data types like proteomics or epigenomics. Future work could expand the dataset and incorporate additional data types.
