📦leoss-cov-19
 ┣ 📂env
 ┃ ┗ 📜environment.yml # conda environment definition

 ┣ 📂models
 ┃ ┣ 📂CV # final model related data
 ┃ ┃ ┣ 📜leoss_weiAft_CV_death_CV.pkl.gz # final Weibull AFT-model
 ┃ ┃ ┣ 📜leoss_weiAft_CV_SHAP-explainer-medianlifetime-n1000.pkl.gz # SHAP explainer final model
 ┃ ┃ ┗ 📜leoss_weiAft_CV_SHAP-shapvalues-medianlifetime-n1000.pkl.gz # SHAP values final model
 ┃ ┣ 📂nCV # nested cross-validation outer-fold Weibull AFT-models
 ┃ ┃ ┣ 📜leoss_weiAft_death_ofold0.gzip.pkl
 ┃ ┃ ┣ 📜leoss_weiAft_death_ofold1.gzip.pkl
 ┃ ┃ ┣ 📜leoss_weiAft_death_ofold2.gzip.pkl
 ┃ ┃ ┣ 📜leoss_weiAft_death_ofold3.gzip.pkl
 ┃ ┃ ┗ 📜leoss_weiAft_death_ofold4.gzip.pkl 

 ┣ 📂results
 ┃ ┣ 📂EDA # explorative data analysis
 ┃ ┃ ┣ 📜EDA_demographis.xlsx
 ┃ ┃ ┣ 📜leoss_eda_kmplot.py
 ┃ ┃ ┣ 📜leoss_kaplan_deathcov19.pdf
 ┃ ┃ ┗ 📜leoss_sweetviz_18-85_onDeath.html
 ┃ ┣ 📂finalmodel # final model SHAP-plots and -values, calibration plots, AFT coefficients
 ┃ ┃ ┣ 📜finalmodel_AFT_Coefficients.csv
 ┃ ┃ ┣ 📜shapf_rank1_Age.pdf
 ┃ ┃ ┣ 📜shapf_rank2_Asymptomatic.pdf
 ┃ ┃ ┣ 📜shap_perfeature.csv
 ┃ ┃ ┣ 📜shap_perfeature.xlsx
 ┃ ┃ ┣ 📜shap_rank10_Gender.pdf
 ┃ ┃ ┣ 📜shap_rank1_Age.pdf
 ┃ ┃ ┣ 📜shap_rank2_Asymptomatic.pdf
 ┃ ┃ ┣ 📜shap_rank3_Blood Oxygen Saturation.pdf
 ┃ ┃ ┣ 📜shap_rank46_Dementia.pdf
 ┃ ┃ ┣ 📜shap_rank4_Hemato-Lab. Hemoglobin.pdf
 ┃ ┃ ┣ 📜shap_rank5_Lab Troponine-T.pdf
 ┃ ┃ ┣ 📜shap_rank6_Symptom Muscle Ache.pdf
 ┃ ┃ ┣ 📜shap_rank7_Lab Ferrit.pdf
 ┃ ┃ ┣ 📜shap_rank8_Lab CRP.pdf
 ┃ ┃ ┣ 📜shap_rank9_Hemato-Lab. Platelets.pdf
 ┃ ┃ ┣ 📜weiAft_CV_AFT-interpretation.html
 ┃ ┃ ┣ 📜weiAft_CV_AFT-interpretation.ipynb
 ┃ ┃ ┣ 📜weiAft_CV_calibration.py
 ┃ ┃ ┣ 📜weiAft_CV_SHAP-n1000-medianlife.ipynb
 ┃ ┃ ┣ 📜weiAft_CV_SHAP-plots.ipynb
 ┃ ┃ ┣ 📜weiAft_CV_survival_probability_calibration_t14.pdf
 ┃ ┃ ┣ 📜weiAft_CV_survival_probability_calibration_t21.pdf
 ┃ ┃ ┣ 📜weiAft_CV_survival_probability_calibration_t28.pdf
 ┃ ┃ ┣ 📜weiAft_CV_survival_probability_calibration_t35.pdf
 ┃ ┃ ┣ 📜weiAft_CV_survival_probability_calibration_t7.pdf
 ┃ ┃ ┗ 📜weiAft_CV_survival_probability_calibration_t80.pdf
 ┃ ┣ 📂nCV # nested cross validation test set performances across models
 ┃ ┃ ┣ 📜metrics_nCV.csv
 ┃ ┃ ┣ 📜metrics_nCV.xlsx
 ┃ ┃ ┣ 📜metrics_nCV_cuno.csv
 ┃ ┃ ┣ 📜metrics_nCV_cuno.joblib.zip
 ┃ ┃ ┣ 📜metrics_nCV_dynuno.csv
 ┃ ┃ ┣ 📜metrics_nCV_dynuno.joblib.zip
 ┃ ┃ ┣ 📜nCVAft_nCV_survival_probability_calibration_fold0.pdf
 ┃ ┃ ┣ 📜nCVAft_nCV_survival_probability_calibration_fold1.pdf
 ┃ ┃ ┣ 📜nCVAft_nCV_survival_probability_calibration_fold2.pdf
 ┃ ┃ ┣ 📜nCVAft_nCV_survival_probability_calibration_fold3.pdf
 ┃ ┃ ┣ 📜nCVAft_nCV_survival_probability_calibration_fold4.pdf
 ┃ ┃ ┣ 📜results.ipynb
 ┃ ┃ ┣ 📜results_nCV_boxplot-legend.pdf
 ┃ ┃ ┣ 📜results_nCV_boxplot_charrell.pdf
 ┃ ┃ ┣ 📜results_nCV_boxplot_cuno.pdf
 ┃ ┃ ┣ 📜results_nCV_boxplot_ibs.pdf
 ┃ ┃ ┣ 📜results_uno_dynamic_auc.pdf
 ┃ ┃ ┗ 📜weiAft_nCV_calibration.py
 
 ┣ 📂src
 ┃ ┣ 📂data # explorative data analysis and preprocessing 
 ┃ ┃ ┃ ┗ 📜explorative data analysis-checkpoint.ipynb
 ┃ ┃ ┃ ┗ 📜decode_columns.ipynb
 ┃ ┣ 📂models # model training scripts
 ┃ ┃ ┣ 📜cox_nCV.py
 ┃ ┃ ┣ 📜create_mlflow_parentruns.py
 ┃ ┃ ┣ 📜deepsurv_nCV_arrayjob.py
 ┃ ┃ ┣ 📜evaluate_models_nCV-cUno.py
 ┃ ┃ ┣ 📜evaluate_models_nCV.py
 ┃ ┃ ┣ 📜evaluate_models_weiAft_dynamicAUC.py
 ┃ ┃ ┣ 📜rsf_death_nCV.py
 ┃ ┃ ┣ 📜weiAft_CV.py
 ┃ ┃ ┣ 📜weiAft_nCV.py
 ┃ ┃ ┣ 📜xgbse_sWeiAft_nCV.py
 ┃ ┃ ┣ 📜xgb_aft_nCV.py
 ┃ ┣ 📂models_interpretation # SHAP training scripts
 ┃ ┃ ┗ 📜weiAft_CV_SHAP-medianlife.py 
 
 ┣ 📂utils # utility functions
 ┃ ┣ 📜general.py
 ┃ ┣ 📜wrappers.py    
 # batch job submission files
 ┣ 📜sbatch_cox_nCV.sh
 ┣ 📜sbatch_deepsurv_nCV.sh
 ┣ 📜sbatch_rsf.sh
 ┣ 📜sbatch_weiAft_CV.sh
 ┣ 📜sbatch_weiAft_nCV.sh
 ┣ 📜sbatch_xgbse_nCV.sh
 # this file
 ┣ 📜README.md 