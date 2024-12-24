# MMT_ML
This repository contains the data analysis code for a study investigating the efficacy of acupuncture in enhancing methadone maintenance treatment (MMT) dosage reduction. 

## Overview
Long-term methadone maintenance treatment (MMT) often involves dosage reduction, which can increase the risk of relapse. Acupuncture has shown potential in improving methadone reduction and alleviating opioid cravings. In this study, we analyzed data from two randomized controlled trials with 197 patients undergoing MMT dosage tapering at six MMT clinics in China. The patients were divided into acupuncture and non-acupuncture groups based on their treatment type. The study used clustered pre-intervention methadone dose trajectories as predictors along with baseline patient data to predict the composite outcome of methadone dose reduction and opioid craving scores.

Machine learning models, including CatBoost, were trained to predict these outcomes, and SHapley Additive exPlanations (SHAP) analysis was performed to determine the impact of different predictors. Exploratory analysis revealed that acupuncture treatment was more effective in patients with certain methadone dose trajectories.

Key Findings
Methadone Dose Trajectories: The methadone dose data were clustered into three distinct trajectories.

Exploratory Insights: Patients with increasing and then decreasing methadone dose trajectories showed significantly better response to acupuncture treatment compared to those with decreasing and then increasing dose trajectories.

Repository Contents

Data Preparation Scripts: Code for preprocessing and clustering the methadone dose data.

SHAP Analysis: Code for SHAP analysis to assess the contribution of each predictor to the model's predictions.

Exploratory Data Analysis: Scripts for performing exploratory analysis on acupuncture treatment efficacy in relation to dose trajectories.

This repository provides a comprehensive framework for analyzing MMT dosage reduction and acupuncture efficacy, offering insights into personalized acupuncture treatment strategies based on methadone dose trajectories.

## Dependencies
Python 3.7

pandas

scikit-learn

pycaret

SHAP

matplotlib

seaborn

Usage

## Clone this repository.
Install the required dependencies using pip install -r requirements.txt.

Run the data preprocessing and analysis scripts to reproduce the results from the study.

For further questions or collaborations, feel free to open an issue or reach out.
