# Using Text Embedding to Improve Mortality Prediction in MIMIC-3

This is a project repository for the class CS 4120 Natural Language Process at Northeastern University

In this project, we explore the utility of doctor’s notes in improving the accuracy of a mortality prediction system that only contains basic demographic features about each patient. We used the MIMIC-3 dataset and extracted the demographic features for each patient, then created additional features by passing the text that contains the patient’s diagnosis through a BERT architecture. We then compared training a logistic regression and random forest model using these features and evaluated their performance. We find that these text embedding does improve the overall performance of both LR and RF models.

A fully ran notebook can be found under notebook_results/

## How to run 

Step 1:
Go to https://physionet.org/content/mimiciii/1.4/ and request access to the MIMIC-3 Data. After you have been approved access to the data download ADMISSIONS.csv.gz and PATIENTS.csv.gz

Step 1 Alternative:
Go to https://physionet.org/content/mimiciii-demo/1.4/ and download ADMISSIONS.csv and PATIENTS.csv. Note that this is the demo version of the full dataset which is a lot smaller and is not what our report is based on. This should only be done to verify that the code runs properly.

Step 2:
unzip ADMISSIONS.csv.gz and PATIENTS.csv.gz into csv and put ADMISSIONS.csv and PATIENTS.csv in the raw_data directory

Step 3:
Run process_data.py. After the script is finished running, you should be able to see 3 files named "baseline_features.csv", "blue_bert_text_embedding.csv", "distil_bert_text_embedding.csv" under the processed_data directory

Step 4:
You can now run the notebook cell by cell and reproduce the results