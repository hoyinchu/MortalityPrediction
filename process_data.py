import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, AutoModel

# Returns the two df from the 
def load_raw_files():
    admission_csv_path = 'raw_data/ADMISSIONS.csv'
    patients_csv_path = "raw_data/PATIENTS.csv"
    if not (os.path.exists(admission_csv_path) and os.path.exists(patients_csv_path)):
        error_msg = """
        Either 'raw_data/ADMISSIONS.csv' or 'raw_data/PATIENTS.csv' does not exist.
        You can download a publicly available demo version of these files at https://physionet.org/content/mimiciii-demo/1.4/
        Download 'ADMISSIONS.csv' and 'PATIENTS.csv' from the website then unzip them into the raw_data directory
        For the full dataset, visit https://physionet.org/content/mimiciii/1.4/
        """
        raise FileNotFoundError(error_msg)
    
    # Convert column headers to uppercase for demo version compatibility
    admission_df = pd.read_csv(admission_csv_path)
    admission_df.columns = [col.upper() for col in admission_df.columns]
    patients_df = pd.read_csv(patients_csv_path)
    patients_df.columns = [col.upper() for col in patients_df.columns]
    return admission_df,patients_df

# Helper function that converts a given string to one of the 7 ethnicites as defined by the US census (or OTHERS)
def convert_to_census_ethnicity(og_string):
    census_ethnicities = ["WHITE", "BLACK","HISPANIC","INDIAN","HAWAIIAN","ASIAN","MULTI"]
    for eth in census_ethnicities:
        if eth in og_string:
            return eth
    return "OTHER"

def process_baseline_data():
    # Loading raw data
    admission_df,patients_df = load_raw_files()

    # Join the two tables
    patient_df_complete = patients_df.merge(admission_df,how='left',on="SUBJECT_ID")

    # Only keep the relevant columns
    columns_to_keep = ['SUBJECT_ID','GENDER','DOB','ADMISSION_TYPE','ADMITTIME','INSURANCE','ETHNICITY','DIAGNOSIS','HOSPITAL_EXPIRE_FLAG']
    patient_df_complete_reduced = patient_df_complete[columns_to_keep]
    
    # Drop patients that were admitted for more than one time. Keep the latest
    # Up tp this point we retrieve relevant information about the patients and dropped duplicates
    patient_df_complete_reduced = patient_df_complete_reduced.sort_values('ADMITTIME').reset_index(drop=True)    
    patient_df_complete_reduced = patient_df_complete_reduced.drop_duplicates(subset=['SUBJECT_ID'],keep='last')

    # We drop NEW BORN events
    patient_df_complete_reduced = patient_df_complete_reduced[patient_df_complete_reduced['ADMISSION_TYPE'] != "NEWBORN"]
    
    # Feature enginnering
    # Extract age from shifted date of birth and admit time. 
    dob_col = [datetime.datetime.strptime(dob,'%Y-%m-%d %H:%M:%S') for dob in patient_df_complete_reduced['DOB'].to_numpy()]
    admit_col = [datetime.datetime.strptime(admit,'%Y-%m-%d %H:%M:%S') for admit in patient_df_complete_reduced['ADMITTIME'].to_numpy()]
    age_col = np.array([(admit_col[i] - dob_col[i]).days // 365 for i in range(len(dob_col))])
    # Patients whose age are above 89 are convert to 300 for HIPPA complience, we convert all of them back to 90
    age_col[age_col >= 300] = 90

    patient_df_complete_reduced['AGE'] = age_col

    # Standardize ethnicity by converting all types of ethnicities into one of the nine ethnicities listed in the US census
    patient_df_complete_reduced['ETHNICITY'] = [convert_to_census_ethnicity(eth) for eth in patient_df_complete_reduced['ETHNICITY'].to_numpy()]
    
    # One hot encodes: GENDER, ADMISSION_TYPE, INSURANCE, ETHNICITY
    one_hot_gender = pd.get_dummies(patient_df_complete_reduced['GENDER'],"GENDER").reset_index(drop=True) 
    one_hot_admission = pd.get_dummies(patient_df_complete_reduced['ADMISSION_TYPE'],"ADMISSION_TYPE").reset_index(drop=True) 
    one_hot_insurance = pd.get_dummies(patient_df_complete_reduced['INSURANCE'],"INSURANCE").reset_index(drop=True) 
    one_hot_ethnicity = pd.get_dummies(patient_df_complete_reduced['ETHNICITY'],"ETHNICITY").reset_index(drop=True) 
      
    subject_id_col = patient_df_complete_reduced['SUBJECT_ID'].to_numpy()
    
    # Final processed dataset
    processed_dataset = pd.DataFrame({
        "SUBJECT_ID": subject_id_col,
        "AGE": age_col
    })
    
    new_processed_dataset = pd.concat([processed_dataset,one_hot_gender,one_hot_admission,one_hot_insurance,one_hot_ethnicity],axis=1)
    new_processed_dataset['DIAGNOSIS'] = patient_df_complete_reduced['DIAGNOSIS'].to_numpy()
    new_processed_dataset['DIED'] = patient_df_complete_reduced['HOSPITAL_EXPIRE_FLAG'].to_numpy()

    return new_processed_dataset


# Code modified from the official huggingface webpage (https://huggingface.co/distilbert-base-uncased) and
# https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
# Given a list of sentences. Convert each sentence into an embedding.
def get_text_embeddings(text_col,tokenizer,model):
    text_col_embeddings = []
    for i in tqdm(range(len(text_col))):
        cur_text = text_col[i]
        text_tokenized = tokenizer(str(cur_text),return_tensors="pt")
        text_embedding = model(**text_tokenized)[0][0,0,:].detach().numpy() # We only take the embedding for the CLS token
        text_col_embeddings.append(text_embedding)
    return text_col_embeddings

# Given a list of sentences. Convert each sentence into an embedding then return them as a dataframe
def get_text_embeddings_df(text_col,tokenizer,model):
    # distil_bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # distil_bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    text_embeddings = get_text_embeddings(text_col,tokenizer,model)
    text_embeddings = np.array(text_embeddings)
    text_embeddings_df = pd.DataFrame(data=text_embeddings)
    text_embeddings_df_cols = [f"text_dim_{i+1}" for i in range(text_embeddings.shape[1])]
    text_embeddings_df.columns = text_embeddings_df_cols
    return text_embeddings_df

# Given the baseline df, returns the text embeddings generated by distilBERT
def creates_distil_bert_embedding_csv(baseline_df):
    distil_bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distil_bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    text_col = baseline_df['DIAGNOSIS'].to_numpy()
    text_embeddings_df = get_text_embeddings_df(text_col,distil_bert_tokenizer,distil_bert_model)
    text_embeddings_df.insert(0,"SUBJECT_ID",baseline_df["SUBJECT_ID"].to_numpy())
    text_embeddings_df['DIED'] = baseline_df['DIED']
    return text_embeddings_df

# Given the baseline df, returns the text embeddings generated by bluebert
def creates_blue_bert_embedding_csv(baseline_df):
    blue_bert_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
    blue_bert_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
    text_col = baseline_df['DIAGNOSIS'].to_numpy()
    text_embeddings_df = get_text_embeddings_df(text_col,blue_bert_tokenizer,blue_bert_model)
    text_embeddings_df.insert(0,"SUBJECT_ID",baseline_df["SUBJECT_ID"].to_numpy())
    text_embeddings_df['DIED'] = baseline_df['DIED']
    return text_embeddings_df


def main():
    # Each block creates a file in the processed_data folder. Besides the first block all other
    # blocks can be commented out to avoid recalculating word embeddings

    # Process raw data from MIMIC-3 and creates the baseline df.
    print("Processing raw data from MIMIC-3 and creating the baseline dataframe")
    baseline_df = process_baseline_data()
    baseline_df.to_csv("processed_data/baseline_features.csv",index=False)

    # Creates the text embeddings from DistilBERT (https://huggingface.co/distilbert-base-uncased)
    print("Creating word embeddings from distilBERT")
    distil_bert_embedding_df = creates_distil_bert_embedding_csv(baseline_df)
    distil_bert_embedding_df.to_csv("processed_data/distil_bert_text_embedding.csv",index=False)

    # Createse the text embeddings from BlueBERT (https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12)
    print("Creating word embeddings from BlueBERT")
    blue_bert_embedding_df = creates_blue_bert_embedding_csv(baseline_df)
    blue_bert_embedding_df.to_csv("processed_data/blue_bert_text_embedding.csv",index=False)


if __name__ == "__main__":
    main()