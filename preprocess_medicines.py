
import pandas as pd
import re
import spacy
import numpy as np
import torch


import pickle
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px



nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

df = pd.read_csv("medicine_details.csv")
df.columns = df.columns.str.strip() 

# Création d'une colonne 'medicine_description'
df['medicine_description'] = (
    df['Medicine Name'].fillna('') + ' is a medicine composed of ' +
    df['Composition'].fillna('') + '. It is used for ' +
    df['Uses'].fillna('') + '. Possible side effects include ' +
    df['Side_effects'].fillna('') + '.'
)

# ---------- Fonctions de traitement ----------

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)


def extract_conditions(text):
    doc = nlp(text)
    conditions = [ent.text for ent in doc.ents if ent.label_ in ['DISEASE', 'SYMPTOM']]
    pattern = r'(used to treat|treatment of|relieves|helps to relieve)(.*?)(?=\.|,|and other|such as|\n)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        conditions.append(match[1].strip())
    return '; '.join(set(conditions)) if conditions else 'unknown'

def extract_side_effects(text):
    doc = nlp(text)
    side_effects = []
    for sent in doc.sents:
        if any(k in sent.text.lower() for k in ['side effect', 'may cause']):
            side_effects.extend([token.text for token in sent if token.pos_ in ['NOUN', 'ADJ']])
    pattern = r'(side effects.*?include|may cause)(.*?)(?=\.|before taking|consult|\n)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        side_effects.append(match[1].strip())
    return '; '.join(set(side_effects)) if side_effects else 'none'

def extract_administration(text):
    pattern = r'(taken orally|injection|applied|used as|should be taken)(.*?)(?=\.|,|\n)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return '; '.join([m[0] + ' ' + m[1].strip() for m in matches]) if matches else 'unknown'

def extract_medicine_type(text):
    types = {
        'antibiotic': ['antibiotic', 'penicillin'],
        'antihistamine': ['antihistamine', 'anti-allergy'],
        'cough': ['cough', 'expectorant'],
        'analgesic': ['pain', 'analgesic'],
        'antiviral': ['virus', 'antiviral']
    }
    text = text.lower()
    for typ, keywords in types.items():
        if any(k in text for k in keywords):
            return typ
    return 'other'

def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

# ---------- Application des traitements ----------

print(" Traitement des descriptions...")

df['processed_description'] = df['medicine_description'].apply(preprocess_text)
df['conditions'] = df['medicine_description'].apply(extract_conditions)
df['side_effects'] = df['medicine_description'].apply(extract_side_effects)
df['administration'] = df['medicine_description'].apply(extract_administration)
df['medicine_type'] = df['medicine_description'].apply(extract_medicine_type)
df['medicine_name'] = df['Medicine Name']

# Combinaison des champs
df['combined_features'] = (
    df['processed_description'] + ' ' +
    df['conditions'] + ' ' +
    df['side_effects'] + ' ' +
    df['administration'] + ' ' +
    df['medicine_type']
)

print("Génération des embeddings BERT...")
bert_embeddings = get_bert_embeddings(df['combined_features'])

print("Calcul des similarités cosinus...")
cosine_sim = cosine_similarity(bert_embeddings)

print("Clustering des médicaments...")
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(bert_embeddings)

# Sauvegarder les coordonnées t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_coords = tsne.fit_transform(bert_embeddings)
df['tsne_x'] = tsne_coords[:, 0]
df['tsne_y'] = tsne_coords[:, 1]

# ---------- Sauvegarde ----------
with open("processed_df.pkl", "wb") as f:
    pickle.dump(df, f)
with open("bert_embeddings.pkl", "wb") as f:
    pickle.dump(bert_embeddings, f)
with open("cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

print("Données sauvegardées : processed_df.pkl, bert_embeddings.pkl, cosine_sim.pkl")


print(" Données sauvegardées : processed_df.pkl, bert_embeddings.pkl, cosine_sim.pkl")
