
# AI-Powered Medicine Recommender with LLM-Based Symptom Analysis, Summarization & Clustering

This is an AI-driven application that recommends alternative medicines based on semantic similarity, symptom interpretation, and side-effect profiling.  
It integrates advanced **Natural Language Processing (NLP)**, **Large Language Models (LLMs)**, and **interactive visualization** - all accessible through an intuitive Gradio web interface.

---

## Features


- **Hybrid medicine recommendation** using BERT embeddings and cosine similarity.
- **Symptom Checker Chatbot** powered by a GPT LLM to understand symptoms and suggest treatments.
- **Medical Notice Summarizer** using DistilBART to simplify long drug descriptions.
- **t-SNE-based clustering** to explore groups of similar medicines visually.
- **Multilingual support** with automatic translation of symptoms from French to English.
- Built with Transformers, spaCy, scikit-learn, Plotly, and Gradio.

---

## Demo Screenshots

### Main Recommendation Interface


![Screenshot](screenshot.png)



### Symptom Checker Example (input in French)
```text
Input: fièvre, toux, fatigue
Output: Recommends medicines treating flu and respiratory conditions.

Project Structure

├── preprocess_medicines.py # processing, embedding, and clustering pipeline
├── app.py                     # Main Gradio application
├── screenshot.png             # Interface demo image
├── bert_embeddings.pkl        # Medicine embeddings
├── cosine_sim.pkl             # Cosine similarity matrix
├── processed_df.pkl           # Preprocessed medicine dataset
├── medicine_details.csv       # Raw dataset of medicines (https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details)
├── requirements.txt           # All required Python packages
└── README.md                  # This file


- How to Use

- Step 1 - Preprocess the data

python processing_medicines.py

This generates:

processed_df.pkl

bert_embeddings.pkl

cosine_sim.pkl

- Step 2 - Launch the app

python app.py


- Technologies Used

Tool	Purpose

spaCy	Entity extraction and text preprocessing

Transformers	BERT (for embeddings), GPT, BART

scikit-learn	Similarity, clustering, and t-SNE

Plotly	Interactive cluster visualizations

Gradio	Web interface

deep-translator	Translate symptoms from French to English

GPU processeur

- Example Use Cases

Recommending drug alternatives for specific conditions

Providing medication suggestions from symptom descriptions

Summarizing medical notices for patient-friendly language

Visualizing therapeutic similarity between drugs

- Notes : Summary generation and chatbot use lightweight models to avoid deployment overhead and Compatible with both CPU and GPU environments.
