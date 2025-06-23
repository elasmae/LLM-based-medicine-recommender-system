# AI-Powered Medicine Recommender

**LLM-Based Symptom Analysis, Summarization & Clustering**

This is an AI-driven application that recommends alternative medicines based on semantic similarity, symptom interpretation, and side-effect profiling.
It integrates advanced **Natural Language Processing (NLP)**, **Large Language Models (LLMs)**, and **interactive visualization** , all accessible through an intuitive Gradio web interface.

---

## Features

* **Hybrid medicine recommendation** using BERT embeddings and cosine similarity.
* **Symptom Checker Chatbot** powered by a GPT-based LLM to understand symptoms and suggest treatments.
* **Medical Notice Summarizer** using DistilBART to simplify long drug descriptions.
* **t-SNE-based clustering** to explore groups of similar medicines visually.
* **Multilingual support** with automatic translation of symptoms from French to English.


---

## Demo Screenshots

<img width="466" alt="screenshot1" src="https://github.com/user-attachments/assets/6161a9db-24aa-4dd9-b085-1ebd394be768" />

<img width="569" alt="project medcine" src="https://github.com/user-attachments/assets/143fb013-7810-424f-93cb-57c819121419" />

---

## Project Structure

<pre>
├── preprocess_medicines.py   # Processing, embedding, and clustering pipeline  
├── app.py                    # Main Gradio application  
├── screenshot.png            # Interface demo image  
├── bert_embeddings.pkl       # Medicine embeddings  
├── cosine_sim.pkl            # Cosine similarity matrix  
├── processed_df.pkl          # Preprocessed medicine dataset  
├── medicine_details.csv      # Raw dataset of medicines  
│                             # (https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details)  
├── requirements.txt          # All required Python packages  
└── README.md                 # This file  
</pre>


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
