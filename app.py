import gradio as gr
import pandas as pd
import pickle
import plotly.express as px
from sklearn.manifold import TSNE
from transformers import pipeline
from deep_translator import GoogleTranslator

# Chargement des fichiers pickle
with open('processed_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Chargement pipeline LLM
chatbot_pipe = pipeline("text-generation", model="distilgpt2")

from transformers import pipeline

# Modèle de résumé (chargé une fois au démarrage)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
def summarize_notice(notice_text):
    if not notice_text.strip():
        return "Please enter a valid notice text."

    try:
        result = summarizer(notice_text, max_length=130, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"An error occurred: {e}"
    
# Fonction de recommandation hybride
def get_hybrid_recommendations(medicine_name=None, symptom=None, top_n=5):
    if medicine_name:
        idx = df[df['medicine_name'].str.lower() == medicine_name.lower()].index
        if len(idx) == 0:
            if symptom:
                idx = df[df['conditions'].str.lower().str.contains(symptom.lower(), na=False)].index
                if len(idx) == 0:
                    return None, "No medicines found for the given symptom.", None
                idx = idx[0]
            else:
                return None, "Medicine not found in the dataset.", None
        idx = idx[0]
    elif symptom:
        idx = df[df['conditions'].str.lower().str.contains(symptom.lower(), na=False)].index
        if len(idx) == 0:
            return None, "No medicines found for the given symptom.", None
        idx = idx[0]
    else:
        return None, "Please provide a medicine name or symptom.", None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    medicine_indices = [i[0] for i in sim_scores]
    recommendations = []
    for idx2 in medicine_indices:
        med_name = df['medicine_name'].iloc[idx2]
        shared_conditions = set(df['conditions'].iloc[idx2].split('; ')) & set(df['conditions'].iloc[idx].split('; '))
        explanation = f"Treats similar conditions: {', '.join(shared_conditions)}" if shared_conditions else "Similar profile"
        recommendations.append({
            'Medicine': med_name,
            'Conditions': df['conditions'].iloc[idx2],
            'Side Effects': df['side_effects'].iloc[idx2],
            'Administration': df['administration'].iloc[idx2],
            'Medicine Type': df['medicine_type'].iloc[idx2],
            'Explanation': explanation,
            'Similarity Score': sim_scores[medicine_indices.index(idx2)][1]
        })

    rec_df = pd.DataFrame(recommendations)

    fig = px.bar(rec_df, x='Medicine', y='Similarity Score',
                 title='Similarity Scores of Recommended Medicines',
                 labels={'Similarity Score': 'Cosine Similarity'}, color='Similarity Score')
    fig.update_layout(xaxis_title="Medicine", yaxis_title="Cosine Similarity")

    cluster_plot = get_cluster_plot(idx)

    return rec_df[['Medicine', 'Conditions', 'Side Effects', 'Administration', 'Medicine Type', 'Explanation']], fig, cluster_plot

# Fonction pour visualiser les clusters avec t-SNE
def get_cluster_plot(selected_idx=None):
    fig = px.scatter(df, x="tsne_x", y="tsne_y", color="cluster",
                     hover_data=["medicine_name", "conditions"],
                     title="t-SNE Visualization of Medicine Clusters")
    if selected_idx is not None:
        selected = df.iloc[selected_idx]
        fig.add_scatter(x=[selected['tsne_x']], y=[selected['tsne_y']],
                        mode='markers+text', marker=dict(size=12, color='red'),
                        text=["Selected"], textposition="top center", showlegend=False)
    return fig



def symptom_chatbot(symptom_text, top_n=5):
    if not symptom_text.strip():
        return pd.DataFrame([{"Message": "Please describe your symptoms."}])

    # Traduction FR -> EN
    translated = GoogleTranslator(source='auto', target='en').translate(symptom_text)
    keywords = translated.lower().split(',')

    matched = df[df['conditions'].str.contains('|'.join(keywords), case=False, na=False)]

    if matched.empty:
        return pd.DataFrame([{"Message": "No medicines found matching these symptoms."}])

    return matched[['medicine_name', 'conditions', 'side_effects', 'administration']].head(top_n)


# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## AltRx Alternative Medicine Recommender")

    with gr.Row():
        medicine_input = gr.Dropdown(choices=[""] + sorted(df['medicine_name'].unique()), label="Select Medicine")
        symptom_input = gr.Textbox(label="Enter Symptom (optional)", placeholder="e.g., pneumonia, cough")

    recommend_button = gr.Button("Get Recommendations")
    output_table = gr.Dataframe(label="Recommended Medicines", interactive=False)
    output_plot = gr.Plot(label="Similarity Scores")
    output_cluster = gr.Plot(label="Cluster Visualization (t-SNE)")

    def handle_recommendation(med, symp):
        result, fig, cluster = get_hybrid_recommendations(med, symp)
        if isinstance(result, str) or result is None:
            return gr.update(value=pd.DataFrame()), gr.update(value=None), gr.update(value=None)
        return result, fig, cluster

    recommend_button.click(fn=handle_recommendation, inputs=[medicine_input, symptom_input],
                           outputs=[output_table, output_plot, output_cluster])

    with gr.Accordion(" Symptom Checker Chatbot", open=False):
        symptom_text_input = gr.Textbox(label="Describe your symptoms", placeholder="e.g., fever, sore throat, cough")
        symptom_response = gr.Dataframe(label="Suggested Medicines")
        symptom_button = gr.Button("Analyze Symptoms")
        symptom_button.click(fn=symptom_chatbot, inputs=symptom_text_input, outputs=symptom_response)

    gr.Markdown("---")
    gr.Markdown("Built with Gradio and powered by **Asmae EL MAHJOUBI**")

    with gr.Accordion(" Medical Notice Summarizer", open=False):
        gr.Markdown("Paste any long medical notice or description below to get a short and clear summary.")
        notice_input = gr.Textbox(label="Paste Medical Notice", lines=10, placeholder="e.g., full drug description or usage guidelines...")
        summary_output = gr.Textbox(label="Summary", lines=5)
        summarize_btn = gr.Button("Summarize")
        summarize_btn.click(fn=summarize_notice, inputs=notice_input, outputs=summary_output)

demo.launch()
