import streamlit as st
import sys
import os
import plotly.express as px
from pathlib import Path

path = str(Path(__file__).resolve().parent.parent.parent / "src")
sys.path.append(
    path
)
print(sys.path)

from data_preprocessing import TextPreprocessor
from sentiment_model import SentimentAnalyzer
from model_evaluation import evaluate_model
from parameter_tuning import (tune_hyperparameters, get_best_model)

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide", page_icon="😊")

# Initialze session state
if "analyzer" not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
    st.session_state.preprocessor = TextPreprocessor()


def main():
    st.title("Sentiment Analysis Application")
    st.markdown("### Sentiment Analysis using Local and Ollama Models")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature",
        ["Sentiment Analysis", "Document Chat", "Model Comparison", "About"],
    )

    if page == "Sentiment Analysis":
        sentiment_analysis_page()
    elif page == "Document Chat":
        document_chat_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "About":
        about_page()


def sentiment_analysis_page():
    st.header("Sentiment Analysis")
    input_text = st.text_area("Enter text for sentiment analysis:", height=150)

    if st.button("Analyze Sentiment", type="primary"):
        if input_text.strip() == "":
            st.warning("Please enter some text for analysis.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    # Load sample data (if needed) and get the best model predictions
                    df = st.session_state.analyzer.load_sample_data()

                    # Train models if not already trained
                    if st.session_state.analyzer.best_model is None:
                        st.session_state.analyzer.train_models(df)

                    best_models = tune_hyperparameters(df.text, df.sentiment)
                    best_model = get_best_model(best_models)
                    st.info(f"Best Model Selected: {best_model}")
                    local_result = st.session_state.analyzer.predict_sentimental_local(
                        input_text, model_name=best_model
                    )
                    ollama_result = (
                        st.session_state.analyzer.predict_sentimental_ollama(input_text)
                    )

                st.subheader("Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Local Model Prediction:**")
                    st.json(local_result)
                with col2:
                    st.markdown("**Ollama Model Prediction:**")
                    st.json(ollama_result)

                # Probablity chart for local model
                st.subheader("Local Model Prediction Probabilities")
                fig = px.bar(
                    x=list(local_result["probabilities"].keys()),
                    y=list(local_result["probabilities"].values()),
                    labels={"x": "Sentiment", "y": "Probability"},
                    title="Prediction Probabilities",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

def document_chat_page():
    st.header("Document Chat")
    uploaded_file = st.file_uploader(
        "Upload a document (txt, pdf, docx):",
        type=["txt", "pdf", "docx"],
        help="Upload a text, PDF, or Word document for analysis.",
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Process the document
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    st.session_state.document_prossed = True
                    text = st.session_state.preprocessor.extract_text(uploaded_file)
                    st.session_state.analyzer.load_and_process_document(text)
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        
        # Chat interface
        if hasattr(st.session_state, 'document_prossed') and st.session_state.document_prossed:
            st.subheader("Chat with the Document")
            user_query = st.text_input("Enter your question about the document:")

            if st.button("Get Answer", type="primary"):
                if user_query.strip() == "":
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Getting answer..."):
                        try:
                            answer = generate_document_answer(user_query, text)
                            st.markdown("**Answer:**")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error getting answer: {e}")

def generate_document_answer(query, document):
    """ Generate answer from the document using the ollama model """
    try:
        import ollama
        prompt = f"""
        You are a helpful assistant. Use the following document to answer the question.
        Document: {document}
        Question: {query}
        Answer:
        """
        response = ollama.generate(model="gpt-oss:20b", prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error generating answer: {e}"

def model_comparison_page():
    st.header("Model Comparison and Evaluation")

    testSentences = [
        ("I love sunny days and beautiful weather!", "positive"),
        ("I hate getting stuck in traffic jams.", "negative"),
        ("The book is on the table.", "neutral"),
        ("The movie was fantastic and I enjoyed it a lot!", "positive"),
        ("The food was terrible and I will never go back there.", "negative"),
        ("It's an average day, nothing special happening.", "neutral"),
        ("The service at the restaurant was excellent!", "positive"),
        ("I am very disappointed with the product quality.", "negative"),
        ("The presentation was okay, not too bad.", "neutral"),
        ("What a wonderful experience, I had a great time!", "positive"),
    ]

    if st.button("Evaluate Local Model", type="primary"):
        with st.spinner("Evaluating Local Model..."):
            result = []
            try:
                for sentence in testSentences:
                    local_result = st.session_state.predict_sentimental_local(sentence[0])
                    ollama_result = st.session_state.predict_sentimental_ollama(sentence[0])

                    result.append({
                        "Text": sentence[0],
                        "Actual Sentiment": sentence[1],
                        "Local Model Prediction": local_result['sentiment'],
                        "Local Model Probabilities": local_result['probabilities'],
                        "Ollama Model Prediction": ollama_result['sentiment'],
                        "Ollama Model Probabilities": ollama_result['probabilities'],
                    })
                
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                result.append({
                    "Text": sentence[0],
                    "Actual Sentiment": sentence[1],
                    "Local Model Prediction": "Error",
                    "Local Model Probabilities": {},
                    "Ollama Model Prediction": "Error",
                    "Ollama Model Probabilities": {},
                })

            # display results
            st.subheader("Comparison Results")
            for res in result:
                st.markdown(f"**Text:** {res['Text']}")
                st.markdown(f"- Actual Sentiment: {res['Actual Sentiment']}")
                st.markdown(f"- Local Model Prediction: {res['Local Model Prediction']}")
                st.markdown(f"- Local Model Probabilities: {res['Local Model Probabilities']}")
                st.markdown(f"- Ollama Model Prediction: {res['Ollama Model Prediction']}")
                st.markdown(f"- Ollama Model Probabilities: {res['Ollama Model Probabilities']}")
                st.markdown("---")

def about_page():
    st.header("About This App")
    st.markdown("""
    This Sentiment Analysis Application allows users to analyze the sentiment of text inputs using both a local machine learning model and an Ollama model. 
    Users can also upload documents and chat with them, as well as compare the performance of the two models on a set of test sentences.

    **Technologies Used:**
    - Streamlit for the web interface
    - Scikit-learn for machine learning models
    - Ollama for advanced language modeling
    - Plotly for data visualization

    **Developed by:** Your Name
    """)
    st.markdown("© 2024 Your Name. All rights reserved.")



if __name__ == "__main__":
    main()
