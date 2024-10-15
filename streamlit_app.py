import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from file_handler import handle_file_upload
from predictor import predict, load_classification_model, load_ner_model, load_clause_model
import torch

# Initialize session state to store extracted information
if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = {}

# Load Legal-BERT models
classification_tokenizer, classification_model = load_classification_model()
ner_tokenizer, ner_model = load_ner_model()
clause_tokenizer, clause_model = load_clause_model()

# Load LLaMA 3.1 Model and Tokenizer for answering questions
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model_llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Function to extract legal information and format into understandable sentences
def extract_legal_information(text):
    # Classification (Relevance)
    relevance_score = predict((classification_tokenizer, classification_model), text)
    relevance_sentence = f"The document's relevance score is {round(relevance_score, 2)}%."

    # Named Entity Recognition (NER)
    entities = predict((ner_tokenizer, ner_model), text)
    
    # Ensure that `entities` is iterable and contains valid data
    if isinstance(entities, list) and len(entities) > 0 and isinstance(entities[0], (list, tuple)):
        entity_sentence = f"The document mentions the following key entities: {', '.join([entity[0] for entity in entities if len(entity) > 0])}."
    else:
        entity_sentence = "No key legal entities were identified in the document."

    # Clause Identification with mapping to human-readable clause names
    clause_type_id = predict((clause_tokenizer, clause_model), text)

    # Check if the clause_type_id is a float and round it to the nearest integer
    if isinstance(clause_type_id, float):
        clause_type_id = round(clause_type_id)

    # Define a mapping for legal clause types based on LegalPro-BERT capabilities
    clause_type_mapping = {
        1: "Confidentiality Clause",
        2: "Termination Clause",
        3: "Non-Compete Clause",
        4: "Payment Terms",
        5: "Indemnity Clause",
        6: "Governing Law Clause",
        7: "Dispute Resolution Clause",
        # Add additional clause types here as necessary
    }
    
    # Get the human-readable clause type or show the ID if unknown
    clause_type = clause_type_mapping.get(clause_type_id, f"Unknown clause type: {clause_type_id}")
    clause_sentence = f"The document contains the following clause type: {clause_type}."

    # Combine the sentences into a list to display later
    return [relevance_sentence, entity_sentence, clause_sentence]

# Function to pass extracted legal information and document content to LLaMA 3.1 for question answering
def ask_legal_question(question, document_content, legal_info):
    combined_text = f"Document Content:\n\n{document_content}\n\nExtracted Legal Information:\n\n" + "\n".join(legal_info)
    prompt = f"### Instruction: Answer the following legal question based on the provided content:\n### Question: {question}\n### Content: {combined_text}\n### Response:"
    inputs = tokenizer_llama(prompt, return_tensors="pt")
    outputs = model_llama.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer_llama.eos_token_id
    )
    return tokenizer_llama.decode(outputs[0], skip_special_tokens=True)

# Streamlit Interface
st.title("LawyerAI ⚖️ - Legal Document Extraction & Summarization")

# Tabs for switching between Extraction and Document Creation
tab1, tab2, tab3 = st.tabs(["Legal Information Extraction", "Document Creation", "Ask Legal Questions"])

# --- Legal Information Extraction Tab ---
with tab1:
    st.header("Legal Document Classifier ⚖️")
    st.write("Upload a legal document (PDF or DOCX), and the system will extract key information using Legal-BERT.")
    
    uploaded_file = st.file_uploader("Upload a legal document (PDF/DOCX):", type=["pdf", "docx"])
    
    document_content, document_images = handle_file_upload(uploaded_file)
    
    if uploaded_file is not None:
        # Display document preview for PDFs or show extracted text for DOCX
        if document_images:
            st.write("### Document Preview (PDF):")
            for img in document_images:
                st.image(img, caption="Page Preview", use_column_width=True)
        else:
            st.write("### Extracted Document Text (DOCX):")
            st.write(document_content[:2000] + "...")  # Show a preview of the document content

        # Extract key legal information
        legal_info_sentences = extract_legal_information(document_content)

        # Display extracted information
        st.write("### Extracted Legal Information:")
        for sentence in legal_info_sentences:
            st.write(sentence)

        # Store the document content and legal information for later use
        st.session_state.extracted_info = (document_content, legal_info_sentences)

# --- Ask Legal Questions Tab ---
with tab3:
    st.header("Ask Legal Questions About the Document 📜")
    st.write("You can ask the model specific legal questions about the document content, and it will generate an answer based on the document context.")

    question = st.text_input("Enter your legal question:")
    if st.button("Get Answer"):
        if "extracted_info" in st.session_state and st.session_state.extracted_info:
            document_content, legal_info = st.session_state.extracted_info
            answer = ask_legal_question(question, document_content, legal_info)
            st.write(f"**Answer**: {answer}")
        else:
            st.warning("Please upload a document to ask questions.")
