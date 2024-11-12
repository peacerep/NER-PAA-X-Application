import streamlit as st
import PyPDF2  # For PDF text extraction
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
from pathlib import Path

# Load the CSVs
df = pd.read_csv("alt_names_actor_table_081124.csv")
pax = pd.read_csv("pax_corpus_v8.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1000000  # Adjust as needed for safety with large text segments

# Initialize the EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Convert alt_names to strings for processing
df['alt_names'] = df['alt_names'].astype(str)

# Create patterns for the EntityRuler
patterns = []
for _, row in df.iterrows():
    all_names = [row['actor_name']] + row['alt_names'].split('|') + [row['abbreviation']]
    for name in all_names:
        patterns.append({
            "label": row['actor_type'],
            "pattern": name,
            "id_paax": row['id_paax'],
            "metadata": {
                "actor_name": row['actor_name'],
                "id_paax": row['id_paax'],
                "actor_type": row['actor_type']
            }
        })

# Add patterns to the EntityRuler
ruler.add_patterns(patterns)

# Add a custom extension to store metadata in the entity
spacy.tokens.Span.set_extension('metadata', default=None, force=True)

# Define a function to assign metadata to the matched entities
@spacy.Language.component("assign_metadata")
def assign_metadata(doc):
    for ent in doc.ents:
        matched_metadata = next((pat['metadata'] for pat in patterns if pat['pattern'] == ent.text), None)
        if matched_metadata:
            ent._.metadata = matched_metadata
    return doc

# Add the metadata assignment function to the pipeline
nlp.add_pipe("assign_metadata", last=True)

# Define PDF text extraction function
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Text segmentation function
def split_text_into_chunks(text, max_length):
    doc = nlp(text)  # Process to get sentences
    sentences = [sent.text for sent in doc.sents]
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Streamlit layout with three columns and grey separators
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        
        body {
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  
        }
        
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;  
            margin-bottom: 20px;
        }
        .header-container img {
            width: 200px;  
            margin: 0 20px;  
        }
        .header-title {
            text-align: center;
            flex-grow: 1;  
            font-size: 3em;  
            margin: 0;  
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  
        }
        .sub-title {
            text-align: center;  
            font-size: 1.5em;  
            margin-top: 10px;  
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  
        }
        
        .column-container {
            display: flex;
            justify-content: space-between;
        }

        .vertical-line {
            border-left: 1px solid grey;
            height: 300px;
            position: absolute;
            left: 33.33%;
            margin-top: 20px;  /* Adjust the top margin to fit the header */
        }
        
        .vertical-line-2 {
            border-left: 1px solid grey;
            height: 300px;
            position: absolute;
            left: 66.66%;
            margin-top: 20px;  /* Adjust the top margin to fit the header */
        }

        .checkbox-container {
            margin-bottom: 10px; /* Space between checkboxes */
        }
    </style>
    <div class="header-container">
        <img src="https://peacerep.github.io/logos/img/PeaceRep_nobg.png" alt="PeaceRep Logo" />
        <h1 class="header-title">Named Entity Recognition: PA-X Peace Agreements Database</h1>
        <img src="https://peacerep.github.io/logos/img/Pax_nobg.png" alt="Logo" />
    </div>
    <div class="sub-title">
        <p><b>Credits: Sanja Badanjak and Niamh Henry (2024), the Peace Agreement Actor Dataset (PAA-X). PeaceRep, University of Edinburgh</b></p>
        <p>This experimental tool allows you to run spaCy's Named Entity Recognition (NER) that includes rule-based approaches from the Peace Agreement Actor Dataset (PAA-X).</p> <p> Find actors who have signed peace agreements in a range of textual fields in PA-X (party, third party or agreements), from your own custom PDFs or csv files or simply paste text into the text box. Then click 'Execte NER' to run the model. Select the checkbox to visualise the NER results within the text. Results will be shown after these overview in tablular format, that can be exported as a csv file.</p>
        <p><b>DISCLAIMER:</b> This is an experimental tool, and will not return 100% accurate results, due to the nature of different naming conventions. <b>Ensure manual corrections of recognised instances before using to inform work.</b>
    </div>
    """, 
    unsafe_allow_html=True
)

# Layout with columns
col1, col2, col3 = st.columns([1, 1, 1])

# Define checkboxes to select the input source
use_pax = st.checkbox("Use PA-X Dataset", key="use_pax")
use_uploaded_file = st.checkbox("Use Uploaded File", key="use_uploaded_file")
use_manual_text = st.checkbox("Use Manual Text Entry", key="use_manual_text")

# Text input options for each source
ner_input_text = ""
if use_pax:
    st.header("PA-X Dataset")
    filter_columns = st.multiselect("Select columns to filter agreements:", pax.columns)
    filtered_pax = pax
    for filter_column in filter_columns:
        filter_values = st.multiselect(f"Select values from '{filter_column}' to include:", pax[filter_column].unique())
        if filter_values:
            filtered_pax = filtered_pax[filtered_pax[filter_column].isin(filter_values)]
    st.write(f"Found {len(filtered_pax)} agreements.")
    text_column = st.radio("Select text column to run NER on:", ['Part', 'ThrdPart', 'Agreement text'])
    ner_input_text = " ".join(filtered_pax[text_column].dropna().tolist())

if use_uploaded_file:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Upload a PDF or CSV", type=['pdf', 'csv'])
    if uploaded_file:
        if uploaded_file.name.endswith('.pdf'):
            ner_input_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            csv_data = pd.read_csv(uploaded_file)
            text_column = st.selectbox("Select text column in CSV:", csv_data.columns)
            ner_input_text = " ".join(csv_data[text_column].dropna().tolist())

if use_manual_text:
    st.header("Manual Text Entry")
    manual_text = st.text_area("Enter text manually:")
    if manual_text:
        ner_input_text = manual_text

# Select custom entity types with additional default SpaCy entity types
default_entities = df['actor_type'].unique().tolist()
spacy_entities = ['PERSON', 'GPE', 'ORG', 'NORP', 'LOC']
selected_entity_types = st.multiselect("Select entity types:", options=default_entities + spacy_entities, default=default_entities)

# Run NER
if st.button("Execute NER"):
    if ner_input_text:
        chunks = split_text_into_chunks(ner_input_text, nlp.max_length // 2)
        entities = []
        
        for chunk in chunks:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in selected_entity_types:
                    entities.append({
                        "Entity": ent.text,
                        "Label": ent.label_,
                        **(ent._.metadata or {})  # Include metadata if available
                    })
        
        # Display entities in DataFrame
        if entities:
            df_results = pd.DataFrame(entities)
            st.write(df_results)
            st.download_button("Download NER Results as CSV", df_results.to_csv(index=False), "ner_results.csv")
            
            # Visualize with Displacy
            if st.checkbox("Visualize results"):
                html = displacy.render(doc, style="ent", options={"ents": selected_entity_types})
                st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("No text provided for NER.")
