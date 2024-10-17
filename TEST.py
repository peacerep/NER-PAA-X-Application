import streamlit as st
import PyPDF2  # For PDF text extraction
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from pathlib import Path

# Load the CSVs
df = pd.read_csv("alt_names_actor_table_111024.csv")
pax = pd.read_csv("pax_corpus_v8.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

# Define PDF text extraction function
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(reader.getNumPages()):
        text += reader.getPage(page).extractText()
    return text

# Streamlit layout with three columns
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1, 1, 1])

# Left Column (PA-X filtering)
with col1:
    st.header("PA-X Dataset")
    # Step 1: Allow the user to select multiple filter columns to narrow down the agreements
    filter_columns = st.multiselect("Select columns to filter agreements:", pax.columns.tolist())

    # Initialize filtered_pax to the full pax dataframe
    filtered_pax = pax

    # Step 2: For each selected filter column, allow the user to select values to include
    for filter_column in filter_columns:
        filter_values = st.multiselect(
            f"Select values from '{filter_column}' to include in the analysis:",
            options=pax[filter_column].unique(),
            key=f"value_selection_{filter_column}"  # Unique key for each multiselect
        )
        # Apply filtering based on user selection
        if filter_values:
            filtered_pax = filtered_pax[filtered_pax[filter_column].isin(filter_values)]

    # Show how many rows match the filter
    st.write(f"Found {len(filtered_pax)} agreements matching your filter.")

    # Step 3: Let the user select which text column to use ('Part' or 'ThrdPart')
    text_column = st.radio("Select the text column to run NER on:", ('Part', 'ThrdPart', 'Agreement text'))

    # Filter the PAX dataframe based on the selected column
    filtered_data = filtered_pax[['AgtId', text_column]].dropna(subset=[text_column])

# Middle Column (Upload PDF or CSV)
with col2:
    st.header("Upload Your Own File")
    # Option to upload either a PDF or CSV
    uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=['pdf', 'csv'])
    
    # Store the extracted text
    custom_text = ""

    # Handle PDF or CSV upload
    if uploaded_file:
        if uploaded_file.name.endswith('.pdf'):
            custom_text = extract_text_from_pdf(uploaded_file)
            st.write("PDF file uploaded and text extracted successfully.")
        elif uploaded_file.name.endswith('.csv'):
            csv_data = pd.read_csv(uploaded_file)
            st.write(csv_data.head())  # Display the first few rows of the CSV file
            # Assuming CSV has a text column, allow the user to select the column to run NER on
            text_column = st.selectbox("Select text column in CSV:", csv_data.columns)
            custom_text = " ".join(csv_data[text_column].dropna().tolist())

# Right Column (Manual Text Entry)
with col3:
    st.header("Enter Text Manually")
    manual_text = st.text_area("Enter or paste your text here:")

# Combine input sources into a single variable
if custom_text:
    ner_input_text = custom_text
elif manual_text:
    ner_input_text = manual_text
elif not filtered_data.empty:
    ner_input_text = " ".join(filtered_data[text_column].dropna().tolist())
else:
    ner_input_text = ""

# Step 4: Checkbox for visualization option before executing NER
visualize_ner = st.checkbox("Visualize results with displacy", value=False)

# Step 5: Select which entity types to visualize
entity_types = df['actor_type'].unique()
selected_entity_types = st.multiselect(
    "Select entity types to visualize:",
    options=entity_types,
    default=entity_types.tolist()  # Default to all selected
)

# Step 6: Button to execute the NER
if st.button("Execute NER"):
    if ner_input_text:
        # Process the text with spaCy
        doc = nlp(ner_input_text)

        # If visualization is selected, render the displacy visualization
        if visualize_ner:
            st.write("Visualizing NER...")
            html = displacy.render(doc, style="ent", options={"ents": selected_entity_types})
            st.write(html, unsafe_allow_html=True)

        # Display NER results
        st.write("NER Results:")
        for ent in doc.ents:
            st.write(f"Entity: {ent.text}, Label: {ent.label_}")
    else:
        st.warning("No text provided for NER.")
