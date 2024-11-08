import streamlit as st
import PyPDF2  # For PDF text extraction
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
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

# Define a color scheme for actor types
color_scheme = {
    'Intergovernmental Organization': '#2980b9',
    'Armed Organization': '#d62728',
    'Civil Society': '#bcbd22',
    'Political Party': '#9467bd',
    'State Coalition': '#03a9f4',
    'Umbrella': '#e377c2',
    'Country/State': '#198038',
    'Entity': '#ff7f0e',
    'Military': '#8c564b',
    'Other': '#f1c40f'
}

# Define PDF text extraction function
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    # Use `len(reader.pages)` to get the total number of pages
    for page in reader.pages:
        # `extract_text` method extracts text from each page
        text += page.extract_text()
    return text

# Streamlit layout with three columns
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        
        body {
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  /* Set core color for text */
        }
        
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;  /* Distribute space between items */
            margin-bottom: 20px;
        }
        .header-container img {
            width: 200px;  /* Adjust height as needed */
            margin: 0 20px;  /* Space around logos */
        }
        .header-title {
            text-align: center;
            flex-grow: 1;  /* Allow title to grow and take up space between logos */
            font-size: 3em;  /* Adjust title size */
            margin: 0;  /* Remove default margin */
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  /* Set core color for text */
        }
        .sub-title {
            text-align: center;  /* Center the subtitle */
            font-size: 1.5em;  /* Adjust subtitle size */
            margin-top: 10px;  /* Space above subtitle */
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  /* Set core color for text */
        }
    </style>
    <div class="header-container">
        <img src="https://peacerep.github.io/logos/img/PeaceRep_nobg.png" alt="PeaceRep Logo" />
        <h1 class="header-title">Named Entity Recognition: PA-X Peace Agreements Database</h1>
          <!-- Replace with your second logo URL -->
        <img src="https://peacerep.github.io/logos/img/Pax_nobg.png" alt="Logo" />
    </div>
    <div class="sub-title">
        <p><b>Credits: Sanja Badanjak and Niamh Henry (2024), the Peace Agreement Actor Dataset (PAA-X). PeaceRep, University of Edinburgh</b></p>
        <p>This experimental tool allows you to run spaCy's Named Entity Recognition (NER) that includes rule-based approaches from the Peace Agreement Actor Dataset (PAA-X) on text data that denotes the party and third party signatories to agreements, and the full peace agreement text. For faster processing time, filter by any PA-X metadata by selecting a column, and the values you want to keep. Then click 'Execte NER' to run the model. Select the checkbox to visualise the NER results within the text. Results will be shown after these visualisations in tablular format, that can be exported as a csv file.</p>
        <p><b>DISCLAIMER:</b> This is an experimental tool, and will not return 100% accurate results, due to the nature of different naming conventions. A mention of an actor in party or third party fields, does not equate to being a signatory. Use the full PAA-X dataset for accurate data on peace agreement signatories. <b>Ensure manual corrections of recognised instances</b>
    </div>
    """,
    unsafe_allow_html=True
)

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
visualize_ner = st.checkbox("Visualize results with displacy", value=True)

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
            html = displacy.render(doc, style="ent", options={"ents": selected_entity_types, "colors": color_scheme})
            st.write(html, unsafe_allow_html=True)

        # Display NER results
        st.write("NER Results:")
        for ent in doc.ents:
            st.write(f"Entity: {ent.text}, Label: {ent.label_}")
    else:
        st.warning("No text provided for NER.")
