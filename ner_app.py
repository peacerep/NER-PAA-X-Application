import streamlit as st
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from spacy.pipeline import EntityRuler
import pandas as pd
from spacy import displacy
import plotly.express as px

# Load the CSVs
df = pd.read_csv("alt_names_actor_table_111024.csv")
pax = pd.read_csv("pax_corpus_v8.csv")

# Load spaCy model
#nlp = spacy.load("en_core_web_sm")

# Initialize the EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")

df['alt_names'] = df['alt_names'].astype(str)

# Create patterns with names, alt names, and abbreviations
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

# Streamlit UI
st.set_page_config(layout="wide")  # Set to wide layout
# Custom HTML for logo and title
# Custom HTML for logo and title
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
    </div>
    """,
    unsafe_allow_html=True
)

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

# Display a message if there's no applicable text
if filtered_data.empty:
    st.warning(f"No applicable text found in the selected '{text_column}' column.")
else:
    st.success(f"Found {len(filtered_data)} agreements to run NER on.")

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

    # Add "Results" section title
    st.markdown("## Results")
    st.markdown("---")

    # Add a loading bar for progress
    progress_bar = st.progress(0)
    total_steps = len(filtered_data)  # Total number of agreements to process
    step_counter = 0

    # Store cleaned entities and metadata in a DataFrame
    results_df = pd.DataFrame(columns=['AgtId', 'entity', 'actor_name', 'actor_id', 'actor_type', 'conflict', 'date', 'pax_link'])

    # Iterate over the selected text
    res_list = []
    for _, row in filtered_data.iterrows():
        agt_id = row['AgtId']
        sentence_text = row[text_column]

        # Get the agreement information from the PAX dataframe
        agt_info = pax[pax['AgtId'] == agt_id]
        if agt_info.empty:
            st.warning(f"No agreement found for AgtId: {agt_id}. Skipping.")
            continue

        # Retrieve relevant agreement info
        agt_info = agt_info.iloc[0]
        conflict = agt_info['Con']  # Conflict
        date = agt_info['Dat']  # Date
        pax_link = agt_info['PAX_Hyperlink']  # PAX Hyperlink

        # Process the sentence with spaCy's NER pipeline
        sentence_doc = nlp(sentence_text)

        # If visualization is selected, render the Displacy visualization and agreement details
        if visualize_ner:
            # Print agreement information including the PAX hyperlink
            agreement_info = f"**Agreement ID:** {agt_id} (Conflict: {conflict}, Date: {date}, PAX Hyperlink: {pax_link})"
            st.markdown(agreement_info, unsafe_allow_html=True)  # Use markdown to style

            # Create custom colors for selected entities
            options = {
                "ents": selected_entity_types,
                "colors": {label: color_scheme.get(label, "lightgray") for label in selected_entity_types}
            }
            # Render the Displacy visualization
            html = displacy.render(sentence_doc, style="ent", options=options)
            st.write(html, unsafe_allow_html=True)

        # Store recognized entities and their metadata
        for ent in sentence_doc.ents:
            if ent._.metadata and ent.text.strip():
                res_dict = {
                    'AgtId': agt_id,
                    'entity': ent.text.strip(),
                    'actor_name': ent._.metadata['actor_name'],
                    'actor_id': ent._.metadata['id_paax'],
                    'actor_type': ent._.metadata['actor_type'],
                    'conflict': conflict,
                    'date': date,
                    'pax_link': pax_link
                }
                res_list.append(res_dict)

        # Update the progress bar after processing each agreement
        step_counter += 1
        progress_bar.progress(step_counter / total_steps)

        # If visualizing, print a horizontal line between agreements
        if visualize_ner:
            st.write("---")  # Add a horizontal line for separation

    # Convert results to a DataFrame
    results_df = pd.DataFrame(res_list)

    # Display the results in the app before any visualizations
    st.write("NER Results:")
    st.dataframe(results_df)

    # Visualizations using Plotly
    if not results_df.empty:
        # Bar chart of the number of agreements by actor_name
        actor_name_count = results_df['actor_name'].value_counts().reset_index()
        actor_name_count.columns = ['Actor Name', 'Count']
        fig_bar = px.bar(actor_name_count, x='Actor Name', y='Count', title='Number of Agreements by Actor Name')
        st.plotly_chart(fig_bar)

        # Pie chart of the main actor types recognized
        actor_type_count = results_df['actor_type'].value_counts().reset_index()
        actor_type_count.columns = ['Actor Type', 'Count']
        
        # Map actor types to their colors from the Displacy color scheme
        actor_type_count['color'] = actor_type_count['Actor Type'].map(color_scheme)

        # Create the pie chart with matching colors
        fig_pie = px.pie(
            actor_type_count, 
            names='Actor Type', 
            values='Count', 
            title='Distribution of Actor Types Recognized',
            color='Actor Type',
            color_discrete_map=color_scheme  # Match colors to Displacy
        )
        st.plotly_chart(fig_pie)
