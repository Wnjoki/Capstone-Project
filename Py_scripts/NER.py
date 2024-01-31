
def b(text_series):
    ner_pipeline = pipeline("ner",model=model,tokenizer=tokenizer, aggregation_strategy="simple")

ner_results = ner_pipeline(dataset)
print(ner_results[:10])


# Download the 'en_core_web_sm' model
spacy.cli.download('en_core_web_sm')
                   
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define a function to perform NER on a text
def perform_ner(text_series):
    named_entities = []
    for text in text_series:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        named_entities.extend(entities)
    return named_entities

# Apply NER to a  dataset
named_entities = perform_ner(dataset)
