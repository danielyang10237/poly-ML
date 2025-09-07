import spacy

# Load the transformer-based English model
nlp = spacy.load("en_core_web_trf")

def find_root_of_sentence(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None

def get_clause(token, subject):
    clause = []
    for child in token.children:
        if child.dep_ not in {"cc", "punct"}:  # Avoid conjunctions and punctuation
            clause.append(child)
    clause.append(token)
    clause.sort(key=lambda x: x.i)  # Sort by token index
    clause_text = " ".join([subject.text if t.dep_ == "nsubj" else t.text for t in clause])
    return clause_text

def get_subjects(doc):
    subjects = []
    for token in doc:
        if token.dep_ == "nsubj":
            subjects.append(token)
    return subjects

def split_into_clauses(doc):
    root_token = find_root_of_sentence(doc)
    if not root_token:
        return []

    subjects = get_subjects(doc)
    clauses = []

    for subject in subjects:
        for token in doc:
            if token.dep_ in {"ROOT", "conj"} and token.pos_ in {"VERB", "AUX"}:
                clause = get_clause(token, subject)
                clauses.append(clause)
            elif token.dep_ in {"dobj", "pobj"}:
                clause = get_clause(token.head, subject)
                clauses.append(clause)

    return clauses

# Example usage
input_string = "John and Steven love the beach and the mountains, but they don't go to the city often."
doc = nlp(input_string)

clauses = split_into_clauses(doc)
for clause in clauses:
    print(clause)
