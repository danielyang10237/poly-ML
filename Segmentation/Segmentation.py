import spacy
from queue import Queue
import re
import string

# spacy.require_gpu(0)

nlp = spacy.load("en_core_web_md")

FANBOYS = ["for", "and", "nor", "but", "or", "yet", "so"]
FANBOYS = [", " + x for x in FANBOYS] + ["," + x for x in FANBOYS]

class WordNode:
    def __init__(self, word):
        self.text = word.text
        self.pos = word.pos_
        self.idx = word.i
        self.lbl = word.dep_
        self.children = []
        self.parent = None
        self.obj_root = False
    
    def add_child(self, child):
        self.children.append(child)
    
    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def add_parent(self, parent):
        self.parent = parent
    
    def print(self):
        print("Node summary:", self.text, [x.text for x in self.children], self.obj_root)

    def is_children(self, word):
        return word in self.children
    
    def set_object(self):
        self.obj_root = True

    def unset_object(self):
        self.obj_root = False

# checks if word is punctuation
def is_punctuation(word):
    punct = [".", ",", "!", "?", ":", ";", "'"]
    for p in punct:
        if p in word or word == ',':
            return True
    return False

# collects all the word indices based off our constructed tree
def dfs_traverse(root_node):
    seen_nodes = [root_node.idx]
    obj_nodes = []
    for child in root_node.children:
        if child.obj_root == True:
            obj_nodes.append(child)
            continue
        more_nodes, more_obj_nodes = dfs_traverse(child)
        seen_nodes.extend(more_nodes)
        obj_nodes.extend(more_obj_nodes)
    return seen_nodes, obj_nodes

# gets rid of all the attached coordinating conjunctions
def strip_conjunctions(root_node):
    for child in root_node.children:
        if child.lbl == "cc":
            root_node.remove_child(child)
            for grandchild in child.children:
                root_node.add_child(grandchild)
        strip_conjunctions(child)

def get_subject(input_sentence, orig_mapping = False):
    doc = nlp(input_sentence)

    word_nodes = []

    # parsing the sentence from spacy
    for token in doc:
        new_node = WordNode(token)
        word_nodes.append(new_node)

        if orig_mapping:
            ancestors = [t.text for t in token.ancestors]
            children = [t.text for t in token.children]
            print(token.text, "\t", token.i, "\t", 
                token.pos_, "\t", token.dep_, "\t", 
                ancestors, "\t", children)
    
    # construct our own model of the tree
    for token in doc:
        for child in token.children:
            word_nodes[token.i].add_child(word_nodes[child.i])
    
    # find the subject of the sentence
    def get_subject():
        for word in doc:
            if (word.dep_ == "nsubj"):
                return word
        
        return None

    subject = get_subject()

    if not subject:
        return None, doc, word_nodes, None
    
    # find the root verb of the sentence
    def find_root_of_sentence(doc):
        root_token = None
        for token in doc:
            if (token.dep_ == "ROOT" and token.pos_ == "VERB"):
                root_token = token
        return root_token
    
    root_token = find_root_of_sentence(doc)

    if not root_token:
        return None, doc, word_nodes, root_token
    
    subject = word_nodes[subject.i]
    root_token = word_nodes[root_token.i]

    # route the root verb to the subject
    root_token.add_parent(subject)
    root_token.remove_child(subject)
    subject.remove_child(root_token)

    return subject, doc, word_nodes, root_token

# give each verb its own sentence
def segment_verbs(subject, word_nodes, root_token):
    verb_sentences = []
    verbs = [root_token]

    # find any other verbs in the sentence
    def parse_other_verbs(root_token):
        other_verbs = []
        for children in root_token.children:
            if (children.pos == "VERB" and children != root_token):
                other_verbs.append(children)
                root_token.remove_child(children)
                children.add_parent(subject)
                other_verbs.extend(parse_other_verbs(children))
                strip_conjunctions(root_token)
            if (children.lbl == "nsubj"):
                root_token.remove_child(children)
                root_token.add_parent(children)
    
        return other_verbs

    verbs.extend(parse_other_verbs(root_token))

    # now we find all dependent verbs (verbs without an object attached)
    dependent_verbs = Queue()
    first_object = None
    for verb in verbs:
        dependent = True
        for child in verb.children:
            if (child.lbl == 'dobj' or child.pos == "ADP" or child.pos == "NOUN" or child.pos == "PRON"):
                if first_object is None:
                    first_object = child
                dependent = False
        if dependent:
            dependent_verbs.put(verb)

    if not first_object:
        return []

    # attach objects to all dependent verbs 
    while not dependent_verbs.empty():
        dependent_verb = dependent_verbs.get()
        dependent_verb.add_child(first_object)


    # print("Segmented verbs")

    # for each verb, we create its own clause
    for action in verbs:
        subjects, _ = dfs_traverse(action.parent)
        subjects.extend(dfs_traverse(action)[0])

        total_sentence = sorted(subjects)
        total_sentence_text = ""
        for idx in total_sentence:
            if total_sentence_text != "" and not is_punctuation(word_nodes[idx].text):
                total_sentence_text += " "
            total_sentence_text += word_nodes[idx].text
        verb_sentences.append(total_sentence_text)

        # print(total_sentence_text)
    
    return verb_sentences

def segment_objects(verb_sentences, custom_mapping = False):
    clauses = []

    for sentence in verb_sentences:
        _, _, word_nodes, root_token = get_subject(sentence)

        if root_token == None:
            # print("could not find root token")
            continue

        visited = set()

        # checks for layered objects
        def check_prepositional_phrases(parent):
            count = 0
            for child in parent.children:
                if child.lbl == "conj" or child.lbl == "dobj" or child.lbl == "pobj":
                    count += 1
                # if child.lbl == "prep":
                #     return True
                # if child.pos == "NOUN" or child.lbl == "dobj":
                #     for grandchild in child.children:
                #         if (grandchild.lbl == "conj" or grandchild.pos == "NOUN" or grandchild.lbl == "dobj") and grandchild.lbl != "compound":
                #             return True
            return count <= 1

        # map conjoining objects to their own branch
        def dfs_down_rewire(grandparent, parent):
            if parent.idx in visited:
                return
            visited.add(parent.idx)

            for child in parent.children:
                next_grandparent, next_parent = parent, child

                # print("CHILD", child.text, child.pos, child.lbl)
                # print("PARENT", parent.text, parent.pos, parent.lbl)

                if (child.lbl == "conj" or child.lbl == "dobj" or child.lbl == "pobj") and child.lbl != "compound":
                    if (parent.lbl == "conj" or parent.lbl == "dobj" or parent.lbl == "pobj") and child.lbl != "compound":
                        grandparent.add_child(child)
                        parent.remove_child(child)
                        child.set_object()
                        parent.set_object()
                        strip_conjunctions(parent)

                        next_grandparent = grandparent
                
                dfs_down_rewire(next_grandparent, next_parent)
        
        if check_prepositional_phrases(root_token):
            for child in root_token.children:
                dfs_down_rewire(root_token, child)
        else:
            for child in root_token.children:
                if (child.lbl == "conj" or child.pos == "NOUN") and child.lbl != "compound":
                    child.set_object()
                    strip_conjunctions(root_token)

        total_sentence_idx = dfs_traverse(root_token.parent)[0]
        sentence_idxs, objects = dfs_traverse(root_token)
        total_sentence_idx.extend(sentence_idxs)

        # print("segmented objects")

        # form all the sentences with unique objects
        if len(objects) == 0:
            total_sentence_idx = sorted(total_sentence_idx)
            total_sentence_text = ""
            for idx in total_sentence_idx:
                total_sentence_text += word_nodes[idx].text + " "
            clauses.append(total_sentence_text)

            # print(total_sentence_text)
        else:
            for obj in objects:
                total_sentence = total_sentence_idx + dfs_traverse(obj)[0]
                total_sentence.sort()
                total_sentence_text = ""
                for idx in total_sentence:
                    if total_sentence_text != "" and not is_punctuation(word_nodes[idx].text):
                        total_sentence_text += " "
                    total_sentence_text += word_nodes[idx].text
                clauses.append(total_sentence_text)

                # print(total_sentence_text)
        
        if custom_mapping:
            for node in word_nodes:
                node.print()
    
    return clauses


def segment_verbs_and_objects(input_string, orig_mapping = False, custom_mapping = False):

    subject, doc, word_nodes, root_token = get_subject(input_string, orig_mapping)
    
    if subject == None:
        return [input_string]
    
    # split the sentences with each sentence having one verb from the subject
    verb_segmented_sentences = segment_verbs(subject, word_nodes, root_token)
    if len(verb_segmented_sentences) == 0:
        return [input_string]
    
    # split the sentences according to unique objects
    clauses = segment_objects(verb_segmented_sentences, custom_mapping)
    
    return clauses

def segment_clauses(input_string, orig_mapping = False, custom_mapping = False):
    # pattern = '|'.join(map(re.escape, FANBOYS))
    
    # clauses = re.split(pattern, input_string)
    
    # for clause in clauses:
    segmented = segment_verbs_and_objects(input_string, orig_mapping, custom_mapping)

    translator = str.maketrans('', '', string.punctuation)

    all_words = (segment.split(" ") for segment in segmented)
    seen = set()

    for words in all_words:
        for word in words:
            cleaned_word = word.translate(translator)
            cleaned_word = word.replace("'s", "")
            seen.add(cleaned_word)
    
    doc = nlp(input_string)
    for token in doc:
        if (token.pos_ == "NOUN" or token.pos_ == "PRON" or token.pos_ == "VERB") and token.text not in seen:
            # print(token.text)
            return [input_string]
    return segmented

import re

def segment_sentences(input_string, orig_mapping=False, custom_mapping=False):
    mini_sentences = []

    # Split the input string by sentence-ending punctuation and newline characters
    parts = re.split(r'[.!?]+\s*|\n+', input_string)

    for part in parts:
        # Remove special and non-English characters, leaving only English letters, numbers, punctuation, and symbols
        cleaned_part = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-]', '', part).strip()

        if cleaned_part:
            mini_sentences.append(cleaned_part)  # Use append instead of extend

    return mini_sentences

        # Only process and add non-empty strings
        # if cleaned_part:
        #     mini_sentences.extend(segment_clauses(cleaned_part, orig_mapping, custom_mapping))

    return mini_sentences

def segment(input_text):
    test_input = input_text
    print("recieved", test_input)
    output = []
    for parsed_sentence in segment_sentences(test_input, False, False):
        output.append(parsed_sentence)
    return output