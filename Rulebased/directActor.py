"""
Applies a rule-based approach for labeling unambiguous animated entities for the training data.
It uses an actor noun list and an inanimate adjective list as patterns.
NER is used to identify organizations and people, and a name database is used for labeling names. 
"""

import os
import spacy
from spacy.matcher import Matcher
from names_dataset import NameDataset

nlp = spacy.load("de_core_news_lg")
matcher = Matcher(nlp.vocab)
name_dataset = NameDataset()

with open("gold_actor_10_0.7.txt", "r", encoding="utf-8") as f:
    for line in f:
        words = line.strip().lower().split()
        for word in words:
            if word:  
                pattern = [{"LOWER": word}]
                matcher.add(word, [pattern])

inanimated_adjectives = set()
with open("inanimated_adjectives_5_0.9.txt", "r", encoding="utf-8") as f:
    for line in f:
        adjective = line.strip().lower()
        inanimated_adjectives.add(adjective)

def should_animate_chunk(chunk_tokens):
    doc = nlp(" ".join(chunk_tokens))
    matches = matcher(doc)
    contains_inanimated_adjective = any(token.lower_ in inanimated_adjectives for token in doc if token.pos_ == "ADJ")
    contains_person_entity = any(ent.label_ == "PERSON" for ent in doc.ents)
    if matches and not contains_inanimated_adjective:
        for match_id, start, end in matches:
            span = doc[start:end]
            if span.root.pos_ == "NOUN":
                return True
    if contains_person_entity:
        return True
    if any(token.lower() in ['wir', 'ich', 'man', 'jemand'] for token in chunk_tokens):
        return True
    if any(ent.label_ == "ORG" for ent in doc.ents):
        return True
    first_name_count = 0
    last_name_count = 0
    for token in doc:
        token_lower = token.text.lower()
        name_info = name_dataset.search(token_lower)
        if 'first_name' in name_info and token.pos_ == "PROPN":
            first_name_count += 1
        if 'last_name' in name_info and token.pos_ == "PROPN":
            last_name_count += 1
    if first_name_count >= 2 and last_name_count >= 2:
        return True
    return False

def process_chunk(chunk):
    chunk_tokens = [token_tuple[2] for token_tuple in chunk]
    is_animated = should_animate_chunk(chunk_tokens)
    if is_animated:
        for i in range(len(chunk)):
            try:
                chunk[i][3] = 'Animated' if chunk[i][4] in ['B', 'I'] else '_'
            except IndexError:
                continue
    else:
        for i in range(len(chunk)):
            try:
                chunk[i][3] = '_'
            except IndexError:
                continue
    return ['\t'.join(token_tuple) for token_tuple in chunk]


input_folder = 'C:/Users/fredh/Documents/BA/Labeling/Chunks'
output_folder = 'C:/Users/fredh/Documents/BA/Labeling/Animated'

file_list = os.listdir(input_folder)

for filename in file_list:
    if os.path.isfile(os.path.join(input_folder, filename)):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '_animated.csv')
        
        output_lines = []
        
        with open(input_file, 'r', encoding="utf-8") as file:
            chunk = []
            inside_text_segment = False
            line_count = 0
            for line in file:
                line = line.strip()
                line_count += 1
                if line_count <= 5:
                    output_lines.append(line)
                    continue
                if line.startswith("#Text="):
                    if inside_text_segment and chunk:
                        output_lines.extend(process_chunk(chunk))
                        chunk = []
                    inside_text_segment = True
                    output_lines.append(line)
                    continue
                if not line:
                    inside_text_segment = False
                    if chunk:
                        output_lines.extend(process_chunk(chunk))
                        chunk = []
                    output_lines.append(line)
                    continue
                if inside_text_segment:
                    parts = line.split('\t')
                    chunk_label = parts[4] if len(parts) > 4 else "_"
                    if chunk_label == 'B' and chunk:
                        output_lines.extend(process_chunk(chunk))
                        chunk = []
                    chunk.append(parts)
            if chunk:
                output_lines.extend(process_chunk(chunk))
        with open(output_file, 'w', encoding="utf-8") as out_file:
            for line in output_lines:
                out_file.write(line + '\n')