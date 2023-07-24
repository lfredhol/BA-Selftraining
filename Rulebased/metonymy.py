"""
Applies a rule-based approach to animated metonymy labeling to the training data. 
It uses a metonymy noun list, an animated adjective list, and a verb list. 
NER and dependency parsing are used to apply the syntactic rules to detect metonymic expressions that represent animated entities.
"""

import csv
import spacy
import os
from spacy.matcher import Matcher

def analyze_text(current_text, tokens, nlp, metonyms, animated_adjectives, verbs):
    output_lines = []
    doc = nlp(current_text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "wir"}, {"LOWER": "als"}]
    matcher.add("WirAls", [pattern])
    pattern_gesicht = [{"LOWER": "das"}, {"LOWER": "gesicht"}, {"LOWER": "deutschlands"}]
    pattern_stimme = [{"LOWER": "die"}, {"LOWER": "stimme"}, {"LOWER": "deutschlands"}]
    matcher.add("GesichtDeutschlands", [pattern_gesicht])
    matcher.add("StimmeDeutschlands", [pattern_stimme])

    for i, token in enumerate(doc):
        is_metonym = token.text in metonyms
        is_loc = token.ent_type_ == 'LOC'
        condition_1, condition_2, condition_3, condition_4 = False, False, False, False

        if token.pos_ in ["NOUN", "PROPN"]:
            if any(child.dep_ == "nk" and child.pos_ == "ADJ" and child.text in animated_adjectives for child in token.children):
                condition_3 = True

            if (token.pos_ in ["NOUN", "PROPN"]) and (is_metonym or is_loc):
                head = token.head
                if token.dep_ == "sb" and head.pos_ == "VERB" and head.text in verbs:
                    condition_1 = True

                if token.dep_ == "nk":
                    grand_head = head.head
                    if head.pos_ == "ADP" and head.dep_ == "sbp" and grand_head.pos_ == "VERB" and grand_head.text in verbs:
                        condition_2 = True

        matches = matcher(doc)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            if string_id in ["WirAls", "GesichtDeutschlands", "StimmeDeutschlands"]:
                if i == end - 1:
                    condition_4 = True

        if condition_1 or condition_2 or condition_3 or condition_4:
            j = i
            while j >= 0:
                try:
                    if tokens[j][4] == 'B':
                        break
                except IndexError:
                    break
                j -= 1

            while j < len(tokens):
                try:
                    if tokens[j][4] in ['B', 'I']:
                        tokens[j][3] = 'animated'
                    else:
                        break
                except IndexError:
                    break
                j += 1

    output_lines.extend(tokens)
    output_lines.append([])
    return output_lines


nlp = spacy.load('de_core_news_lg')

with open('gold_metonymy_10_0.7.txt', 'r', encoding='utf-8') as file:
    metonyms = [line.strip() for line in file]

with open('konjugierte_verben.txt', 'r', encoding='utf-8') as file:
    verbs = [line.strip() for line in file]

with open('animated_adjectives_5_0.9.txt', 'r', encoding='utf-8') as file:
    animated_adjectives = [line.strip() for line in file]

input_directory = 'C:/Users/fredh/Documents/BA/Labeling/Animated'
output_directory = 'C:/Users/fredh/Documents/BA/Labeling/Metonym'

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    file_path = os.path.join(input_directory, filename)

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        lines = list(reader)

    current_text = ""
    tokens = []
    output_lines = lines[:3]
    for line in lines[3:]:
        if not line or len(line) == 0:
            output_lines.extend(analyze_text(current_text, tokens, nlp, metonyms, animated_adjectives, verbs))
            current_text = ""
            tokens = []
        elif line[0].startswith("#Text="):
            current_text = line[0][6:]
            output_lines.append(line)
        else:
            tokens.append(line)

    if current_text and tokens:
        output_lines.extend(analyze_text(current_text, tokens, nlp, metonyms, animated_adjectives, verbs))

    output_filename = os.path.splitext(filename)[0] + '_metonym' + os.path.splitext(filename)[1]
    output_path = os.path.join(output_directory, output_filename)
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(output_lines)
