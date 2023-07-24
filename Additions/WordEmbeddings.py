# Extending wordlist using fastText

import traceback
import spacy
from gensim.models.fasttext import load_facebook_model

def extend_word_list(model_path, word_list_path, extended_word_list_path, similarity_threshold=0.7, max_similar_words=10, allowed_word_types={"NOUN"}):
    try:
        nlp = spacy.load('de_core_news_lg')
        model = load_facebook_model(model_path)

        with open(word_list_path, 'r', encoding='utf-8') as file:
            word_list = [word.strip() for word in file.readlines()]

        extended_word_list = set(word_list)
        for word in word_list:
            similar_words = model.wv.most_similar(positive=[word], topn=max_similar_words)
            for similar_word, similarity in similar_words:
                if similarity >= similarity_threshold:
                    doc = nlp(similar_word)
                    if doc[0].pos_ in allowed_word_types:
                        extended_word_list.add(similar_word)

        with open(extended_word_list_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(sorted(extended_word_list)))

        for word in extended_word_list:
            print(word)

    except Exception as e:
        traceback.print_exc()

model_path = r'C:\Users\domin\Documents\cc.de.300.bin'
word_list_path = 'word_list.txt'
extended_word_list_path = 'word_list_extended.txt'
extend_word_list(model_path, word_list_path, extended_word_list_path, similarity_threshold=0.7, max_similar_words=10, allowed_word_types={"NOUN"})
