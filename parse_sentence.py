import os
import re
import nltk
import matplotlib as mpl

from parse import PEOPLE_WORDS

NUM2WORDS = {
    'a': 1,
    'the': 1,
    '1': 1,
    'one': 1,
    '2': 2,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
}

grammar = "NP: {(<DT>|<CD>)*(<NN.*>|<JJ.*>)*<NN.*>}"

def count_num_of_people(tree: nltk.tree.Tree) -> int:
    '''count people num in sentence
    Args:
        sentence_tag: list of tuple, [(word, tag), ...]
        tree: nltk.Tree, 
    Returns:
        text: str
        num_of_people: int
    '''
    print('tree:', tree)
    num_of_people = 0
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            for leaf in subtree.leaves():
                if leaf[1] in ('NN', 'NNP', 'NNPS', 'CD', 'DT') and leaf[0].lower() in NUM2WORDS.keys():
                    num_of_people += NUM2WORDS[leaf[0].lower()]

    return num_of_people



def parse_np(text):
    '''parse np in sentence
    Args:
        text: str
    Returns:
        text: str, parsed text
        num_of_people: int,  number of people in the sentence
        tree: nltk.tree.Tree
    '''
    sentence = nltk.word_tokenize(text)
    sentence_tag = nltk.pos_tag(sentence)

    grammar = "NP: {(<DT>|<CD>)*(<NN.*>|<JJ.*>)*<NN.*>}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(sentence_tag)

    words = []
    num_of_people = count_num_of_people(tree)
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            for leaf in subtree.leaves():
                if leaf[0] in PEOPLE_WORDS:
                    # print(f'----------------this subtree {subtree} should be replace----------------')
                    ww = ''
                    for w in subtree.leaves():
                        ww += w[0] + ' '
                    ww = ww.strip()
                    words.append(ww)

    return words, num_of_people, tree


def replace_np(text, words, name_of_people, names, tree):
    # replace np using re
    if name_of_people != len(names):
        print(f'name of people: {name_of_people} != names: {names}')
        return text

    print(f'{words} : {names}') 
    if len(words) == len(names):
        # print('perfectly match')
        for i in range(len(words)):
            word, name = words[i], names[i]
            if name == 'nobody':
                continue
            text = re.sub(word, name, text)  

    # cope with 'two men' cases
    else:
        for subtree in tree.subtrees():
            if subtree.label() != 'NP':
                continue
            
            cnt = 0
            for leaf in subtree.leaves():
                if leaf[0].lower() in NUM2WORDS.keys():
                    num_of_people = NUM2WORDS[leaf[0].lower()]
                    name = names[cnt] if names[cnt] != 'nobody' else 'someone'
                    subtree.leaves()[cnt] = (name, leaf[1])
                    
                    # text = re.sub(leaf[0], name, text)
                    cnt += 1
                    if cnt == len(names):
                        break
                    
                    cnt += 1 
                
    return text


if __name__ == '__main__':
    # test
    texts = [
        # 'A child in a pink dress is climbing up a set of stairs in an entry way .',
        # 'A girl going into a wooden building .',
        # 'A black dog and a white dog with brown spots are staring at each other in the street .',
        # 'A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it .',
        # 'The man with pierced ears is wearing glasses and an orange hat .',
        # 'Two young children is walking on a stone paved street with a metal pole and a man behind him .',
        'Three men are Having dinner and a woman is behind them'
    ]
    
    for text in texts:
        words, num_of_people, tree = parse_np(text)
        print('num of people', num_of_people)
        # if words:
        #     print(f'{words} | {text}.')
        # else:
        #     print(f'[not found] | {text}.')
        text = replace_np(text, words, num_of_people, ['Biden', 'trump', 'nobody', 'nobody'], tree)
        print(text + '\n')
