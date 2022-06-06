# %%
import os
import re
import nltk
import matplotlib as mpl

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

PEOPLE_WORDS = ('boy', 'boys', 'girl', 'girls', 'man', 'men',
                'woman', 'women', 'person', 'people', 'child', 'children')

grammar = "NP: {(<DT>|<CD>)*(<NN.*>|<JJ.*>)*<NN.*>}"

# %%

def is_np(subtree):
    '''check if subtree is np for people
    Args:
        subtree: nltk.tree.Tree
    Returns:
        None if not np, else return a list of str
    '''
    if subtree.label() != 'NP':
        return False

    phrase = [leaf[0] for leaf in subtree.leaves()]
    flag_num = flag_people = False

    for word in phrase:
        if NUM2WORDS.get(word.lower()) is not None:
            flag_num = True
        if word.lower() in PEOPLE_WORDS:
            flag_people = True
    if not flag_num or not flag_people:
        return False
    return True


def count_num_of_people(subtree: nltk.tree.Tree) -> int:
    '''count people num in sentence
    Args:
        subtree: nltk.Tree.Tree, 
    Returns:
        text: str
        num_of_people: int
    '''
    if not is_np(subtree):
        return 0

    num_of_people = 0
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

    total_num_of_people = 0
    for subtree in tree.subtrees():
        if not is_np(subtree): continue

        word = [leaf[0] for leaf in subtree.leaves()]
        words.append(' '.join(word).strip())
        total_num_of_people += count_num_of_people(subtree)

    return words, total_num_of_people, tree



def replace_np(text, words, total_num_of_people, names, tree):
    '''replace np in sentence

    Args:
        text: str
        words: list of str, words to replace
        name_of_people: list of str, name of people
        names: list of str, names to replace
        tree: nltk.tree.Tree

    Returns:
        text: str, parsed text
    '''

    if total_num_of_people != len(names):
        # print(f'name of people: {nums_of_people} != names: {names}')
        return text

    # print(f'{words} : {names}') 
    if len(words) == len(names):
        for i in range(len(words)):
            word, name = words[i], names[i]
            if name == 'nobody':
                continue
            text = re.sub(word, name, text)  

    # cope with 'two men' cases
    else:
        cnt = 0
        for subtree in tree.subtrees():
            if not is_np(subtree): continue

            word = [leaf[0] for leaf in subtree.leaves()]
            text_to_be_replaced = ' '.join(word).strip()
            entity_name = ''
            num_of_people = count_num_of_people(subtree)

            for i in range(num_of_people):
                if num_of_people == 1:
                    name = names[cnt] if names[cnt] != 'nobody' else text_to_be_replaced
                else:
                    name = names[cnt + i] if names[cnt + i] != 'nobody' else 'someone'
                if i < num_of_people - 1:
                    entity_name += name + ' and '
                else:
                    entity_name += name

            text = re.sub(text_to_be_replaced, entity_name, text)  
            cnt += num_of_people
    return text

# texts = [
#     # 'A child in a pink dress is climbing up a set of stairs in an entry way .',
#     # 'A girl going into a wooden building .',
#     # 'A black dog and a white dog with brown spots are staring at each other in the street .',
#     # 'A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it .',
#     # 'The man with pierced ears is wearing glasses and an orange hat .',
#     # 'Two young children is walking on a stone paved street with a metal pole and a man behind him .',
#     'Three men are Having dinner and a woman is behind them'
# ]



if __name__ == '__main__':
    # test
    texts = [
        ('A child in a pink dress is climbing up a set of stairs in an entry way .', ['Biden']),
        ('A girl going into a wooden building .', ['Sara']),
        ('A black dog and a white dog with brown spots are staring at each other in the street .', ['']),
        ('A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it .',
        ['Lucy']),
        ('The man with pierced ears is wearing glasses and an orange hat .', ['Trump']),
        ('Two young children is walking on a stone paved street with a metal pole and a man is behind them.', 
        ['Trump', 'nobody', 'Jim']),
        ('Three men are Having dinner and a woman is behind them', ['Biden', 'trump', 'nobody', 'nobody'])
    ]
    
    for text, face_list in texts:
        words, num_of_people, tree = parse_np(text)
        # print('num of people', num_of_people)
        # if words:
        #     print(f'{words} | {text}.')
        # else:
        #     print(f'[not found] | {text}.')
        text = replace_np(text, words, num_of_people, face_list, tree)
        print(text + '\n')

# %%
