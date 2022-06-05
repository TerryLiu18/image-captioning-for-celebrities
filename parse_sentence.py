import os
import re
import nltk
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# text = 'A boy wearing a blue life vest and camo shorts jumps off a diveboard into a blue lake .'
# text = 'A man and a woman are having dinner .'
# sentence = nltk.word_tokenize(text)
# people = 'Biden'

# sentence_tag = nltk.pos_tag(sentence)
# print(sentence_tag)

# define grammar
# NP(noun phrase): DT (JJ | NN) NN
# grammar = "NP: {<DT>*(<NN.*>|<JJ.*>)*<NN.*>}"
# cp = nltk.RegexpParser(grammar)
# tree = cp.parse(sentence_tag)
# print(tree)
# print(sentence)
# words = []

def parse_np(text):
    sentence = nltk.word_tokenize(text)
    sentence_tag = nltk.pos_tag(sentence)
    grammar = "NP: {<DT>*(<NN.*>|<JJ.*>)*<NN.*>}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(sentence_tag)

    words = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            # print("this is subtree",subtree)
            # print("this is subtree leaves", subtree.leaves())
            # print("this is subtree leaves[0][0]", subtree.leaves()[0][0])
            # print(type(subtree.leaves()))
            for leaf in subtree.leaves():
                if leaf[0] in ('boy', 'girl', 'man', 'woman', 'people', 'child'):
                    # print(f'----------------this subtree {subtree} should be replace----------------')
                    ww = ''
                    for w in subtree.leaves():
                        ww += w[0] + ' '
                    ww = ww.strip()
                    words.append(ww)
    return words


def replace_np(text, words, names):
    # replace np using re
    if len(words) == len(names):
        print('perfectly match')
        for i in range(len(words)):
            word, name = words[i], names[i]
            if name == 'nobody':
                continue
            text = re.sub(word, name, text)  
    else:
        print(f'{words} : {names}') 

    return text


if __name__ == '__main__':
    # test
    texts = [
        'A child in a pink dress is climbing up a set of stairs in an entry way .',
        'A girl going into a wooden building .',
        'A black dog and a white dog with brown spots are staring at each other in the street .',
        'A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it .',
        'The man with pierced ears is wearing glasses and an orange hat .',
        'A young child is walking on a stone paved street with a metal pole and a man behind him .'
    ]
    
    for text in texts:
        words = parse_np(text)
        # if words:
        #     print(f'{words} | {text}.')
        # else:
        #     print(f'[not found] | {text}.')
        text = replace_np(text, words, ['Biden', 'trump'])
        print(text + '\n')
