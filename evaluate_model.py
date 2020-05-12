from spacy.scorer import Scorer
from spacy.gold import GoldParse


def evaluate():
    scorer = Scorer()
    output_dir = "".join([os.path.dirname(os.path.abspath(__file__)) , '/model'])
    examples =[
    ('Who is Java?', {
        'entities': [(7, 11, 'Skill')]
    }),
    ('I like C++ and C.', {
        'entities': [(7, 10, 'Skill'), (15, 17, 'Skill')]
    })
]
    ner_model = spacy.load(output_dir)
    #resumeText = readFile(examples)
    
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    print(scorer.scores)
