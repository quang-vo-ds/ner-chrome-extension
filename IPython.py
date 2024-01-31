from backend.ner_src.vocab import Vocab
from backend.ner_src import utils

tag_path = "backend/ner_src/vocab/tag_vocab.json"
vocab = Vocab.load(tag_path)
utils.print_var(**vocab.get_word2id())
print(vocab["3"])