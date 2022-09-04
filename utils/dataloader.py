import spacy
from torchtext.legacy import data, datasets
import torchtext
def main():
    def tokenize(text):
        return text.split()

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    TGT = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)

    print("Loading Dataset")
    music_text = open('./data.txt', 'r')
    music_lines = list(music_text)

    fields = [("trg", TGT)]
    examples = [torchtext.legacy.data.Example.fromlist([(music_lines[i])], fields) for i in range(len(music_lines))]

    MAX_LEN = 1500
    train, val = torchtext.legacy.data.Dataset(examples, fields=fields, filter_pred=lambda x:
    len(vars(x)['trg']) <= MAX_LEN).split()

    MIN_FREQ = 1
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
if __name__ == "__main__":
    main()
