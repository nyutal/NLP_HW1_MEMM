
class Corpus(object):
    def __init__(self, sentences, sentences_t, sentences_w, tags, fileName, maxSentences):
        self.sentences = sentences
        self.sentences_w = sentences_w
        self.sentences_t = sentences_t
        self.tags = tags
        self.fileName = fileName
        self.maxSentences = maxSentences

    def getSentences(self):
        return self.sentences

    def getSentencesW(self):
        return self.sentences_w

    def getSentencesT(self):
        return self.sentences_t

    def getTags(self):
        return self.tags

    def getFileInfo(self):
        s = ''
        s += self.fileName + ' with #samples: ' + str(self.maxSentences)
        return s

class SentenceParser(object):
    def parseTaggedFile(self, fileName, maxSentences=-1):
        sentences = []
        sentences_w = []
        sentences_t = []
        tags = set()

        lines = [line.rstrip('\n') for line in open(fileName)]
        samples = 0
        for line in lines:
            w = ['*_*', '*_*']  # start
            w.extend(line.split(" "))
            w.append('SSS_SSS')  # stop
            sentences.append(w)
            samples += 1
            if maxSentences != -1 and samples == maxSentences: break

        # sentences = sentences[0:100]

        capital_pos = ['NNP', 'NNPS', 'PRP'] # these words have to stay capitalized

        for sentence in sentences:
            w = []
            tag = []
            for idx, word in enumerate(sentence):
                a, b = word.split("_")
                if (idx == 2) and (b not in capital_pos):
                    w.append(a.lower())
                else:
                    w.append(a)
                tag.append(b)
                tags.add(b)  # create a set of all tags

            sentences_w.append(w)
            sentences_t.append(tag)
        # tags.remove("SSS")
        tags.remove("*")

        print('SentenceParser parsed tag file with ', len(sentences))
        print('all tags(', len(tags), '):', tags)

        return Corpus(sentences, sentences_t, sentences_w, tags, fileName, maxSentences)
        # return (sentences, sentences_t, sentences_w, tags)

    def parseUnTaggedFile(self, fileName, maxSentences=-1):
        sentences = []
        sentences_w = []
        sentences_t = []
        tags = {'VBN', '.', 'WRB', 'RBS', 'NN', 'CC', 'MD', 'PRP$', 'UH', 'EX', 'PRP', 'WP$', 'RB', 'CD', 'NNS', 'DT', 'JJS', '``', 'PDT', "''", 'POS', 'WP', 'NNPS', 'VBG', '-LRB-', 'RP', 'IN', 'TO', 'NNP', 'SYM', '-RRB-', '$', 'FW', 'RBR', 'WDT', 'JJ', '#', ':', 'VBP', 'JJR', 'VBZ', ',', 'VBD', 'SSS', 'VB'}
        lines = [line.rstrip('\n') for line in open(fileName)]
        samples = 0
        for line in lines:
            w = ['*','*']  # start
            w.extend(line.split(" "))
            w.append('SSS')  # stop
            sentences.append(w)
            samples += 1
            sentences_w.append(w)
            if maxSentences != -1 and samples == maxSentences: break

        print('SentenceParser parsed word file with ', len(sentences))
        print(sentences_w)

        return Corpus(sentences, sentences_t, sentences_w, tags, fileName, maxSentences)