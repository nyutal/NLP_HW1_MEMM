
class SentenceParser(object):

    def parseTagedFile(self, sentences, sentences_t, sentences_w, tags, fileName, maxSentences=-1):
        # sentences = []
        # sentences_w = []
        # sentences_t = []
        # tags = set()

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
        for sentence in sentences:
            w = []
            tag = []
            for word in sentence:
                a, b = word.split("_")
                w.append(a)
                tag.append(b)
                tags.add(b)  # create a set of all tags

            sentences_w.append(w)
            sentences_t.append(tag)
        # tags.remove("SSS")
        tags.remove("*")

        print('SentenceParser parded tag file with ', len(sentences))
        print('all tags(', len(tags), '):', tags)
        # return (sentences, sentences_t, sentences_w, tags)