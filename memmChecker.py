from viterbi import *
import time

class MemmChecker(object):
    def check(self, model, corpus):
        v = Viterbi(model)

        resultFileName = 'results_' + time.strftime("%Y%m%d_%H%M%S") + '.txt'
        print('Starting validation, writing to ', resultFileName)
        fp = open(resultFileName, 'w')
        totalTags = 0
        totalErrors = 0
        totalSentence = 0
        for i in range(len(corpus.getSentences())):
            totalSentence += 1
            # print('sample ', str(i), ':')
            tags = v.solve(corpus.getSentencesW()[i])
            vTags = corpus.getSentencesT()[i]
            vWords = corpus.getSentencesW()[i]
            if len(tags) != len(vWords) - 2:
                print(len(tags))
                print(tags)
                print(len(vWords))
                print(vWords)
                exit()
            for j in range(2, len(vTags) - 1):
                totalTags += 1
                if (vTags[j] != tags[j - 2]):
                    totalErrors += 1
                    # print('Error:', vTags[j], tags[j - 2])
                    fp.write(
                        "Error! sentence:%0d  tagIdx:%0d  tag[i]:%s  vTag[i]:%s  word[i]:%s  vTag[i-1]:%s  word[i-1]:%s  vTag[i-2]:%s  word[i-2]:%s\n" % (
                            totalSentence, totalTags, tags[j - 2], vTags[j], vWords[j], vTags[j - 1], vWords[j - 1], vTags[j - 2], vWords[j - 2]))

            print('#sentence: ', totalSentence, '#tags: ', totalTags, 'Total Errors: ', totalErrors, 'Precision: ',
                  float(totalTags - totalErrors) / totalTags)
            fp.write("#sentence: %s, #Tags: %s, #errors: %s, Precision: %s\n" % (
                totalSentence, totalTags, totalErrors, float(totalTags - totalErrors) / totalTags))
