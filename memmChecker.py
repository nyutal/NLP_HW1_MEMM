from viterbi import *
import time
from consts import CONST
import multiprocessing as mp
import uuid

class MemmChecker(object):
    def check(self, model, corpus, fpMain):
        j = []
        num_of_proc = CONST.num_of_viterbers
        sen_per_proc = int( ( len(corpus.getSentences()) + num_of_proc - 1 ) / num_of_proc)
        fpMain.write("learn corpus = %s\n" % model.corpus.getFileInfo())

        if CONST.parallel:
            for i in range(num_of_proc):
                if i < num_of_proc-1:
                    j.append((corpus, model, sen_per_proc * i, sen_per_proc * (i + 1)))
                else:
                    j.append((corpus, model, sen_per_proc*i, len(corpus.getSentences())))

            with mp.Pool() as pool:
                ret = pool.map(calc_viterbi, j)
            totalTags, totalErrors = [sum(x) for x in zip(*ret)]
            print('totalTags=', totalTags, ', totalErrors=', totalErrors, ', precision=', (totalTags - totalErrors) / totalTags)
            fpMain.write("#sentence: %s, #Tags: %s, #errors: %s, Precision: %s\n" % (
                totalSentence, totalTags, totalErrors, float(totalTags - totalErrors) / totalTags))
        else:
            v = Viterbi(model)
            resultFileName = 'results_' + time.strftime("%Y%m%d_%H%M%S") + '.txt'
            print('Starting validation, writing to ', resultFileName)
            fp = open(resultFileName, 'w')
            fp.write("lambda=%s\n" % CONST.reg_lambda)
            fp.write("feature generators = %s\n" % model.getFeatureGenString())
            fp.write("learn corpus = %s\n" % model.corpus.getFileInfo())
            totalTags = 0
            totalErrors = 0
            totalSentence = 0
            for i in range(len(corpus.getSentences())):
                totalSentence += 1
                tags = v.fullSolve(corpus.getSentencesW()[i])
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
                                totalSentence, (j - 1), tags[j - 2], vTags[j], vWords[j], vTags[j - 1],
                                vWords[j - 1],
                                vTags[j - 2], vWords[j - 2]))

                print('#sentence: ', totalSentence, '#tags: ', totalTags, 'Total Errors: ', totalErrors, 'Precision: ',
                      float(totalTags - totalErrors) / totalTags)
                fp.write("#sentence: %s, #Tags: %s, #errors: %s, Precision: %s\n" % (
                    totalSentence   , totalTags, totalErrors, float(totalTags - totalErrors) / totalTags))
                fp.flush()
                fpMain.write("#sentence: %s, #Tags: %s, #errors: %s, Precision: %s\n" % (
                    totalSentence, totalTags, totalErrors, float(totalTags - totalErrors) / totalTags))
            fp.close()

    def compete(self, model, corpus):
        v = Viterbi(model)
        resultFileName = 'comp_' + time.strftime("%Y%m%d_%H%M%S") + '.txt'
        resultFileName2 = 'MYcomp_' + time.strftime("%Y%m%d_%H%M%S") + '.txt'
        print('Starting validation, writing to ', resultFileName)
        fp = open(resultFileName, 'w')
        fp2 = open(resultFileName2, 'w')
        for i in range(len(corpus.getSentences())):
            print(corpus.getSentencesW()[i])
            tags = v.fullSolve(corpus.getSentencesW()[i])
            print(tags)
            print()
            vWords = corpus.getSentencesW()[i]
            if len(tags) != len(vWords) - 2:
                print(len(tags))
                print(tags)
                print(len(vWords))
                print(vWords)
                exit()
            vWords = vWords[2:]
            for j in range(len(tags) - 1):
                fp.write(str(vWords[j]) + '_' + str(tags[j]) + ' ')
                fp2.write(str(vWords[j]) + '|' + str(tags[j]) + ' ')
            fp.write('\n')
            fp.flush()
            fp2.write('\n')
            fp2.flush()
        fp.close()


def calc_viterbi(params):
    corpus, model, start, batch_size = params
    v = Viterbi(model)
    resultFileName = 'results_' + time.strftime("%Y%m%d_%H%M%S") + '_' + str(uuid.uuid1())[0:5] + '.txt'
    print('Starting validation, writing to ', resultFileName)
    fp = open(resultFileName, 'w')
    fp.write("lambda=%s\n" % CONST.reg_lambda)
    fp.write("feature generators = %s\n" % model.getFeatureGenString())
    fp.write("learn corpus = %s\n" % model.corpus.getFileInfo())
    totalTags = 0
    totalErrors = 0
    totalSentence = 0
    for i in range(start, batch_size):
        totalSentence += 1
        tags = v.fullSolve(corpus.getSentencesW()[i])
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
                        (start+totalSentence), (j-1), tags[j - 2], vTags[j], vWords[j], vTags[j - 1], vWords[j - 1],
                        vTags[j - 2], vWords[j - 2]))

        print('#sentence: ', totalSentence, '#tags: ', totalTags, 'Total Errors: ', totalErrors, 'Precision: ',
              float(totalTags - totalErrors) / totalTags)
        fp.write("#sentence: %s, #Tags: %s, #errors: %s, Precision: %s\n" % (
            (start + totalSentence), totalTags, totalErrors, float(totalTags - totalErrors) / totalTags))
        fp.flush()
    fp.close()
    return totalTags, totalErrors