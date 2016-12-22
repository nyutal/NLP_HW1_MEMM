class c(object):
    def __init__(self, word):
        self.count = 1
        self.words = set()
        self.words.add(word)

    def addWord(self,word):
        self.words.add(word)
        self.count += 1

sentences = []
conf = {}
conf2 = {}
for i in range(1,8):
    lines = [line.rstrip('\n') for line in open('t' + str(i) + '.txt')]
    samples = 0
    for line in lines:
        w = line.split(" ")
        if w[0] == 'Error!':
            our_tag  = w[5].split(":")
            real_tag = w[7].split(":")
            word = w[9].split(":")[1]
            h = (our_tag[1], real_tag[1])
            if h in conf:
                conf[h].addWord(word)
                conf2[h] += 1
            else:
                conf[h] = c(word)
                conf2[h] = 1

for w in sorted(conf2, key=conf2.get, reverse=True)[0:20]:
  print ('our:'+ str(w[0])+ ', real:'+ str(w[1]), 'errors:' + str(conf2[w]), conf[w].words)