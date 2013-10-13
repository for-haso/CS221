
import argparse
import sys
import util
import wordsegUtil

def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument('--text-corpus', help='Text training corpus')
    p.add_argument('--model', help='Always use this model')
    return p.parse_args()

############################################################
# Problem 1d: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return self.query
        # END_YOUR_CODE

    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return state == ""
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (around 7 lines of code expected)
        succAndCosts = list()
        for i in range(1, len(state) + 1):
            succAndCosts.append((state[:i], state[i:], self.unigramCost(state[:i])))
        return succAndCosts
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (around 3 lines of code expected)
    return " ".join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 2c: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # state = (prev word, remaining words)
        return (wordsegUtil.SENTENCE_BEGIN, tuple(self.queryWords))
        # END_YOUR_CODE

    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return state[1] == tuple()
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (around 9 lines of code expected)
        next = state[1][0]
        remaining = tuple(list(state[1])[1:])
        possibles = self.possibleFills(next)
        succAndCosts = list()
        for word in possibles:
            succAndCost = (word, (word, remaining), self.bigramCost(state[0], word))
            succAndCosts.append(succAndCost)
        return succAndCosts
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (around 3 lines of code expected)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    # BEGIN_YOUR_CODE (around 3 lines of code expected)
    if ucs.actions == None:
        return " ".join(queryWords)
    return " ".join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        # state = (prev word, rest of string)
        return (wordsegUtil.SENTENCE_BEGIN, self.query)
        # END_YOUR_CODE

    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return state[1] == ""
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (around 15 lines of code expected)
        succAndCosts = list()
        for i in range(1, len(state[1]) + 1):
            # get all substrings of the remaining string
            # then get all possibleFills for the first part, and add those as succAndCosts
            next = state[1][:i]
            remaining = state[1][i:]
            possibles = self.possibleFills(next)
            for word in possibles:
                succAndCost = (word, (word, remaining), self.bigramCost(state[0], word))
                succAndCosts.append(succAndCost)
        return succAndCosts
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    if ucs.actions == None:
        return ""
    return " ".join(ucs.actions)
    # END_YOUR_CODE

############################################################
# REPL and main entry point

def repl(unigramCost, bigramCost, possibleFills, command=None):
    '''REPL: read, evaluate, print, loop'''

    while True:
        sys.stdout.write('>> ')
        line = sys.stdin.readline().strip()
        if not line:
            break

        if command is None:
            cmdAndLine = line.split(None, 1)
            cmd, line = cmdAndLine[0], ' '.join(cmdAndLine[1:])
        else:
            cmd = command
            line = line

        print ''

        if cmd == 'help':
            print 'Usage: <command> [arg1, arg2, ...]'
            print ''
            print 'Commands:'
            print '\n'.join(a + '\t\t' + b for a, b in [
                ('help', 'This'),
                ('seg', 'Segment character sequences'),
                ('ins', 'Insert vowels into words'),
                ('both', 'Joint segment-and-insert'),
                ('fills', 'Query possibleFills() to see possible vowel-fillings of a word'),
                ('ug', 'Query unigram cost function'),
                ('bg', 'Query bigram cost function'),
            ])
            print ''
            print 'Enter empty line to quit'

        elif cmd == 'seg':
            line = wordsegUtil.cleanLine(line)
            parts = wordsegUtil.words(line)
            print '  Query (seg):', ' '.join(parts)
            print ''
            print '  ' + ' '.join(segmentWords(part, unigramCost) for part in parts)

        elif cmd == 'ins':
            line = wordsegUtil.cleanLine(line)
            ws = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(line)]
            print '  Query (ins):', ' '.join(ws)
            print ''
            print '  ' + insertVowels(ws, bigramCost, possibleFills)

        elif cmd == 'both':
            line = wordsegUtil.cleanLine(line)
            smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
            parts = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(line)]
            print '  Query (both):', ' '.join(parts)
            print ''
            print '  ' + ' '.join(
                segmentAndInsert(part, smoothCost, possibleFills)
                for part in parts
            )

        elif cmd == 'fills':
            line = wordsegUtil.cleanLine(line)
            print '\n'.join(possibleFills(line))

        elif cmd == 'ug':
            line = wordsegUtil.cleanLine(line)
            print unigramCost(line)

        elif cmd == 'bg':
            grams = tuple(wordsegUtil.words(line))
            prefix, ending = grams[:-1], grams[-1]
            print bigramCost(prefix, ending)

        else:
            print 'Unrecognized command:', cmd

        print ''

def main(args):
    if args.model and args.model not in ['seg', 'ins', 'both']:
        print 'Unrecognized model:', args.model
        sys.exit(1)

    corpus = args.text_corpus or 'leo-will.txt'

    sys.stdout.write('Training language cost functions [corpus: %s]... ' % corpus)
    sys.stdout.flush()

    unigramCost, bigramCost = wordsegUtil.makeLanguageModels(corpus)
    possibleFills = wordsegUtil.makeInverseRemovalDictionary(corpus, 'aeiou')

    print 'Done!'
    print ''

    repl(unigramCost, bigramCost, possibleFills, command=args.model)

if __name__ == '__main__':
    args = parseArgs()
    main(args)
