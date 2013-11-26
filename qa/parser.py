# Simple semantic parser that converts sentences into first-order logic.
# @author Percy Liang

import sys, collections, re
from logic import *

############################################################
# Representation of a natural language utterance.

class Utterance:
    def __init__(self, sentence):
        # Preprocess
        sentence = re.sub(r"\ban\b", 'a', sentence)
        sentence = re.sub(r"\bdon't\b", 'do not', sentence)
        sentence = re.sub(r"\bdoesn't\b", 'does not', sentence)
        sentence = re.sub(r"\bit's\b", 'it is', sentence)
        sentence = re.sub(r"\bIt's\b", 'It is', sentence)
        self.sentence = sentence

        # Tokenize, POS tag, stem
        try:
            import nltk

            ### Perform tokenization.
            self.tokens = nltk.word_tokenize(self.sentence)

            ### Part-of-speech tagging
            self.pos_tags = map(lambda x : x[1], nltk.pos_tag(self.tokens))

            # Hack arround errors in crappy POS tagger
            # Mistakingly tags VBZ as NNS after a NNP (e.g., John lives in a house)
            for i in range(len(self.pos_tags)-1):
                if self.pos_tags[i] == 'NNP' and self.pos_tags[i+1] == 'NNS':
                    self.pos_tags[i+1] = 'VBZ'
            for i in range(len(self.pos_tags)):
                if self.tokens[i] == 'red': self.pos_tags[i] = 'JJ' # Mistagged as VBN
                if self.tokens[i] == 'lives': self.pos_tags[i] = 'VBZ' # Mistagged as NNS

            # Lemmatize
            lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
            def lemmatize(token, tag):
                # Only lemmatize with the proper POS
                if token == 'people': return 'person'
                if tag[0] in 'VN':
                    return lemmatizer.lemmatize(token.lower(), tag[0].lower())
                else:
                    return token
            self.tokens = [lemmatize(token, tag) for token, tag in zip(self.tokens, self.pos_tags)]

        except ImportError, e:
            print "NLTK failed, falling back on simple tokenization", e
            def tag(w):
                if w[0].isupper(): return 'NNP'
                if w.endswith('ing'): return 'VBG'
                if w.endswith('ed'): return 'VBD'
                return 'NN'
            self.tokens = self.sentence.split()
            self.pos_tags = [tag(token) for token in self.tokens]

        # Lowercase after NLP is done
        self.tokens = [x.lower() for x in self.tokens]

    def __str__(self):
        return ' '.join('%s/%s' % x for x in zip(self.tokens, self.pos_tags))

############################################################
# A Grammar maps a sequence of natural language tokens to a set of ParserDerivations.

# Try to look inside a lambda (not perfect)
def lambda_rstr(x):
    if isinstance(x, tuple): return str(tuple(map(lambda_rstr, x)))
    if callable(x):
        v = Constant('v')
        try:
            return "(%s => %s)" % (v, lambda_rstr(x(v)))
        except:
            try:
                return "(%s => %s)" % (v, lambda_rstr(x(lambda a : Constant('v('+str(a)+')'))))
            except:
                pass
    return rstr(x)
#print lambda_rstr(lambda x : lambda y : Atom('A', x, y))

# A grammar rule parses the right-hand side |rhs| into a |lhs|
class GrammarRule:
    def __init__(self, lhs, rhs, sem, score = 0):
        self.lhs = lhs
        self.rhs = rhs
        self.sem = sem
        self.score = score
    def __str__(self):
        return "{%s -> %s, score=%s}" % (self.lhs, self.rhs, self.score)

def createSimpleEnglishGrammar():
    rules = []

    ### Lexical entries

    # Entity
    for n in [1, 2, 3]:
        rules.append(GrammarRule('$Entity', ['$NNP'] * n, lambda args : Constant('_'.join(args))))
    rules.append(GrammarRule('$Entity', ['$NN'], lambda args : args[0]))
    # Binary
    for tag in ['$VB', '$VBP', '$VBD', '$VBZ']:
        rules.append(GrammarRule('$Binary', [tag], lambda args : lambda y : lambda x : Atom(args[0].capitalize(), x, y)))
        rules.append(GrammarRule('$Binary', [tag, '$IN'], lambda args : lambda y : lambda x : Atom(args[0].capitalize()+'_'+args[1], x, y)))
    rules.append(GrammarRule('$Binary', ['be', '$IN'], lambda args : lambda y : lambda x : Atom(args[0].capitalize(), x, y)))
    rules.append(GrammarRule('$Binary', ['be', '$VBN', '$IN'], lambda args : lambda y : lambda x : Atom(args[0].capitalize()+'_'+args[1], x, y)))
    # Unary
    for tag in ['$NN', '$NNS']:
        rules.append(GrammarRule('$Unary', [tag], lambda args : lambda x : Atom(args[0].capitalize(), x)))
    for tag in ['$JJ', '$RB', '$VBN']:
        rules.append(GrammarRule('$Modifier', [tag], lambda args : lambda x : Atom(args[0].capitalize(), x)))
    # Zeroary
    for tag in ['$JJ', '$VBN', '$VBG', '$NN']:  # wet, raining, summer
        rules.append(GrammarRule('$Zeroary', [tag], lambda args : Atom(args[0].capitalize())))

    ### Simple sentences

    # Zeroary: "It's raining."
    rules.append(GrammarRule('$Clause', ['it', 'be', '$Zeroary'], lambda args : args[0]))
    rules.append(GrammarRule('$Clause', ['it', 'be', 'not', '$Zeroary'], lambda args : Not(args[0])))
    rules.append(GrammarRule('$Clause-be', ['it', '$Zeroary'], lambda args : args[0]))

    # "John was born in Seattle.", "John is happy.", "John is a cat"
    rules.append(GrammarRule('$Clause', ['$Entity', '$VP'], lambda args : args[1](args[0])))
    rules.append(GrammarRule('$Clause-be', ['$Entity', '$VP-be'], lambda args : args[1](args[0])))
    # Questions: "who is happy?"
    rules.append(GrammarRule('$Clause', ['$WP', '$VP'], lambda args : newVar(lambda v : args[1](v))))

    # "is happy", "are cats"
    rules.append(GrammarRule('$VP', ['be', '$Unary'], lambda args : args[0]))
    rules.append(GrammarRule('$VP-be', ['$Unary'], lambda args : args[0]))
    rules.append(GrammarRule('$VP', ['be', 'not', '$Unary'], lambda args : lambda x : Not(args[0](x))))
    # "is a cat"
    rules.append(GrammarRule('$VP', ['be', 'a', '$Unary'], lambda args : args[0]))
    rules.append(GrammarRule('$VP-be', ['a', '$Unary'], lambda args : args[0]))
    rules.append(GrammarRule('$VP', ['be', 'not', 'a', '$Unary'], lambda args : lambda x : Not(args[0](x))))
    # "was born in Seattle"
    rules.append(GrammarRule('$VP', ['$Binary', '$Entity'], lambda args : args[0](args[1])))
    rules.append(GrammarRule('$VP', ['do', 'not', '$Binary', '$Entity'], lambda args : lambda x : Not(args[0](args[1])(x))))

    ### Miscellaneous compositionality

    # "red house"
    rules.append(GrammarRule('$Unary', ['$Modifier'], lambda args : args[0]))
    rules.append(GrammarRule('$Unary', ['$Modifier', '$Unary'], lambda args : lambda x : And(args[0](x), args[1](x))))

    # "If it's raining, it's wet."
    rules.append(GrammarRule('$Then', [','], lambda args : None))
    rules.append(GrammarRule('$Then', [',', 'then'], lambda args : None))
    rules.append(GrammarRule('$Then', ['then'], lambda args : None))
    rules.append(GrammarRule('$Clause', ['if', '$Clause', '$Then', '$Clause'], lambda args : Implies(args[0], args[2])))

    ### Coordination

    # On entities: "John and Bill"
    rules.append(GrammarRule('$QNP', ['$Entity', 'and', '$Entity'], lambda args : lambda nuclearScope : And(nuclearScope(args[0]), nuclearScope(args[1]))))
    rules.append(GrammarRule('$QNP', ['$Entity', 'or', '$Entity'], lambda args : lambda nuclearScope : Or(nuclearScope(args[0]), nuclearScope(args[1]))))
    rules.append(GrammarRule('$QNP', ['either', '$Entity', 'or', '$Entity'], lambda args : lambda nuclearScope : Xor(nuclearScope(args[0]), nuclearScope(args[1]))))
    # On unaries: "happy or sad"
    rules.append(GrammarRule('$Unary', ['$Unary', 'and', '$Unary'], lambda args : lambda x : And(args[0](x), args[1](x))))
    rules.append(GrammarRule('$Unary', ['$Unary', 'or', '$Unary'], lambda args : lambda x : Or(args[0](x), args[1](x))))
    rules.append(GrammarRule('$Unary', ['either', '$Unary', 'or', '$Unary'], lambda args : lambda x : Xor(args[0](x), args[1](x))))

    # On sentences: "It's raining and it's snowing"
    rules.append(GrammarRule('$Clause', ['$Clause', 'and', '$Clause'], lambda args : And(args[0], args[1])))
    rules.append(GrammarRule('$Clause', ['$Clause', 'or', '$Clause'], lambda args : Or(args[0], args[1])))
    rules.append(GrammarRule('$Clause', ['either', '$Clause', 'or', '$Clause'], lambda args : Xor(args[0], args[1])))

    ### Quantification

    # Creates new variables which are not used anywhere else (using these rules).
    varCount = [0]
    def newVar(f): # Call f with the new variable
        varCount[0] += 1
        return f(Variable('$x'+str(varCount[0])))

    # Universal quantification (definitional)
    # "A cat is an animal."
    rules.append(GrammarRule('$Clause', ['a', '$Unary', 'be', 'a', '$Unary'], lambda args : newVar(lambda v : Forall(v, Implies(args[0](v), args[1](v))))))
    rules.append(GrammarRule('$Clause-be', ['a', '$Unary', 'a', '$Unary'], lambda args : newVar(lambda v : Forall(v, Implies(args[0](v), args[1](v))))))
    # "Cats are animals."
    rules.append(GrammarRule('$Clause', ['$Unary', 'be', '$Unary'], lambda args : newVar(lambda v : Forall(v, Implies(args[0](v), args[1](v))))))
    rules.append(GrammarRule('$Clause-be', ['$Unary', '$Unary'], lambda args : newVar(lambda v : Forall(v, Implies(args[0](v), args[1](v))))))
    # "If a person lives in California, he is happy. [bound anaphora]
    rules.append(GrammarRule('$Clause', ['if', 'a', '$Unary', '$VP', '$Then', '$PRP', '$VP'], lambda args : newVar(lambda v : Forall(v, Implies(And(args[0](v), args[1](v)), args[4](v))))))
    rules.append(GrammarRule('$Clause', ['if', 'a', '$Unary', '$VP', 'and', '$Clause', '$Then', '$PRP', '$VP'], lambda args : newVar(lambda v : Forall(v, Implies(And(And(args[0](v), args[1](v)), args[2]), args[5](v))))))

    # Quantifiers take a restrictor and a nuclearScope, which are both Unaries and returns a truth value
    # Universal quantification
    for text in ['each', 'all', 'every']:
        rules.append(GrammarRule('$Quantifier', [text], lambda args : \
                lambda restrictor : lambda nuclearScope : newVar(lambda v : Forall(v, Implies(restrictor(v), nuclearScope(v))))))
    rules.append(GrammarRule('$Quantifier', ['only'], lambda args : \
            lambda restrictor : lambda nuclearScope : newVar(lambda v : Forall(v, Implies(nuclearScope(v), restrictor(v))))))
    # Existential quantification
    for text in ['a', 'some']:  # Careful: 'a' could be universal quantification
        rules.append(GrammarRule('$Quantifier', [text], lambda args : \
                lambda restrictor : lambda nuclearScope : newVar(lambda v : Exists(v, And(restrictor(v), nuclearScope(v))))))
    # Negation
    for text in ['no']:
        rules.append(GrammarRule('$Quantifier', [text], lambda args : \
                lambda restrictor : lambda nuclearScope : newVar(lambda v : Not(Exists(v, And(restrictor(v), nuclearScope(v)))))))

    # Quantifier + restrictor: "every person"
    rules.append(GrammarRule('$QNP', ['$Quantifier', '$Unary'], lambda args : args[0](args[1])))
    # "likes every person" (in object position): x => Exists(y,And(House(y),Owns(x,y))
    rules.append(GrammarRule('$VP', ['$Binary', '$QNP'], lambda args : lambda x : args[1](lambda z : args[0](z)(x))))
    # "Every person ..." [note that subject quantifier takes scope over object quantifier if it exists]
    rules.append(GrammarRule('$Clause', ['$QNP', '$VP'], lambda args : args[0](args[1])))
    rules.append(GrammarRule('$Clause-be', ['$QNP', '$VP-be'], lambda args : args[0](args[1])))

    ### Final

    # Make variables x1, x2, x3, etc. (mostly cosmetic)
    def updateSubst(subst, var):
        newVar = Variable('$x' + str(len(subst) + 1))
        subst[var] = newVar
        return subst
    def canonicalizeVariables(form, subst):
        if form.isa(Variable):
            # Allow free variables
            return subst.get(form, form)
            #if form not in subst: raise Exception("Free variable found: %s" % form)
            #return subst[form]
        if form.isa(Constant): return form
        if form.isa(Atom): return apply(Atom, [form.name] + [canonicalizeVariables(arg, subst) for arg in form.args])
        if form.isa(Not): return Not(canonicalizeVariables(form.arg, subst))
        if form.isa(And): return And(canonicalizeVariables(form.arg1, subst), canonicalizeVariables(form.arg2, subst))
        if form.isa(Or): return Or(canonicalizeVariables(form.arg1, subst), canonicalizeVariables(form.arg2, subst))
        if form.isa(Implies): return Implies(canonicalizeVariables(form.arg1, subst), canonicalizeVariables(form.arg2, subst))
        if form.isa(Exists):
            newSubst = updateSubst(subst, form.var)
            return Exists(newSubst[form.var], canonicalizeVariables(form.body, newSubst))
        if form.isa(Forall):
            newSubst = updateSubst(subst, form.var)
            return Forall(newSubst[form.var], canonicalizeVariables(form.body, newSubst))
        raise Exception("Unhandled: %s" % form)

    rules.append(GrammarRule('$Statement', ['$Clause', '.'], lambda args : args[0]))
    rules.append(GrammarRule('$Question', ['$Clause', '?'], lambda args : args[0]))
    rules.append(GrammarRule('$Question', ['be', '$Clause-be', '?'], lambda args : args[0]))
    rules.append(GrammarRule('$Question', ['do', '$Clause', '?'], lambda args : args[0]))

    rules.append(GrammarRule('$ROOT', ['$Statement'], lambda args : ('tell', canonicalizeVariables(args[0], {}))))
    rules.append(GrammarRule('$ROOT', ['$Question'], lambda args : ('ask', canonicalizeVariables(args[0], {}))))

    return rules

def createToyGrammar():
    rules = []
    rules.append(GrammarRule('$S', ['the', '$B'], lambda args : ('ask', args[0])))
    rules.append(GrammarRule('$B', ['rain'], lambda args : Atom('Rain')))
    return rules

############################################################
# Parser takes an utterance, a grammar and returns a set of ParserDerivations,
# each of which contains a logical form.

# A Derivation includes a logical form, a rule, children derivations (if any) and a score.
class ParserDerivation():
    def __init__(self, form, rule, children, score):
        self.form = form
        self.rule = rule
        self.children = children
        self.score = score
    def dump(self, indent=""):
        print "%s%s: score=%s, rule: %s" % (indent, lambda_rstr(self.form), self.score, self.rule)
        for child in self.children:
            child.dump(indent + "  ")

# Return a sorted list of ParserDerivations.
# Standard bottom-up CKY parsing
def parseUtterance(utterance, rules, verbose=0):
    def isCat(x): return x.startswith('$')
    tokens = utterance.tokens
    n = len(tokens)
    beamSize = 5

    # 0    start       mid -> split       end        n
    def applyRule(start, end, mid, rule, rhsIndex, children, score):
        #print "applyRule: start=%s end=%s mid=%s, rule=%s[%s], children=%s" % (start, end, mid, rule, rhsIndex, children)
        # Need to arrive at end of tokens exactly when run out of rhs
        if (mid == end) != (rhsIndex == len(rule.rhs)):
            return

        if rhsIndex == len(rule.rhs):
            deriv = ParserDerivation(rule.sem([child.form for child in children]), rule, children, score + rule.score)
            if verbose >= 3:
                print "applyRule: %s:%s%s %s += %s" % (start, end, tokens[start:end], rule.lhs, rstr(deriv.form))
            chart[start][end][rule.lhs].append(deriv)
            return
        a = rule.rhs[rhsIndex]
        if isCat(a):  # Category
            for split in range(mid+1, end+1):
                #print "split", mid, split
                for child in chart[mid][split].get(a, {}):
                    applyRule(start, end, split, rule, rhsIndex+1, children + [child], score + child.score)
        else:  # Token
            if tokens[mid] == a:
                applyRule(start, end, mid+1, rule, rhsIndex+1, children, score)

    # Initialize 
    chart = [None] * n # start => end => category => top K derivations (logical form, score)
    for start in range(0, n):
        chart[start] = [None] * (n+1)
        for end in range(start+1, n+1):
            chart[start][end] = collections.defaultdict(list)

    # Initialize with POS tags
    for start in range(0, n):
        if utterance.tokens[start] == 'be': continue  # Don't tag this as a verb, because coupla needs to be treated specially
        chart[start][start+1]['$'+utterance.pos_tags[start]].append(ParserDerivation(utterance.tokens[start], None, [], 0))

    # Parse
    for length in range(1, n+1):  # For each span length...
        for start in range(n - length + 1):  # For each starting position
            end = start + length
            for rule in rules:
                applyRule(start, end, start, rule, 0, [], 0)

            # Prune derivations
            cell = chart[start][end]
            for cat in cell.keys():
                cell[cat] = sorted(cell[cat], key = lambda deriv : -deriv.score)[0:beamSize]

    derivations = chart[0][n]['$ROOT']
    if verbose >= 1:
        print "parseUtterance: %d derivations" % len(derivations)
    if verbose >= 2:
        for deriv in derivations:
            if verbose >= 3:
                deriv.dump("  ")
            else:
                print "  %s: score=%s" % (rstr(deriv.form), deriv.score)
    return derivations

############################################################

# Train the grammar rule scores to get the training examples correct.
# Also acts as a unit test (we should get 100% accuracy).
def trainGrammar(rules):
    # Training examples
    examples = []

    # Zeroary
    examples.append(('It is raining.', ('tell', Atom('Rain'))))
    examples.append(('It is summer.', ('tell', Atom('Summer'))))
    examples.append(('It is wet.', ('tell', Atom('Wet'))))
    examples.append(('It is not summer.', ('tell', Not(Atom('Summer')))))

    # Simple sentences
    examples.append(('John is happy.', ('tell', Atom('Happy', 'john'))))
    examples.append(('John is not happy.', ('tell', Not(Atom('Happy', 'john')))))
    examples.append(('John is a cat.', ('tell', Atom('Cat', 'john'))))
    examples.append(('John is not a dog.', ('tell', Not(Atom('Dog', 'john')))))
    examples.append(('John was born in Seattle.', ('tell', Atom('Bear_in', 'john', 'seattle'))))
    examples.append(('John lives in Seattle.', ('tell', Atom('Live_in', 'john', 'seattle'))))
    examples.append(('John lives in New York.', ('tell', Atom('Live_in', 'john', 'new_york'))))
    examples.append(('John does not live in New York.', ('tell', Not(Atom('Live_in', 'john', 'new_york')))))

    # Miscellaneous
    examples.append(('New York is a big city.', ('tell', And(Atom('Big', 'new_york'), Atom('City', 'new_york')))))
    examples.append(('If it is raining, it is wet.', ('tell', Implies(Atom('Rain'), Atom('Wet')))))

    # Coordination
    examples.append(('John and Bill are cats.', ('tell', And(Atom('Cat', 'john'), Atom('Cat', 'bill')))))
    examples.append(('John is either happy or sad.', ('tell', Xor(Atom('Happy', 'john'), Atom('Sad', 'john')))))
    examples.append(('John lives in Seattle or Portland.', ('tell', Or(Atom('Live_in', 'john', 'seattle'), Atom('Live_in', 'john', 'portland')))))
    examples.append(('Either it is raining or it is snowing.', ('tell', And(Or(Atom('Rain'), Atom('Snow')), Not(And(Atom('Rain'), Atom('Snow')))))))

    # Quantification
    examples.append(('Cats are animals.', ('tell', Forall('$x1', Implies(Atom('Cat', '$x1'), Atom('Animal', '$x1'))))))
    examples.append(('A cat is an animal.', ('tell', Forall('$x1', Implies(Atom('Cat', '$x1'), Atom('Animal', '$x1'))))))
    examples.append(('If a person lives in California, he is happy.', ('tell', Forall('$x1', Implies(And(Atom('Person', '$x1'), Atom('Live_in', '$x1', 'california')), Atom('Happy', '$x1'))))))
    examples.append(('John visited every city.', ('tell', Forall('$x1', Implies(Atom('City', '$x1'), Atom('Visit', 'john', '$x1'))))))
    examples.append(('Every city was visited by John.', ('tell', Forall('$x1', Implies(Atom('City', '$x1'), Atom('Visit_by', '$x1', 'john'))))))
    examples.append(('Every person likes some cat.', ('tell', Forall('$x1', Implies(Atom('Person', '$x1'), Exists('$x2', And(Atom('Cat', '$x2'), Atom('Like', '$x1', '$x2'))))))))
    examples.append(('No city is perfect.', ('tell', Not(Exists('$x1', And(Atom('City', '$x1'), Atom('Perfect', '$x1')))))))

    examples.append(('Does John live in Sweden?', ('ask', Atom('Live_in', 'john', 'sweden'))))

    ### Train the model using Perceptron

    print "============================================================"
    print "Training the grammar on %d examples" % len(examples)
    numUpdates = 0
    def updateWeights(deriv, incr):
        if deriv.rule:
            deriv.rule.score += incr
        for child in deriv.children:
            updateWeights(child, incr)

    # target, pred are both (mode, form)
    # Need to use unify because the variables could be different
    def isCompatible(target, pred):
        return target == pred
        #return target[0] == pred[0] and unify(target[1], pred[1], {})

    for iteration in range(0, 10):
        print '-- Iteration %d' % iteration 
        numMistakes = 0
        for x, y in examples:
            # Predict on example
            utterance = Utterance(x)
            derivations = parseUtterance(utterance, rules)
            targetDeriv = None
            for deriv in derivations:
                if isCompatible(y, deriv.form):
                    targetDeriv = deriv
                    break
            if targetDeriv == None:
                print "Impossible to get correct: %s => %s" % (x, rstr(y))
                print "  Utterance:", utterance
                print "  Derivations:"
                for deriv in derivations:
                    print '   ', rstr(deriv.form)
                continue
            predDeriv = derivations[0]

            if predDeriv != targetDeriv:
                print "Mistake: %s => %s, predicted %s" % (x, rstr(y), rstr(predDeriv.form))
                numMistakes += 1
                # Update weights
                numUpdates += 1
                stepSize = 1.0 # / math.sqrt(numUpdates)
                updateWeights(targetDeriv, +stepSize)
                updateWeights(predDeriv, -stepSize)
        if numMistakes == 0: break

    print 'Rules with non-zero weights:'
    for rule in rules:
        if rule.score != 0:
            print ' ', rule
