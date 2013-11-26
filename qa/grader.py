#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""

from logic import *

import pickle, gzip
import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')

# name: name of this formula (used to load the models)
# predForm: the formula predicted in the submission
# preconditionForm: only consider models such that preconditionForm is true
def checkFormula(name, predForm, preconditionForm=None):
    objects, targetModels = pickle.load(gzip.open(name + '.pklz'))
    # If preconditionion exists, change the formula to
    preconditionPredForm = Implies(preconditionForm, predForm) if preconditionForm else predForm
    predModels = performModelChecking([preconditionPredForm], findAll=True, objects=objects)
    ok = True
    def hashkey(model): return tuple(sorted(str(atom) for atom in model))
    targetModelSet = set(hashkey(model) for model in targetModels)
    predModelSet = set(hashkey(model) for model in predModels)
    for model in targetModels:
        if hashkey(model) not in predModelSet:
            grader.fail("Your formula (%s) says the following model is FALSE, but it should be TRUE:" % predForm)
            ok = False
            printModel(model)
            return
    for model in predModels:
        if hashkey(model) not in targetModelSet:
            grader.fail("Your formula (%s) says the following model is TRUE, but it should be FALSE:" % predForm)
            ok = False
            printModel(model)
            return
    grader.addMessage('You matched the %d models' % len(targetModels))
    grader.addMessage('Example model: %s' % rstr(targetModels[0]))
    grader.assignFullCredit()

# name: name of this formula set (used to load the models)
# predForms: formulas predicted in the submission
# predQuery: query formula predicted in the submission
def addParts(name, numForms, predictionFunc):
    # part is either an individual formula (0:numForms), all (combine everything)
    def check(part):
        predForms, predQuery = predictionFunc()
        if len(predForms) < numForms:
            grader.fail("Wanted %d formulas, but got %d formulas:" % (numForms, len(predForms)))
            for form in predForms: print '-', form
            return
        if part == 'all':
            checkFormula(name + '-all', AndList(predForms))
        elif part == 'run':
            # Actually run it on a knowledge base
            #kb = createResolutionKB()  # Too slow!
            kb = createModelCheckingKB()

            # Need to tell the KB about the objects to do model checking
            objects, targetModels = pickle.load(gzip.open(name + '-all.pklz'))
            for obj in objects:
                kb.tell(Atom('Object', obj))

            # Add the formulas
            for predForm in predForms:
                response = kb.tell(predForm)
                showKBResponse(response)
                grader.requireIsEqual(CONTINGENT, response.status)
            response = kb.ask(predQuery)
            showKBResponse(response)

        else:  # Check the part-th formula
            checkFormula(name + '-' + str(part), predForms[part])

    def createCheck(part): return lambda : check(part)  # To create closure

    for part in range(numForms) + ['all', 'run']:
        grader.addBasicPart(name + '-' + str(part), createCheck(part), maxPoints=1, maxSeconds=10000)

############################################################
# Problem 1: propositional logic

grader.addBasicPart('formula1a', lambda : checkFormula('formula1a', submission.formula1a()), 1)
grader.addBasicPart('formula1b', lambda : checkFormula('formula1b', submission.formula1b()), 1)
grader.addBasicPart('formula1c', lambda : checkFormula('formula1c', submission.formula1c()), 1)
grader.addBasicPart('formula1d', lambda : checkFormula('formula1d', submission.formula1d()), 1)

############################################################
# Problem 2: first-order logic

formula2a_precondition = AntiReflexive('Mother')
formula2b_precondition = AntiReflexive('Child')
formula2c_precondition = AntiReflexive('Child')
formula2d_precondition = AntiReflexive('Parent')
formula2e_precondition = AntiReflexive('Parent')
grader.addBasicPart('formula2a', lambda : checkFormula('formula2a', submission.formula2a(), formula2a_precondition), 1)
grader.addBasicPart('formula2b', lambda : checkFormula('formula2b', submission.formula2b(), formula2b_precondition), 1)
grader.addBasicPart('formula2c', lambda : checkFormula('formula2c', submission.formula2c(), formula2c_precondition), 1)
grader.addBasicPart('formula2d', lambda : checkFormula('formula2d', submission.formula2d(), formula2d_precondition), 1)
grader.addBasicPart('formula2e', lambda : checkFormula('formula2e', submission.formula2e(), formula2e_precondition), 1)

############################################################
# Problem 3: liar puzzle

# Add liar-[0-5], liar-all, liar-run
addParts('liar', 6, submission.liar)

############################################################
# Problem 4: Modus Ponens

def testModusPonens1():
    A = Atom('A')
    A1 = Atom('A1')
    A2 = Atom('A2')
    A3 = Atom('A3')
    B = Atom('B')
    C = Atom('C')
    rule = submission.ModusPonensRule()  # Load submission
    grader.requireIsEqual(rstr([C]), rstr(rule.applyRule(Implies(A, C), A)))
    grader.requireIsEqual(rstr([Implies(And(A1, A3), C)]), rstr(rule.applyRule(Implies(And(And(A1, A2), A3), C), A2)))
    grader.requireIsEqual(rstr([]), rstr(rule.applyRule(Implies(A, C), B)))

grader.addBasicPart('4-1', testModusPonens1, 10)

grader.addManualPart('4-2', 5)

grader.addManualPart('4-3', 5)

############################################################
# Problem 5: odd and even integers

# Add ints-[0-5], ints-all, ints-run
addParts('ints', 6, submission.ints)

grader.addManualPart('5-2', 5)

grader.grade()
