import collections, sys, os
from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a():
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return Implies(And(Summer, California), Not(Rain))
    # END_YOUR_CODE

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b():
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return Equiv(Wet, Or(Rain, Sprinklers))
    # END_YOUR_CODE

# Sentence: "It only snows in the summer."
# (Note: sentences can be true or false, so don't be troubled.)
def formula1c():
    # Predicates to use:
    Snow = Atom('Snow')            # whether it's snowing
    Summer = Atom('Summer')        # whether it's summer
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return Implies(Snow, Summer)
    # END_YOUR_CODE

# Sentence: "Either it's day or night (but not both)."
def formula1d():
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return And(Or(Day, Night), Not(And(Day, Night)))
    # END_YOUR_CODE

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a mother."
def formula2a():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Mother(x, y): return Atom('Mother', x, y)  # whether x's mother is y
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return Forall(Variable('$x'), Implies(Person('$x'), Exists(Variable('$y'), Mother('$x', '$y'))))
    # END_YOUR_CODE

# Sentence: "At least one person has no children."
def formula2b():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return Exists(Variable('$x'), And(Person('$x'), Forall(Variable('$y'), Not(Child('$x', '$y')))))
    # END_YOUR_CODE

# Return a formula which defines Daughter in terms of Female and Child.
# See parentChild() in examples.py for a relevant example.
def formula2c():
    # Predicates to use:
    def Female(x): return Atom('Female', x)            # whether x is female
    def Child(x, y): return Atom('Child', x, y)        # whether x has a child y
    def Daughter(x, y): return Atom('Daughter', x, y)  # whether x has a daughter y
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    return Forall('$x', Forall('$y', Equiv(And(Child('$x', '$y'), Female('$y')), Daughter('$x', '$y'))))
    # END_YOUR_CODE

# Return a formula which defines Grandmother in terms of Female and Parent.
# Note: It is ok for a person to be her own parent
def formula2d():
    # Predicates to use:
    def Female(x): return Atom('Female', x)                  # whether x is female
    def Parent(x, y): return Atom('Parent', x, y)            # whether x has a parent y
    def Grandmother(x, y): return Atom('Grandmother', x, y)  # whether x has a grandmother y
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    def CheckGrandmother(x, z):
        return  Equiv(
                    And(
                        Exists('$y', And(
                            Parent(x, '$y'), 
                            Parent('$y', z)
                        )),
                    Female(z)
                    ),
                Grandmother(x, z)
                )
        
    return Forall('$x', 
                 Forall('$z',
                        CheckGrandmother('$x', '$z')
                        )  
                  )
                                
    # END_YOUR_CODE

# Return a formula which defines Ancestor in terms of Parent.
# Hint: think recursively.
def formula2e():
    # Predicates to use:
    def Parent(x, y): return Atom('Parent', x, y)      # whether x has a parent y
    def Ancestor(x, y): return Atom('Ancestor', x, y)  # whether x has an ancestor y
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    def CheckAncestor(x, y):
        return Or(
                  Parent(x, y),
                  Exists('$z', 
                         And(
                             And(
                                 Not(Equals(x, '$z')),
                                 Parent(x, '$z')
                                 ),
                             CheckAncestor('$z', y)
                             )
                         )
                  )
                  
    return Forall('$x', 
                  Forall('$y', 
                      Equiv(
                            CheckAncestor('$x', '$y'), 
                            Ancestor('$x', '$y')
                            )
                      )
                  )
    # END_YOUR_CODE

############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. John: "It wasn't me!"
# 1. Susan: "It was Nicole!"
# 2. Mark: "No, it was Susan!"
# 3. Nicole: "Susan's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts.
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False if x is not equal to y.
def liar():
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    john = Constant('john')
    susan = Constant('susan')
    nicole = Constant('nicole')
    mark = Constant('mark')
    formulas = []
    # We provide the formula for fact 0 here.  
    formulas.append(Equiv(TellTruth(john), Not(CrashedServer(john))))
    # You should add 5 formulas, one for each of facts 1-5.
    # BEGIN_YOUR_CODE (around 11 lines of code expected)
    formulas.append(Equiv(TellTruth(susan), CrashedServer(nicole)))
    formulas.append(Equiv(TellTruth(mark), CrashedServer(susan)))
    formulas.append(Equiv(TellTruth(nicole), Not(TellTruth(susan))))
    formulas.append(And(Forall('$x', Forall('$y', Implies(Not(Equals('$x', '$y')), Not(And(TellTruth('$x'), TellTruth('$y')))))), Exists('$z', TellTruth('$z'))))
    formulas.append(And(Forall('$x', Forall('$y', Implies(Not(Equals('$x', '$y')), Not(And(CrashedServer('$x'), CrashedServer('$y')))))), Exists('$z', CrashedServer('$z'))))
    # END_YOUR_CODE
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 4: Modus Ponens inference.

# Implies(A,B), A |- B
class ModusPonensRule(BinaryRule):
    # form1: the implication rule (antecedent => consequent)
    # form2: the argument to be matched on the antecedent
    # return: list of formulas resulting from the application of the Modus
    # Ponens rule.  If Modus Ponens does not apply, return [].
    #
    # Example:
    # - form1 = Implies(A, C)
    # - form2 = A
    # - return [C]
    # Example (antecedent is conjunction of things)
    # - form1 = Implies(And(And(A1, A2), A3), C)
    # - form2 = A2
    # - return [Implies(And(A1, A3), C)]
    # Example (rule does not apply)
    # - form1 = Implies(A, C)
    # - form2 = B
    # - return []
    #
    # Tips:
    # - Extract antecedent and consequent:
    #   If x = Implies(A, C), x.arg1 = A, x.arg2 = C
    # - Get list of conjuncts from a formula:
    #   flattenAnd(And(And(A, B), C)) = [A, B, C]
    # - Construct a formula from a list of conjuncts:
    #   AndList([A, B, C]) = And(And(A, B), C)
    #   AndList([]) = AtomTrue
    # - You can use == to test equality of atoms
    # - You should probably take a look at the implementation of AndList.  In
    #   particular, note what happens if you pass in an empty list to AndList.
    # - Think carefully about how to check if the left side of an implication
    #   is empty.  In particular, a conjunction of AtomTrue atoms is
    #   equivalent to an empty formula.  For example, you should never return
    #   something like
    #   Implies(And(AtomTrue, AtomTrue), Foo)
    #   This should be returned as Foo.
    def applyRule(self, form1, form2):
        if not form1.isa(Implies): return []  # Rule does not apply
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        flatForm1 = flattenAnd(form1.arg1)
        flatForm2 = flattenAnd(form2)
        for arg in flatForm2:
            if arg in flatForm1:
                flatForm1.remove(arg)
        if len(flatForm1) == 0:
            return [form1.arg2]
        if len(flatForm1) == len(flattenAnd(form1.arg1)):
            return []
        return [Implies(AndList(flatForm1), form1.arg2)]
        # END_YOUR_CODE

############################################################
# Problem 5: Odd and even integers

# Return the following 6 laws:
# 0. Each number $x$ has a unique successor that is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints():
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y
    # Note: all objects are numbers, so we don't need to define Number as an
    # explicit predicate.
    # Note: pay attention to the order of arguments of Successor and Larger.
    # Populate |formulas| with the 6 laws above and set |query| to be the
    # query.
    # Hint: You might want to use the Equals predicate, defined in logic.py.  This
    # predicate is used to assert that two objects are the same.
    formulas = []
    query = None
    # BEGIN_YOUR_CODE (around 23 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
    # For part (b), your job is to show that adding the following formula
    # would result in a contradiction for finite domains.
    #formulas.append(Forall('$x', Not(Larger('$x', '$x'))))
    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)
