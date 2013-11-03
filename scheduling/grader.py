#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""


import random

import graderUtil
import util
import collections
import copy
grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Manual problems

# .addBasicPart is a basic test and always run.
grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf'))


############################################################
# Problem 1a: N-Queens

def test1a():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    grader.requireIsEqual(1.0, nQueensSolver.optimalWeight)
    grader.requireIsEqual(92, nQueensSolver.numOptimalAssignments)
    grader.requireIsEqual(2057, nQueensSolver.numOperations)

grader.addBasicPart('1a-0', test1a)

############################################################
# Problem 1c: Most constrained variable

def test1c():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    grader.requireIsEqual(1.0, mcvSolver.optimalWeight)
    grader.requireIsEqual(92, mcvSolver.numOptimalAssignments)
    grader.requireIsLessThan(1500, mcvSolver.numOperations)

grader.addBasicPart('1c-0', test1c)

############################################################
# Problem 1d: Least constraining value

def test1d():
    lcvSolver = submission.BacktrackingSearch()
    lcvSolver.solve(submission.create_nqueens_csp(8), lcv = True)
    grader.requireIsEqual(92, lcvSolver.numOptimalAssignments)
    grader.requireIsLessThan(60, lcvSolver.firstAssignmentNumOperations)

grader.addBasicPart('1d-0', test1d)

############################################################
# Problem 1e: Arc consistency

def test1e_0():
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(submission.create_nqueens_csp(8), mcv = True, lcv = True, mac = True)
    grader.requireIsEqual(92, acSolver.numOptimalAssignments)
    grader.requireIsLessThan(20, acSolver.firstAssignmentNumOperations)
    grader.requireIsLessThan(1000, acSolver.numOperations)

grader.addBasicPart('1e-0', test1e_0)




############################################################
# Problem 2a: Conditional OR potential

def test2a_0():
    csp = util.CSP()
    csp.add_variable('A', ['a', 'b', 'c'])
    csp.add_variable('B', ['b', 'c'])

    # Create the or constraint but not enforcing it.
    var = submission.get_or_variable(csp, 'at-least-one-b', ['A', 'B'], 'b')
    orSolver = submission.BacktrackingSearch()
    orSolver.solve(csp)
    grader.requireIsEqual(6, orSolver.numOptimalAssignments)

    # Enforce the or constraint.
    csp.add_unary_potential(var, lambda val: val)
    orSolver = submission.BacktrackingSearch()
    orSolver.solve(csp)
    grader.requireIsEqual(4, orSolver.numOptimalAssignments)

grader.addBasicPart('2a-0', test2a_0)

############################################################
# Problem 2b: Sum potential

def test2b_0():
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2, 3])
    csp.add_variable('B', [0, 6, 7])
    csp.add_variable('C', [0, 5])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-15', ['A', 'B', 'C'], 15)
    csp.add_unary_potential(sumVar, lambda n: n in [12, 13])
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(4, sumSolver.numOptimalAssignments)

    csp.add_unary_potential(sumVar, lambda n: n == 12)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(2, sumSolver.numOptimalAssignments)

grader.addBasicPart('2b-0', test2b_0)


def verify_schedule(bulletin, profile, schedule, checkUnits = True):
    """
    Returns true if the schedule satisifies all requirements given by the profile.
    """
    goodSchedule = True
    all_courses_taking = dict((s[1], s[0]) for s in schedule)

    # No course can be taken twice.
    goodSchedule *= len(all_courses_taking) == len(schedule)
    if not goodSchedule:
        print 'course repeated'
        return False

    # Each course must be offered in that quarter.
    goodSchedule *= all(bulletin.courses[s[1]].is_offered_in(s[0]) for s in schedule)
    if not goodSchedule:
        print 'course not offered'
        return False

    # If specified, only take the course at the requested time.
    for req in profile.requests:
        if len(req.quarters) == 0: continue
        goodSchedule *= all([s[0] in req.quarters for s in schedule if s[1] in req.cids])
    if not goodSchedule:
        print 'course taken at wrong time'
        return False

    # If a request has multiple courses, at most one is chosen.
    for req in profile.requests:
        if len(req.cids) == 1: continue
        goodSchedule *= len([s for s in schedule if s[1] in req.cids]) <= 1
    if not goodSchedule:
        print 'more than one exclusive group of courses is taken'
        return False

    # Must take a course after the prereqs
    for req in profile.requests:
        if len(req.prereqs) == 0: continue
        cids = [s for s in schedule if s[1] in req.cids] # either empty or 1 element
        if len(cids) == 0: continue
        quarter, cid, units = cids[0]
        for prereq in req.prereqs:
            if prereq in profile.taking:
                goodSchedule *= prereq in all_courses_taking
                if not goodSchedule:
                    print 'not all prereqs are taken'
                    return False
                goodSchedule *= profile.quarters.index(quarter) > \
                    profile.quarters.index(all_courses_taking[prereq])
    if not goodSchedule:
        print 'course is taken before prereq'
        return False

    if not checkUnits: return goodSchedule
    # Check for unit loads
    unitCounters = collections.Counter()
    for quarter, c, units in schedule:
        unitCounters[quarter] += units
    goodSchedule *= all(profile.minUnits <= u and u <= profile.maxUnits \
        for k, u in unitCounters.items())
    if not goodSchedule:
        print 'unit count out of bound for quarter'
        return False

    return goodSchedule

# Load all courses.
bulletin = util.CourseBulletin('courses.json')

############################################################
# Problem 3a: Quarter specification

def test3a_0():
    profile = util.Profile(bulletin, 'profile3a.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(3, alg.numOptimalAssignments)
    solution = util.extract_course_scheduling_solution(profile, alg.optimalAssignment)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution, False))

grader.addBasicPart('3a-0', test3a_0)

############################################################
# Problem 3b: Weighting

def test3b_0():
    profile = util.Profile(bulletin, 'profile3b.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_request_weights(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(2, alg.numOptimalAssignments)
    grader.requireIsEqual(3, alg.numAssignments)
    grader.requireIsEqual(5, alg.optimalWeight)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution, False))

grader.addBasicPart('3b-0', test3b_0)

############################################################
# Problem 3c: Prerequisites

def test3c_0():
    profile = util.Profile(bulletin, 'profile3c.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_prereq_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(6, alg.numOptimalAssignments)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution, False))

grader.addBasicPart('3c-0', test3c_0)

############################################################
# Problem 3d: Unit load

def test3d_0():
    profile = util.Profile(bulletin, 'profile3d.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_unit_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(15, alg.numOptimalAssignments)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution))

grader.addBasicPart('3d-0', test3d_0)

############################################################
# Check that profile.txt is valid.

def valid_profile_txt():
    try:
        profile = util.Profile(bulletin, 'profile.txt')
    except:
        print 'profile.txt is not valid'
        grader.fail()
    grader.assignFullCredit()

grader.addBasicPart('profile.txt', valid_profile_txt)

grader.grade()
