#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""


import random
import numpy as np

import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Manual problems

# .addBasicPart is a basic test and always run. 
grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf')) 


############################################################
# Problem 2: runKMeans

# Use lambda for short tests
def testKMeansBasic():
    np.random.seed(5)
    x = np.array([[1., 1., 2.],[-1., -1., 2.],[-2., -1., 1.]])
    ans1 = np.array([[2., 1.],[ 2., -1.],[1., -1.5]])
    ans2 = submission.runKMeans(2,x,5)
    grader.requireIsEqual(ans1, ans2)
grader.addPart('2-0', testKMeansBasic, 5)



############################################################
# Problem 3: extractFeatures

patches = np.array([[1.,1.5,3.,-2.5],[2.,2.5,1.,-1.],[3.,3.5,1.,4.]])
centroids = np.array([[2.0,2.3,2.5],[-1.0,-2.5,1.0],[1.5,-3.,-1.5]])
features = np.array([[1.81983758, 0., 0.47215773],[1.81019111, 0., 0.5571374],
       [1.10930137, 0., 0.85107947],[2.0069789, 0., 0.]])
grader.addBasicPart('3-0', lambda : grader.requireIsEqual(features,submission.extractFeatures(patches,centroids)),3)


############################################################
# Problem 4: Supervised Training


theta = np.array([1.5,2.5])
fv1 = np.array([1.0,1.0])
fv2 = np.array([0.2,0.2])
grader.addBasicPart('4b-0', lambda : grader.requireIsEqual(np.array([-0.01798621, -0.01798621]), submission.logisticGradient(theta,fv1, 1)),1)
grader.addBasicPart('4b-1', lambda : grader.requireIsEqual(np.array([-0.0620051, -0.0620051]), submission.logisticGradient(theta,fv2, 1)),1)
grader.addBasicPart('4b-2', lambda : grader.requireIsEqual(np.array([0.1379949, 0.1379949]), submission.logisticGradient(theta,fv2, 0)),1)



theta = np.array([1.5,2.5])
fv1 = np.array([1.0,1.0])
fv2 = np.array([0.2,0.2])
grader.addBasicPart('4c-0', lambda : grader.requireIsEqual(np.array([0.,0.]), submission.hingeLossGradient(theta,fv1, 1)),1)
grader.addBasicPart('4c-1', lambda : grader.requireIsEqual(-fv2, submission.hingeLossGradient(theta,fv2, 1)),1)
grader.addBasicPart('4c-2', lambda : grader.requireIsEqual(fv2, submission.hingeLossGradient(theta,fv2, 0)),1)


############################################################
# Problem 5 : Extra credit (awarded to top 3 submissions)

grader.grade()
