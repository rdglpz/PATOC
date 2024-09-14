#!/usr/bin/env python3

#Object Oriented implementation for computing the
#TOC= Total Operating Characteristic Curve, functions for probabilistic analysis
#Author: S. Ivvan Valdez
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.
import numpy as np
import tensorflow as tf
import math
import scipy.linalg as la
import copy
import gc
import rasterio
from rasterio.transform import from_origin

#import annfit as af
#from rasterio import CRS


#Object to store a TOC curve
class PATOC:
    """
    This class implements the Total Operating Characteristic Curve computation and analysis using probability functions. That is to say, a cumulative distribution is derived from the TOC, then, a mass or density probability function is computed from it.

    :param rank: The class is instantiated with the optional parameter ``rank`` that is a numpy array of a predicting feature.

    :param groundtruth: The class is instantiated with the optional parameter ``groundtruth`` that is a numpy array of binary labels (0,1).

    :cvar kind: A string with the value 'None' by default, it indicates the kind of TOC, for instance: continuous, discrete, semicontinuous, or None for an empty TOC. A semicontinuous TOC is that with continuous disjoint segments.

    :cvar area: The area under the curve of the TOC, in discrete cases it is the sum of heights. Notice that this area is not normalized neither is the rate of the parallelepiped.

    :cvar areaRatio: The areaRation with respect to the parallelepiped area.

    :cvar isorted: All the variables of indices start with an "i", they could be either a true-false array (true in the selected indices), or an integer array. In this case this array stores the indices of the sorted rank, they are sorted from the minimum to the maximum.

    :cvar ndata: All the sizing variables start with "n", in this case, ndata is the total number of data, that is to say the "rank" and "groundtruth" sizes

    :return: The class instance, if ``rank`` and ``groundtruth`` are given it computes the TOC, otherwise it is an empty class.

    :rtype: ``PATOC``

    """


    area=0
    """
    _`area`=     under the curve of the TOC substracting the right triangle of the paralellogram.

    """
    Darea=0
    """
    _`Darea`, since te TOC is approximated via a dicrete curve, this the approximation of the `area`_ (area under the TOC minus the right parellelogram triangle area), it is useful to determine whether the approximation is sufficiently close to the actual value.

    """

    areaRatio=0
    """
    _`areaRatio` with respect to the parallelepiped. The maximum TOC area is that of the parallelepiped, hence this is the ratio of the TOC area inside the parallelepiped divided by the parallelepiped area, hence its value is between 0 and 1.
    Notice that the area inside the parallelepiped is usually different from the area under the curve, it is computed substracting from the ``area`` variable the triangle in the left side of the parallelepiped.

    """
    areaDRatio=0
    """
    _`areaRatio` with respect to the parallelepiped. The maximum TOC area is that of the parallelepiped, hence this is the ratio of the TOC area inside the parallelepiped divided by the parallelepiped area, hence its value is between 0 and 1.
    Notice that the area inside the parallelepiped is usually different from the area under the curve, it is computed substracting from the ``area`` variable the triangle in the left side of the parallelepiped_`areaRatio` with respect to the parallelepiped. The maximum TOC area is that of the parallelepiped, hence this is the ratio of the TOC area inside the parallelepiped divided by the parallelepiped area, hence its value is between 0 and 1.
    Notice that the area inside the parallelepiped is usually different from the area under the curve, it is computed substracting from the ``area`` variable the triangle in the left side of the parallelepiped

    """


    kind='None'
    """
        The _`kind` attribute indicates the type of TOC curve, that is to say: "continuous","discrete","semicontinuous","forcedContinuous". The continuous is a continuous approximation of the Hits and Hits plus False Alarms, while the discrete is computed when discrete rank values are detected, hence a kind of cummulative histogram is computed, and the cumulative probability function is a cummulative histogram of frequencies, and the probability function is discrete as well. In the "semicontinuous case, it could have segments of contiinuous domains and discrete points, most of the time if ``forceContinuity``_ is ``True``, a semincotinuous TOC is converted to "forcedContinuous", that is to say the discrete approximation is continuous, but it is a repaired semicontinuous curve.
    """

    isorted=None
    """
       The _`isorted` attibute stores the indices fo the sorted rank. Storing the indices is useful to save computational time when converting a rank array to a probability value.

    """

    ndata=0
    """
        _`ndata` stores the number of data in the arrays of the class,  notice that the number of positives (``np``) or other counts are altered when interpolations, normalization or vectorization of the TOC is applied, while ``ndata`` stores the length of the data arrays independent of the mentioned counts.

    """

    np=0
    """
    _`np` stores the number of positive data (1-labeled data, or data with presence of a characteristic).

    """

    PDataProp=0
    """
    _`PDataProp` is the proportion of positive (1-labeled)data in the data. The purpose is to maintain the proportion of class 1 data in TOC operations, hence to preserve the knowlkedege about data imbalance and proportion of classes even if the TOC is normalized.

    """


    HpFA=None
    """
    _`HpFA` is a numpy array with the sum of Hits plus False Alarms.

    """

    Hits=None
    """
    _`Hits` is a numpy array with the sum of true positives

    """

    Thresholds=None
    """
    _`Thresholds` is a numpy array with thresholds of the TOC, they are computed using the ranks, that is to say, most of the times they are equal to the ranks.

    """

    npiecewise=0
    """
    _`npiecewise` is a number of segments lower or equal to the number of data, it is used to compute a TOC approximation for a better visualization fo the TOC changes with respect to the rank. If continuous the piecewise TOC is used for deriving a density probability function.
    """

    pwHits=None
    """
    _`pwHits` in a continuous TOC it stores an array of size `npiecewise`_ that represents the TOC curve. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these represantations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothHits`_ and `smoothHpFA`_ representation.
    In a piecewise TOC this array is the same that Hits and stores all the discrete coordinates of the TOC.
    """

    pwHpFA=None
    """
    _`pwHpFA` in a continuous TOC it stores an array of size `npiecewise`_ that represents the TOC curve. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these represantations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothHits`_ and `smoothHpFA`_ representation.
    In a piecewise TOC this array is the same that HpFA and stores all the discrete coordinates of the TOC.
    """

    pwLabels=None
    """
    _`pwLabels` in a continuous TOC it stores an array of size `npiecewise`_ that represents proportion of 1-labeled data for each threshold of the rank (feature) values. In a piecewise TOC is the same proportion but it is computed on all the data and it is not a discretized approximation.
    """


    pwRank=None
    """
    _`pwRank` in a continuous TOC it stores an array of size `npiecewise`_ that represents the TOC curve in the rank/threshold domain. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ , `pwHpFA`_ `pwRank`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these represantations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothHits`_ and `smoothHpFA`_ representation.
    In a piecewise TOC this array is the same that HpFA and stores all the discrete coordinates of the TOC.
    """

    pwndata=None
    """
    _`pwndata` stores the number of data used in the TOC discretization by each discrete point. It is of special interest for piecewise TOC, becaue it represents the frequency of each discrete bin.
    """

    CPF_Rank_given_Presence=None
    """
    _`CPF_Rank_given_Presence` is the conditional cumulative distribution function. CPF_Rank_given_Presence=P(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """

    PF_Rank_given_Presence=None
    """
    _`PF_Rank_given_Presence` is the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    DPF_Rank_given_Presence=None
    """
    _`DPF_Rank_given_Presence` is the firsat derivative of the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    smoothPF_Rank_given_Presence=None
    """
    _`smoothPF_Rank_given_Presence` in a continuous TOC it stores an array of size `npiecewise`_ that represents a ``smoothed`` or ``regularized`` derivative of the cummulatve (conditional to presence) distribution function. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these representations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothPF_Rank_given_Presence`_ representation, that is intended for a noise-free visualization, obviously there is a payoff of regularizing the TOC, hence we suggest to prefer `pwHits`_ and `pwHpFA`_ for computations unless  you are aware of the information lost.
    In a piecewise TOC it stores a histogram.
    """
    smoothDPF_Rank_given_Presence=None
    """
    _`smoothDPF_Rank_given_Presence` is te first derivative of `smoothPF_Rank_given_Presence`_.
    """

    CPF_Rank=None
    """
    _`CPF` is the conditional cumulative distribution function. CPF=PF(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """

    PF_Rank=None
    """
    _`PF_RankPF_Rank` is the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    DPF_Rank=None
    """
    _`DPF_Rank` is the firsat derivative of the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    smoothPF_Rank=None
    """
    _`smoothPF_Rank` in a continuous TOC it stores an array of size `npiecewise`_ that represents a ``smoothed`` or ``regularized`` derivative of the cummulatve (conditional to presence) distribution function. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these representations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothPF`_ representation, that is intended for a noise-free visualization, obviously there is a payoff of regularizing the TOC, hence we suggest to prefer `pwHits`_ and `pwHpFA`_ for computations unless  you are aware of the information lost.
    In a piecewise TOC it stores a histogram.
    """


    smoothDPF_Rank=None
    """
    _`smoothDPF_Rank` is te first derivative of `smoothPF_Rank`_.
    """

    JPF=None
    """
    _`JPF` in a continuous TOC it stores an array of size `npiecewise`_ that represents the join probability function (conditional to presence) distribution function. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these representations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothPF`_ representation, that is intended for a noise-free visualization, obviously there is a payoff of regularizing the TOC, hence we suggest to prefer `pwHits`_ and `pwHpFA`_ for computations unless  you are aware of the information lost.
    In a piecewise TOC it stores a histogram.
    """

    smoothJPF=None
    """
    _`smoothJPF` in a continuous TOC it stores an array of size `npiecewise`_ that represents a ``smoothed`` or ``regularized`` derivative of the cummulatve (conditional to presence) distribution function. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these representations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothPF`_ representation, that is intended for a noise-free visualization, obviously there is a payoff of regularizing the TOC, hence we suggest to prefer `pwHits`_ and `pwHpFA`_ for computations unless  you are aware of the information lost.
    In a piecewise TOC it stores a histogram.
    """


    CPF_Presence_given_Rank=None
    """
    _`CPF` is the conditional cumulative distribution function. CPF=PF(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """

    PF_Presence_given_Rank=None
    """
    _`PF` is the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    DPF_Presence_given_Rank=None
    """
    _`DPF` is the firsat derivative of the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `npiecewise`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    smoothPF_Presence_given_Rank=None
    """
    _`smoothPF` in a continuous TOC it stores an array of size `npiecewise`_ that represents a ``smoothed`` or ``regularized`` derivative of the cummulatve (conditional to presence) distribution function. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `pwHits`_ and `pwHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these representations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothPF`_ representation, that is intended for a noise-free visualization, obviously there is a payoff of regularizing the TOC, hence we suggest to prefer `pwHits`_ and `pwHpFA`_ for computations unless  you are aware of the information lost.
    In a piecewise TOC it stores a histogram.
    """
    smoothDPF_Presence_given_Rank=None
    """
    _`smoothDPF` is te first derivative of `smoothPF`_.
    """

    smoothingFactor=1
    """
    _`smoothingFactor` in a continuous TOC it is used in the `PFsmooth`_ method for the RLS method the greather smoothing factor the less smoothing.
    """

    icontinuous=None
    """
    _`icontinuous` is an array of size `npiecewise`_ with 1 in the continuous segments and 0 in the discrete and -1 in the discontinuos (the segment is not in the domain of the rank).
    """

    iUnique=None
    """
    _`iUnique` is an array of size `ndata`_ with True in the last unique elements of the sorted rank and False otherwise. That is to say, if the sorted rank is  [0.3,0.3,0.3,0.5,0.7,1,1], iUnique is [False,False,True,True,True,False,True].
    """
    featureName='X'
    """
    _`featureName` is the name of the feature to analyze for plotting purposes, it is 'X' by default.
    """
    boostrapFlag=False
    CImin=0.05
    CImax=0.95

    def __init__(self,rank=[], groundtruth=[], npiecewise=-1, forceContinuity=True,smoothingMethod='ANN'):
        """
        _`__init__`
        Constructor of the TOC. Here the hits and false alarms are computed, as well as the kind (discrete, continuous, semicontinuous), the `Area`_  and `areaRatio`_ according to the definition in the documentation.
        """

        #validating rank, groundtruth pairs. They can never be 0 and unequal
        if (len(rank)!=0 and len(groundtruth)!=0 and len(rank)==len(groundtruth)):
            self.maxr=np.max(rank)
            """
            _`maxr` is the maximal rank value, it is a member of the PATOC object.
            """
            self.maxgt=np.max(groundtruth)
            """
            _`maxgt` is the maximal (groundtruth) label value, it is usuaaly 1, it is a member of the PATOC object.
            """
            self.minr=np.min(rank)
            
            """
            _`minr` is the minimal rank value, it is a member of the PATOC object.
            """
            
            self.mingt=np.min(groundtruth)
            """
            _`mingt` is the minimal (groundtruth) label value, it is usuaaly 0, it is a member of the PATOC object.
            """
            
            self.Rank=rank
            """
            stores the rank values.
            """
            self.forceContinuity=forceContinuity
            self.groundtruth=groundtruth
            #Sorting the classification rank and getting the indices
            self.isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
            #Data size, this is the total number of samples
            self.ndata=len(rank)
            #This is the number of class 1 in the input data
            self.np=sum(groundtruth==1)
            #Hits plus false alarms
            self.HpFA=np.array(range(self.ndata))+1
            #True positives
            self.Hits=np.cumsum(groundtruth[self.isorted])
            #Thresholds
            self.Thresholds=rank[self.isorted]
            #Proportion of positives and data (positive class proportion)
            self.PDataProp=self.np/self.ndata
            #Detecting unique and repeated ranks/thresholds
            self.iUnique=np.append(~((self.Thresholds[:-1]-self.Thresholds[1:])==0),True)
            self.sumIUnique=np.sum(self.iUnique)
            #Computing the discretization for the TOC representation
            if (npiecewise<0):
                self.discretization()
            else:
                self.npiecewise=npiecewise
            if (forceContinuity):
                self.npiecewise=min(self.npiecewise,np.sum(self.iUnique))
            if (self.kind=='discrete' and smoothingMethod=='ANN'):
                smoothingMethod='wmeans'
            #Detecting continuous or discontinuous TOC.
            self.areaComputation()
            self.computePF()

            # default parameters def PFsmoothing(self, method='RLS', PFsmoothingFactor=-1, 
            # DPFsmoothingFactor=-1, CDFsmoothingFactor=-1, pwHitssmoothingFactor=-1 ):
            self.PFsmoothing(smoothingMethod)
    pass



########################################Class definition ends########################################################


########################################BEGIN METHOD discretization##################################################

def discretization(self):
    """
    _`discretization` computes the number of segments to partition the rank domain, then it is used to determine whether the function is continuous or discontinuous.

    """
    self.npiecewise=min(self.sumIUnique,min(min(int(self.ndata),10000),max(int(self.ndata/30),1000)))
    stopCond=int(min(self.sumIUnique,self.npiecewise,self.ndata/self.npiecewise))+1
    nCSegments=self.continuity()
    ite=0
    dfactor=0.5
    while(nCSegments<self.npiecewise and (self.npiecewise>stopCond and ite<25)):
        ite+=1
        self.npiecewise=int(self.npiecewise*(1-dfactor)+1)
        dfactor=max(0.1,dfactor/2)
        nCSegments=self.continuity()
    if (self.npiecewise<=2):
        print('Can not find a continuous dicretization!!')
    else:
        self.areaDComputation()



########################################END METHOD discretization####################################################



########################################BEGIN METHOD continuity######################################################

def continuity(self):
    """
    _`continuity` computes the continuous, and discontinuous segments of the TOC.

    """
    thetaInf=self.minr-(0.5/self.ndata)*(self.maxr-self.minr)
    thetaSup=self.maxr+(0.5/self.ndata)*(self.maxr-self.minr)
    deltar=(thetaSup-thetaInf)/(self.npiecewise)
    self.pwThresholds=(np.array(range(self.npiecewise))+1)*deltar+thetaInf
    j=0
    meanr=0
    sumHits=0
    lastSumHits=0
    lastSumHpFA=0
    sumHpFA=0
    nmean=0
    thetaSup=thetaInf+deltar
    self.icontinuous=np.zeros(self.npiecewise)
    self.pwRank=np.zeros(self.npiecewise)
    self.pwHits=np.zeros(self.npiecewise)
    self.pwHpFA=np.zeros(self.npiecewise)
    self.pwndata=np.zeros(self.npiecewise)
    self.icontinuous[0]=1
    for  i in range(self.ndata):
        testrank=self.Rank[self.isorted[i]]
        while (testrank>=thetaSup):
            if (nmean>=1):
                self.icontinuous[j]=1
                self.pwRank[j]=meanr/nmean
                if (sumHits==0):
                    self.pwHits[j]=lastSumHits
                    self.pwHpFA[j]=lastSumHpFA
                self.pwHits[j]=sumHits/nmean
                self.pwHpFA[j]=sumHpFA/nmean
                self.pwndata[j]=nmean
                lastSumHits=self.pwHits[j]
                lastSumHpFA=self.pwHpFA[j]
            else:
                if (self.forceContinuity):
                    self.pwRank[j]=(thetaInf+thetaSup)/2
                    self.pwHits[j]=lastSumHits
                    self.pwHpFA[j]=lastSumHpFA
                else:
                    self.pwRank[j]=float('nan')
                    self.pwHits[j]=float('nan')
                    self.pwHpFA[j]=float('nan')
            j+=1
            nmean=0
            meanr=0
            sumHits=0
            sumHpFA=0
            thetaInf=thetaSup
            thetaSup=thetaInf+deltar
        if (testrank>=thetaInf and testrank<thetaSup):
            meanr+=testrank
            nmean+=1
            sumHits+=self.Hits[i]
            sumHpFA+=self.HpFA[i]
    if (nmean>=1):
        self.icontinuous[j]=1
        self.pwRank[j]=meanr/nmean
        if (sumHits==0):
            self.pwHits[j]=lastSumHits
        self.pwHits[j]=sumHits/nmean
        self.pwHpFA[j]=sumHpFA/nmean
        self.pwndata[j]=nmean
        lastSumHits=self.pwHits[j]
    else:
        if (self.forceContinuity):
            self.pwRank[j]=(thetaInf+thetaSup)/2
            self.pwHits[j]=lastSumHits
        else:
            self.pwRank[j]=float('nan')
            self.pwHits[j]=float('nan')
            self.pwHpFA[j]=float('nan')

    continuousSegments=np.sum(self.icontinuous)
    if (continuousSegments==(self.npiecewise)):
        self.kind='continuous'
    elif (continuousSegments>=int(self.npiecewise/2)): #If there are some gaps in the function
        self.kind='semicontinuous'
        if (self.forceContinuity): #If there are some gaps in the function
            self.kind='forcedContinuous'
    else:
        self.kind='discrete'
    if ((self.sumIUnique)<(self.ndata/2) and self.kind!='continuous'):
        self.kind='discrete'
    if (self.kind=='discrete'):
        self.pwRank=self.Thresholds[self.iUnique]
        self.pwHits=self.Hits[self.iUnique]
        self.pwHpFA=self.HpFA[self.iUnique]

    self.pwLabels=self.pwHits/self.pwHpFA

    return continuousSegments
##########################################END METHOD continuity#####################################################

########################################BEGIN METHOD computePF######################################################


def centeredDF(self,n,X,Y): #centered finite differeces for derivatives
    """
    _`centeredDF`Computes derivatives using centered finite differences.
    """

    DX=np.zeros(n)
    n=len(X)
    DX[1:(n-2)]=0.5*(Y[1:(n-2)]-Y[:(n-3)])/(X[1:(n-2)]-X[:(n-3)])+0.5*(Y[2:(n-1)]-Y[1:(n-2)])/(X[2:(n-1)]-X[1:(n-2)])
    DX[0]=(Y[1]-Y[0])/(X[1]-X[0])
    DX[-1]=(Y[-1]-Y[-2])/(X[-1]-X[-2])
    return DX



def computePF(self,method='centeredDF'):
    """

    This method scales the Hits axis to the interval [0,1]. The self TOC (the TOC which the method is called from) is normalized, there is not new memory allocation.
    :return: Returns the modified TOC curve
    :rtype: ``TOC``

    The ``kind`` TOC curve is *'normalized'*.
    The  true positives plus false positives count is 1, ntppfp=1,
    and true positives, TP=1.
    Nevertheless the basentppfp and basenpos stores the values of the self TOC.

    """
    self.CPF_Rank_given_Presence=np.zeros(self.npiecewise)
    self.CPF_Rank=np.zeros(self.npiecewise)
    self.CPF_Presence_given_Rank=np.zeros(self.npiecewise)
    self.CPF_Rank_given_Presence=self.pwHits/self.np
    self.CPF_Rank=self.pwHpFA/self.ndata
    self.JPF=np.zeros(self.npiecewise)

    if (self.kind=='continuous'):
        self.PF_Rank_given_Presence=self.centeredDF(self.npiecewise,self.pwRank,self.CPF_Rank_given_Presence)
        self.DPF_Rank_given_Presence=self.centeredDF(self.npiecewise,self.pwRank,self.PF_Rank_given_Presence)

        self.PF_Rank=self.centeredDF(self.npiecewise,self.pwRank,self.CPF_Rank)
        self.DPF_Rank=self.centeredDF(self.npiecewise,self.pwRank,self.PF_Rank)

        self.PF_Presence_given_Rank=self.PF_Rank_given_Presence/(self.PF_Rank+(1e-6/self.npiecewise))
        self.PF_Presence_given_Rank=self.PF_Presence_given_Rank/np.sum(0.5*(self.PF_Presence_given_Rank[:-1]+self.PF_Presence_given_Rank[1:])*(self.pwRank[1:]-self.pwRank[:-1]))
        self.CPF_Presence_given_Rank=np.append(np.cumsum(0.5*(self.PF_Presence_given_Rank[:-1]+self.PF_Presence_given_Rank[1:])*(self.pwRank[1:]-self.pwRank[:-1])),1)

        self.DPF_Presence_given_Rank=self.centeredDF(self.npiecewise,self.pwRank,self.PF_Presence_given_Rank)
        self.JPF=self.PF_Rank_given_Presence*(self.np/self.ndata)
    if (self.kind=='discrete'):
        self.PF_Rank_given_Presence=np.zeros(self.npiecewise)
        self.DPF_Rank_given_Presence=np.zeros(self.npiecewise)
        self.PF_Rank_given_Presence[1:]=self.CPF_Rank_given_Presence[1:]-self.CPF_Rank_given_Presence[:-1]
        self.PF_Rank_given_Presence[0]=self.CPF_Rank_given_Presence[0]
        self.DPF_Rank_given_Presence[1:]=self.PF_Rank_given_Presence[1:]-self.PF_Rank_given_Presence[:-1]
        self.DPF_Rank_given_Presence[0]=self.PF_Rank_given_Presence[0]

        self.PF_Rank=np.zeros(self.npiecewise)
        self.DPF_Rank=np.zeros(self.npiecewise)
        self.PF_Rank[1:]=self.CPF_Rank[1:]-self.CPF_Rank[:-1]
        self.PF_Rank[0]=self.CPF_Rank[0]
        self.DPF_Rank[1:]=self.PF_Rank[1:]-self.PF_Rank[:-1]
        self.DPF_Rank[0]=self.PF_Rank[0]

        self.PF_Presence_given_Rank=np.zeros(self.npiecewise)
        self.DPF_Presence_given_Rank=np.zeros(self.npiecewise)
        deltaR=max(np.max((self.PF_Rank_given_Presence*self.np)-(self.PF_Rank*self.ndata)),(1e-6/self.npiecewise))
        print(deltaR)
        self.PF_Presence_given_Rank=(self.PF_Rank_given_Presence*self.np)/(self.PF_Rank*self.ndata+deltaR)
        self.PF_Presence_given_Rank=self.PF_Presence_given_Rank/np.sum(self.PF_Presence_given_Rank)
        self.CPF_Presence_given_Rank=np.cumsum(self.PF_Presence_given_Rank)
        self.DPF_Presence_given_Rank[1:]=self.PF_Presence_given_Rank[1:]-self.PF_Presence_given_Rank[:-1]
        self.DPF_Presence_given_Rank[0]=self.PF_Presence_given_Rank[0]
        self.JPF=self.PF_Rank_given_Presence*(self.np/self.ndata)

    return self.PF_Rank_given_Presence


########################################END METHOD computePF####################################################

def PFsmoothing(self, method='ANN', PFsmoothingFactor=-1, DPFsmoothingFactor=-1, CDFsmoothingFactor=-1, pwHitssmoothingFactor=-1 ):
    # Aqui podemos implementar recepción de argumentos variables con kwargs o un diccionario https://python-intermedio.readthedocs.io/es/latest/args_and_kwargs.html
    
    if (method=='wmeans'):
        self.smoothpwHits=self.meanWindowSmoothing(self.pwRank,self.pwHits,self.npiecewise,-1)
        self.smoothCPF_Rank_given_Presence=self.meanWindowSmoothing(self.pwRank,self.CPF_Rank_given_Presence,self.npiecewise,-1)
        self.smoothPF_Rank_given_Presence=self.meanWindowSmoothing(self.pwRank,self.PF_Rank_given_Presence,self.npiecewise,-1)
        def mfunc(x):
            return x[np.argmax(np.abs(x))]
        self.smoothDPF_Rank_given_Presence=self.meanWindowSmoothing(self.pwRank,self.DPF_Rank_given_Presence,self.npiecewise,-1,mfunc)

        self.smoothpwHpFA=self.meanWindowSmoothing(self.pwRank,self.pwHpFA,self.npiecewise,-1)
        self.smoothCPF_Rank=self.meanWindowSmoothing(self.pwRank,self.CPF_Rank,self.npiecewise,-1)
        self.smoothPF_Rank=self.meanWindowSmoothing(self.pwRank,self.PF_Rank,self.npiecewise,-1)
        self.smoothDPF_Rank=self.meanWindowSmoothing(self.pwRank,self.DPF_Rank,self.npiecewise,-1,mfunc)

        self.smoothCPF_Presence_given_Rank=self.meanWindowSmoothing(self.pwRank,self.CPF_Presence_given_Rank,self.npiecewise,-1)
        self.smoothPF_Presence_given_Rank=self.meanWindowSmoothing(self.pwRank,self.PF_Presence_given_Rank,self.npiecewise,-1)
        self.smoothDPF_Presence_given_Rank=self.meanWindowSmoothing(self.pwRank,self.DPF_Presence_given_Rank,self.npiecewise,-1,mfunc)

        self.smoothJPF=self.meanWindowSmoothing(self.pwRank,self.JPF,self.npiecewise,-1,mfunc)

    elif (method == "ANN"):
        """
        Artificial Neural Networks
        """
        X, Y, DY, DDY = self.fitNN(self.pwRank, self.pwHits)
        maxY=np.max(np.array(Y))
        self.smoothpwHits = np.array(Y)
        self.smoothCPF_Rank_given_Presence = np.array(Y)/maxY
        integ=np.sum(0.5*(DY[:-1]+DY[1:])*(self.pwRank[1:]-self.pwRank[:-1]))
        self.smoothPF_Rank_given_Presence = np.array(DY)/integ
        self.smoothDPF_Rank_given_Presence = np.array(DDY)/integ

        X, Y, DY, DDY = self.fitNN(self.pwRank, self.pwHpFA)
        maxY=np.max(np.array(Y))
        self.smoothpwHpFA = np.array(Y)
        self.smoothCPF_Rank = np.array(Y)/maxY
        integ=np.sum(0.5*(DY[:-1]+DY[1:])*(self.pwRank[1:]-self.pwRank[:-1]))
        self.smoothPF_Rank = np.array(DY)/integ
        self.smoothDPF_Rank = np.array(DDY)/integ

        X, Y, DY, DDY = self.fitNN(self.pwRank, self.pwLabels)
        maxY=np.max(np.array(Y))
        self.smoothpwLabels = np.array(Y)
        self.smoothCPF_Presence_given_Rank = np.array(Y)/maxY
        integ=np.sum(0.5*(DY[:-1]+DY[1:])*(self.pwRank[1:]-self.pwRank[:-1]))
        self.smoothPF_Presence_given_Rank = np.array(DY)/integ
        self.smoothDPF_Presence_given_Rank = np.array(DDY)/integ

        self.smoothJPF=self.smoothPF_Rank_given_Presence*(self.np/self.ndata)
    else:
        print("Smoothing method=",method,"is not implemented!")



def meanWindowSmoothing(self,X,Y,n,smoothingFactor=-1,mfunction=np.mean):
    Yg=np.zeros(n)
    #Smoothing the density using a mean filter, it is similar to a uniform kernel with a window size=smoothing
    if (smoothingFactor==-1):
        nw=min(int(n/2),200)
        smoothing=int(n/nw)
        #density.smwindow=smoothing
    if (smoothing>0):
        Yg[0:smoothing]= mfunction(Y[0:smoothing])
        Yg[(n-smoothing):n]=  mfunction(Y[(n-smoothing):n])
        for i in range(smoothing,n-smoothing):
            Yg[i]=mfunction(Y[(i-smoothing):(i+smoothing)])
    return Yg




def fitNN(self, X, Y, structure = [25, 25, 25], afunctions = ["sigmoid", "sigmoid", "sigmoid"]):

    #intepolacion lineal para tener valores en cada TPpFP
    #0,5,10
    #X = 0,1,2,3,4,5,6,7,8,9,10
    #sigmoidal expand tails for reinforce learning at those extremes
    #stail = 1000000
    maxX0=np.max(X)
    maxY0=np.max(Y)
    minX0=np.min(X)
    #Xinterp = np.arange(0, len(X))
    Xinterp = ((np.arange(0, len(X)+1))*(maxX0-minX0)/(len(X)))+minX0
    Yinterp = np.interp(Xinterp, X, Y)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-7,
        # "no longer improving" being further defined as "for at least 10 epochs"
        patience = 100,
        verbose=0,
        )
    ]

    Dx0=Xinterp[1]-Xinterp[0]
    Dy0=Yinterp[1]-Yinterp[0]
    Dx1=Xinterp[-1]-Xinterp[-2]
    Dy1=Yinterp[-1]-Yinterp[-2]
    X_train_e=Xinterp
    Y_train_e=Yinterp
    for i in range(max(int(0.05*len(Xinterp)),3)):
        X_train_e = np.append(Xinterp[0]-float(i)*Dx0,X_train_e)
        Y_train_e = np.append(Yinterp[0]-float(i)*Dy0,Y_train_e)
        X_train_e = np.append(X_train_e ,X_train_e[-1]+float(i)*Dx1)
        Y_train_e = np.append(Y_train_e ,Y_train_e[-1]+float(i)*Dy1)

    self.ntrain=len(X_train_e)
    maxX=np.max(X_train_e)
    maxY=np.max(Y_train_e)
    minX=np.min(X_train_e)
    minY=np.min(Y_train_e)

    #Normalized TOC
    X_train = ((X_train_e-minX)/(maxX-minX))
    Y_train = ((Y_train_e-minY)/(maxY-minY))

    X_train_s = X_train[::2]
    Y_train_s = Y_train[::2]

    X_valid_s = X_train[1:][::2]
    Y_valid_s = Y_train[1:][::2]

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),        # Input layer of size 1
    tf.keras.layers.Dense(structure[0], input_dim = 1, activation= afunctions[0]),
    tf.keras.layers.Dense(structure[1], input_dim = structure[0], activation=afunctions[1]),
    tf.keras.layers.Dense(structure[2], input_dim = structure[1], activation= afunctions[2]),
    tf.keras.layers.Dense(1, input_dim = structure[2], activation='linear')
    ])

        # Compile the model
    model.compile(optimizer = 'adam',  loss = "mse")

    # Train the model
    model.fit(X_train_s, Y_train_s, validation_data=(X_valid_s, Y_valid_s), epochs=10000, callbacks=callbacks, verbose=0, batch_size=min(max(int(len(X_train_s)/20),30),len(X_train_s)) )


    #regresar la predicción el dominio en las coordenadas de X

    yhat = model.predict( (X-minX)/(maxX-minX))
    realHits = (yhat)*(maxY-minY)+minY
    #print(realHits)
    input_data = tf.convert_to_tensor((X-minX)/(maxX-minX))
    #output = tf.convert_to_tensor(Y*(maxY-minY)+minY)
    with tf.GradientTape() as tape2:
        tape2.watch(input_data)
        with tf.GradientTape() as tape1:
            tape1.watch(input_data)
            output = model(input_data)
        first_derivative = tape1.gradient(output, input_data)
    second_derivative = tape2.gradient(first_derivative, input_data)


    return np.array(X), np.array(realHits), np.array(first_derivative), np.array(second_derivative)



##########################################BEGIN METHOD areaComputacion###############################################

def areaComputation(self):
    """

    This method computes the areaun der the curve of the TOC and parallelogram and the proportional ratio
    :return: Returns the TOC's area under the curve
    :rtype: ``float``

    """
    self.area=0
    AUC=0
    pararea=0
    if (self.kind=='discrete'):
        #Th equivalent area under the curve is the sum of hits
        AUC=np.sum(self.Hits[self.iUnique])
        #Sum of parallelogram heights in the first section
        idx=self.HpFA[self.iUnique]<self.np
        parareaBP=np.sum(self.HpFA[self.iUnique][idx])

        #Sum of parallelogram heights in the second section
        idx=self.HpFA[self.iUnique]>=self.np
        parareaP=self.np*np.sum(idx)

        #Sum of parallelogram heights in the last  section
        xoffset=self.ndata-self.np
        idx=self.HpFA[self.iUnique]>xoffset
        parareaAP=np.sum((self.HpFA[self.iUnique][idx]-xoffset))
        #This is th equivalent area under the curve miinus the last triangle of the paralellogram
        self.area=AUC-parareaAP
        pararea=parareaBP+parareaP-parareaAP
    else:
        AUC=np.sum(0.5*(self.Hits[self.iUnique][:-1]+self.Hits[self.iUnique][1:])*(self.HpFA[self.iUnique][1:]-self.HpFA[self.iUnique][:-1]))
        pararea=self.np*self.ndata
        self.area=AUC-(self.np*self.np)/2
    self.areaRatio=self.area/pararea
    self.pararea=pararea
    return AUC


##########################################END METHOD areaComputacion#################################################



##########################################BEGIN METHOD areaDComputacion###############################################

def areaDComputation(self):
    """

    This method computes the areaun der the curve of the TOC and parallelogram and the proportional ratio
    :return: Returns the TOC's area under the curve
    :rtype: ``float``

    """
    area=0
    AUC=0
    pararea=0
    ##########!!!!!!!!!!ME FALTA PROBAR EL CASO DISCRETO############
    if (self.kind=='discrete'):
        #Th equivalent area under the curve is the sum of hits
        AUC=np.sum(self.pwHits)
        #Sum of parallelogram heights in the first section
        idx=self.pwHpFA<self.np
        parareaBP=np.sum(self.pwHpFA[idx])

        #Sum of parallelogram heights in the second section
        idx=self.pwHpFA>=self.np
        parareaP=self.np*np.sum(idx)

        #Sum of parallelogram heights in the last  section
        xoffset=self.ndata-self.np
        idx=self.pwHpFA>xoffset
        parareaAP=np.sum((self.pwHpFA[idx]-xoffset))
        #This is th equivalent area under the curve miinus the last triangle of the paralellogram
        self.Darea=AUC-parareaAP
        pararea=parareaBP+parareaP-parareaAP
        #self.areaRatio=area/pararea
    else:
        nonan=~np.isnan(self.pwHits)
        AUC=np.sum(0.5*(self.pwHits[nonan][:-1]+self.pwHits[nonan][1:])*(self.pwHpFA[nonan][1:]-self.pwHpFA[nonan][:-1]))
        pararea=self.np*self.ndata
        self.Darea=AUC-(self.np*self.np)/2
    self.areaDRatio=self.Darea/pararea
    pararea=pararea
    #print('AreaRatio', self.areaDRatio)
    #print('pararea', pararea)
    return self.areaDRatio



##########################################END METHOD areaComputacion#################################################


#This function computes a probability value given

def rank2prob(self,rank,Prob=None):

    """

    This function computes probability values associated to a rank value. The ``thresholds`` array of the density TOC is used for this purpose.
    Very possibly the rank is the same than those used to compute a standard TOC instantiated by TOC(rank,groundtruth), hence the indices used
    in the constructor are available and can save computational time. Otherwise the indices are recomputed. In any case the inputed ``rank``
    array must be in the same interval than the thresholds.

    :param rank: A numpy array with the ran. The intended uses is that this rank comes from the standard TOC computation, and to associate this rank with geolocations, hence the probabilities can be associated in the same order.

    :param kind: if ``kind`` is ''density'' the probabilitites are computed with the non-smoothed density function, otherwise the smooth version is used. Notice that a this function only must be called from a ''density'' kind TOC. Optional

    :param indices: Indices of the reversely sorted rank, this array is computed by the standard TOC computation. Hence the computational cost of recomputing them could be avoided, otherwise the indices are recomputed and they are not stored. Optional

    return: a numpy array with the probabilities. The probabilities do not sum 1, instead they sum ``PDataProp``, that is to say they sum the proportion of positives in the data. That is an estimation of the probability of having a 1-class valued datum.

    :rtype: numpy array

    """
    indices=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
    if (Prob==None):
        Prob=self.PF_Presence_given_Rank

    DF=Prob
    nr=len(rank)
    nd=self.npiecewise
    prob=np.zeros(nr)
    j=0
    minr=np.min(self.pwRank)
    maxr=np.max(self.pwRank)
    deltammr=(maxr-minr)/self.npiecewise
    for i in range(nd-1):
        nd=0
        jini=j
        while(j<nr  and  (rank[indices[j]])<=(self.pwRank[i+1])):
            if (rank[indices[j]]>=minr and rank[indices[j]]<=maxr):
                deltar=self.pwRank[i+1]-self.pwRank[i]
                nd=nd+1
                prob[indices[j]]=  (DF[i+1]*(rank[indices[j]]-self.pwRank[i])+DF[i]*(self.pwRank[i+1]-rank[indices[j]]))/deltar
            elif(rank[indices[j]]<minr and rank[indices[j]]>=(minr-deltammr)):
                prob[indices[j]]=DF[0]
                nd=nd+1
            elif(rank[indices[j]]>maxr and rank[indices[j]]<=(maxr+deltammr)):
                prob[indices[j]]=DF[-1]
                nd=nd+1
            else:
                prob[indices[j]]=0
            j+=1
            jend=j
    return(prob)


def simulate(self,rank,nprop=1):
    prob=self.rank2prob(rank)
    prob=np.cumsum(prob)
    prob=prob/prob[-1]
    #print(prob[-1])
    simulation=np.zeros(len(rank))
    if (nprop<1):
        nprop=int(nprop*len(prob)+0.5)
    else:
        nprop=int(nprop)
    nsim=0
    while(nsim<nprop):
        rsamples=sorted(np.random.rand(nprop-nsim))
        j=0
        for i in range(len(prob)):
            while(rsamples[j]<prob[i] and j<len(rsamples)):
                simulation[i]=1
                j+=1
                if (j==len(rsamples)):
                    break
            if (j==len(rsamples)):
                break
        nsim=int(np.sum(simulation))
        #print('nsim',nsim,'nprop',nprop)
    return simulation


def rasterize(self,feature, lat, lon,crs=32641,rfile='raster.tif'):
    from rasterio.io import MemoryFile
    #from affine import Affine
    latdif=np.abs(lat[1:]-lat[:-1])
    londif=np.abs(lon[1:]-lon[:-1])
    Dlat=np.min(latdif[latdif>0])
    Dlon=np.min(londif[londif>0])
    minLat=min(lat)
    minLon=min(lon)
    maxLat=max(lat)
    maxLon=max(lon)
    lenLat=(maxLat-minLat)
    lenLon=(maxLon-minLon)
    nrow=round(lenLat/Dlat+1)
    ncol=round(lenLon/Dlon+1)
    transform = from_origin( maxLon+Dlon/2,maxLat+Dlat/2, Dlon,Dlat)
    with MemoryFile() as memfile:
        meta = {"count": 1, "width": ncol, "height": nrow, "transform": transform, "nodata": float('nan'), "dtype": "float64"}
        with memfile.open(driver='GTiff', **meta) as raster_file:
            raster_file.write(float('nan')*np.ones(ncol*nrow).reshape(1,nrow,ncol))
            self.raster=raster_file.read(1)
            i,j=rasterio.transform.rowcol(raster_file.transform,xs=lon,ys=lat)
            i=i
            j=j
            self.raster[i,j]=feature
            raster_file.write(self.raster, 1)
            raster_file.close()



##########################################BEGIN METHOD tickPositions#################################################

#def tickPositions(self,sorted_ranks,Thresholds, HpFA, n = 30):

    #TS = np.linspace(Thresholds[0], Thresholds[-1], n)
    #unique_rank_positions = HpFA
    #P = list([])
    #for t in TS:
        #ix = np.argmin((Thresholds - t)**2)
        #P.append(unique_rank_positions[ix])
    #return np.array(P), np.around(TS, 1)

def tickPositions(self,Xupper, n = 30):
    if (n>len(Xupper)):
        n=len(Xupper)
    equalXs = np.linspace(Xupper[0], Xupper[-1], n)
    iX = np.zeros(n)
    j=0
    for i in range(n):
        while(equalXs[i]>=Xupper[j]):
            j+=1
            if (j==len(Xupper)):
                j=j-1
                break
        iX[i]=j
    return equalXs, iX.astype(int)


#########################################END METHOD tickPositions####################################################


##########################################BEGIN METHOD __plotTOC#####################################################

def __plotTOC(self,filename = '',title='default',Legend='TOC',kind='TOC',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))
    if (title=='default'):
        title = "TOC/CPF"
    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    #Preparing the overlaped plot with the top and right axis for the CDF
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    #Small ticks positions at the secondary axis
    equalXs,iX = self.tickPositions(self.Thresholds[self.iUnique])
    ax2.set_xticks(self.HpFA[self.iUnique][iX])
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(self.HpFA[self.iUnique][iX]))

    ##Large ticks
    n = np.argmax(self.HpFA[self.iUnique][iX][1:]-self.HpFA[self.iUnique][iX][:-1])
    ax2.xaxis.set_major_locator(ticker.FixedLocator( self.HpFA[self.iUnique][iX][[0, n, n+1, -1]] ))
    ##Large ticks labels
    prtdec=max(-np.log10(np.max(abs(equalXs))),3)
    ax2.set_xticklabels(np.around(equalXs[[0, n, n+1, -1]],prtdec), rotation = 90)
    if ('intersections' in options):
        ax1.vlines(self.HpFA[self.iUnique][iX][[n, n+1]],0,self.np,linestyles='dotted',colors="tab:orange",alpha=0.5)
        ax1.hlines(self.Hits[self.iUnique][iX][[n, n+1]],0,self.ndata,linestyles='dotted',colors="tab:orange",alpha=0.5)

    tlabel='$X$'
    if (self.featureName!='X'):
        tlabel='$X=$'+self.featureName

    ##print(tlabel)
    ax2.set_xlabel(tlabel, color = "tab:blue")
    ax2.set_ylabel("P($X \leq Threshold~~ | ~~Y=presence$)", color = "tab:blue")
    ax2.tick_params(labeltop = True)
    ax2.tick_params(labelright = True)
    ax2.tick_params(labelbottom = False)
    ax2.tick_params(labelleft = False)
    ax2.tick_params(labelsize=labelsize)
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.set_title(title,va='baseline')

    #parallelogram coordinates
    rx = np.array([0, self.np, self.ndata, self.ndata-self.np, 0])
    ry = np.array([0, self.np, self.np, 0, 0])

    ax1.set_ylim(0, 1.01*self.np)
    ax1.set_xlim(-0.001, 1.01*self.ndata)
    ax1.tick_params(labelsize=labelsize)
    ax1.text(0.575*self.ndata, 0.025*self.np, 'AR = ')
    ax1.text(0.675*self.ndata, 0.025*self.np, str(round(self.areaRatio, 4)))

    #Ploting the uniform distribution line
    ax1.plot(np.array([0, self.ndata]), np.array([0, self.np]),'b-.',
    label = "Uniform random distribution")

    ax1.plot(rx, ry, '--')
    marker='-o'
    markersize=2
    markerb='-o'
    if (kind=='discrete'):
        marker='s'
        markersize=1
        ax1.vlines(self.HpFA[self.iUnique],0,self.Hits[self.iUnique],colors="tab:red",alpha=0.2)

    #TOC thresholds
    ax1.plot(self.HpFA[self.iUnique],self.Hits[self.iUnique],marker,markersize = markersize,label = Legend, linewidth = 1,color = "tab:red")

    ax1.set_xlabel("Hits + False Alarms")
    ax1.set_ylabel("Hits")
    ax1.legend(loc = 'upper left')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")

##########################################END METHOD __plotTOC#######################################################


##########################################BEGIN METHOD __plotPiecewiseTOC#####################################################

def __plotPiecewiseTOC(self,filename = '',title='default',Legend='TOC',kind='TOC',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    #Preparing the overlaped plot with the top and right axis for the CDF
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    #Small ticks positions at the secondary axis
    equalXs,iX = self.tickPositions(self.pwThresholds)
    ax2.set_xticks(self.pwHpFA[iX])
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(self.pwHpFA[iX]))

    ##Large ticks
    n = np.argmax(self.pwHpFA[iX][1:]-self.pwHpFA[iX][:-1])
    ax2.xaxis.set_major_locator(ticker.FixedLocator( self.pwHpFA[iX][[0, n, n+1, -1]] ))
    ##Large ticks labels
    prtdec=max(-np.log10(np.max(abs(equalXs))),1)
    ax2.set_xticklabels(np.around(equalXs[[0, n, n+1, -1]],prtdec), rotation = 90)
    if ('intersections' in options):
        ax1.vlines(self.pwHpFA[iX][[n, n+1]],0,self.np,linestyles='dotted',colors="tab:orange",alpha=0.5)
        ax1.hlines(self.pwHits[iX][[n, n+1]],0,self.ndata,linestyles='dotted',colors="tab:orange",alpha=0.5)

    tlabel='$X$'
    if (self.featureName!='X'):
        tlabel='$X=$'+self.featureName


    ax2.set_xlabel(tlabel, color = "tab:blue")
    ax2.set_ylabel("P($X \leq Threshold~~ | ~~Y=presence$)", color = "tab:blue")
    ax2.tick_params(labeltop = True)
    ax2.tick_params(labelright = True)
    ax2.tick_params(labelbottom = False)
    ax2.tick_params(labelleft = False)
    ax2.tick_params(labelsize=labelsize)
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.set_title(title,va='baseline')



    title = "Piecewise approximation to the TOC"
    #parallelogram coordinates
    rx = np.array([0, self.np, self.ndata, self.ndata-self.np, 0])
    ry = np.array([0, self.np, self.np, 0, 0])
    ax1.set_ylim(0, 1.01*self.np)
    ax1.set_xlim(-0.001, 1.01*self.ndata)
    ax1.tick_params(labelsize=labelsize)
    ax1.text(0.575*self.ndata, 0.025*self.np, 'AUC = ')
    ax1.text(0.675*self.ndata, 0.025*self.np, str(round(self.areaDRatio, 4)))

    #Ploting the uniform distribution line
    ax1.plot(np.array([0, self.ndata]), np.array([0, self.np]),'b-.',
    label = "Uniform random distribution")

    ax1.plot(rx, ry, '--')
    marker='-o'
    markerb='-o'
    markersize=2
    if (self.kind=='discrete'):
        marker='s'
        markersize=1
        ax1.vlines(self.pwHpFA,0, self.pwHits,colors="tab:red",alpha=0.2)
    #TOC thresholds
    ax1.plot(self.pwHpFA, self.pwHits, marker,markersize = markersize,label = Legend, linewidth = 1,color = "tab:red")
    ax1.set_xlabel("Hits + False Alarms")
    ax1.set_ylabel("Hits")
    ax1.legend(loc = 'upper left')

    tlabel='$X$'
    if (self.featureName!='X'):
        tlabel='$X=$'+self.featureName

    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")

##########################################END METHOD __plotTOC#######################################################



def __plotCDF(self,filename = '',title='default',Legend='CPF',kind='CPF-Rank|Presence',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))
    if (kind=='CPF-Rank|Presence'):
        CDF=self.CPF_Rank_given_Presence
        titleM="Cumulative Probability Function of Rank given Presence"
    elif(kind=='CPF-Rank'):
        CDF=self.CPF_Rank
        titleM="Cumulative Probability Function of the Rank"
    elif(kind=='CPF-Presence|Rank'):
        CDF=self.CPF_Presence_given_Rank
        titleM="Cumulative Probability Function of Presence given Rank"
    else:
        print(kind,' is not implemented!')
        return 0
    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    if (title=='default'):
        title = titleM
    ax1.set_title(title,va='baseline')
    ax1.set_ylim(0, 1.01)
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)
    marker='-o'
    markerb='-o'
    markersize=2


    if (self.kind=='discrete'):
        marker='s'
        markersize=1
    if ('vlines' in options):
        ax1.vlines(self.pwRank,0,CDF,colors="tab:orange",alpha=0.1)
    #TOC thresholds
    ax1.plot(self.pwRank,CDF, marker, markersize = markersize, label = Legend, linewidth = 1,color = "tab:orange")

    #Ploting the uniform distribution line
    ax1.plot(np.array([self.minr, self.maxr]), np.array([0, 1]),'b-.',label = "Uniform random distribution")

    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    if (kind=='CPF-Rank|Presence'):
        ax1.set_ylabel("P($X \leq Threshold~~ | ~~Y=presence$)")
    elif (kind=='CPF-Rank'):
        ax1.set_ylabel("P($X \leq Threshold$)")
    if (kind=='CPF-Presence|Rank'):
        ax1.set_ylabel("P($Y=presence~~|X\leq Threshold $)")

    ax1.legend(loc = 'lower right')

    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")


##########################################END METHOD __plotTOC#######################################################

def __plotPF(self,filename = '',title='default',Legend='PF',kind='PF-Rank|Presence',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (kind=='PF-Rank|Presence' or kind=='smoothPF-Rank|Presence'):
        PF=self.PF_Rank_given_Presence
        smoothPF=self.smoothPF_Rank_given_Presence
        CDF=self.CPF_Rank_given_Presence
    elif(kind=='PF-Rank' or kind=='smoothPF-Rank'):
        PF=self.PF_Rank
        smoothPF=self.smoothPF_Rank
        CDF=self.CPF_Rank
    elif(kind=='PF-Presence|Rank' or kind=='smoothPF-Presence|Rank'):
        PF=self.PF_Presence_given_Rank
        smoothPF=self.smoothPF_Presence_given_Rank
        CDF=self.CPF_Presence_given_Rank
    else:
        print(kind,' is not implemented!' )
        return 0


    maxpf=np.max(PF)
    #print(autodpi)
    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    if (title=='default'):
        if (kind[0:8]=='smoothPF'):
            title = "Smoothed Probability Density Function (conditional to presence)"
        else:
            title = "Probability Density Function (conditional to presence)"


    ax1.set_ylim(0, 1.01*np.max(PF))
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)

    #Ploting the uniform distribution line
    ax1.plot(np.array([self.minr, self.maxr]), np.array([1/(self.maxr-self.minr),1/(self.maxr-self.minr)]),'b-.',
    label = "Random classifier")
    if ('vlines' in options):
        if (kind[0:8]=='smoothPF'):
            ax1.vlines(self.pwRank,smoothPF,maxpf,colors="tab:gray",alpha=0.5,linewidth = 0.2)
            if ('quartiles' in options):
                ax1.vlines(self.pwRank,0,smoothPF,colors="#2c03fc",alpha=0.05,linewidth = 0.1)
            else:
                ax1.vlines(self.pwRank,0,smoothPF,colors="#2c03fc",alpha=0.5,linewidth = 0.2)
        else:
            ax1.vlines(self.pwRank,PF,maxpf,colors="tab:gray",alpha=0.5,linewidth = 0.2)
            if ('quartiles' in options):
                ax1.vlines(self.pwRank,0,PF,colors="#2c03fc",alpha=0.05,linewidth = 0.1)
            else:
                ax1.vlines(self.pwRank,0,PF,colors="#2c03fc",alpha=0.5,linewidth = 0.2)


    if ('quartiles' in options):
        i1=np.argmax(CDF>0.25)
        i2=np.argmax(CDF>0.5)
        i3=np.argmax(CDF>0.75)
        ax1.fill_between(self.pwRank[0:i1+1],PF[0:i1+1],color='#fc9d03',alpha=0.55)
        ax1.fill_between(self.pwRank[i1:i2+1],PF[i1:i2+1],color='#fc0703',alpha=0.55)
        ax1.fill_between(self.pwRank[i2:i3+1],PF[i2:i3+1],color='#ca03fc',alpha=0.55)
        ax1.fill_between(self.pwRank[i3:],PF[i3:],color='#016e32',alpha=0.55) ##fc3d03

    marker='-o'
    markersize=0.5
    if (self.kind=='discrete'):
        marker='h'
        markersize=3
        if (kind[0:8]=='smoothPF'):
            ax1.vlines(self.pwRank,0,smoothPF,colors="#4287f5",alpha=0.95,linewidth = 3)
        else:
            ax1.vlines(self.pwRank,0,PF,colors="#4287f5",alpha=0.95,linewidth = 3)
        if (title=='default'):
            title="Mass Probability Function (conditional to presence)"
            if (kind=='smoothPF'):
                title="Regularized Mass Probability Function (conditional to presence)"

    ax1.set_title(title,va='baseline')
    #TOC thresholds
    if (kind[0:8]=='smoothPF'):
        ax1.plot(self.pwRank,smoothPF,marker,markersize = markersize,label = 'Smoothed PF', linewidth = 1,color = "#4287f5")
    else:
        ax1.plot(self.pwRank, PF,marker,markersize = markersize,label = Legend, linewidth = 1,color = "#4287f5")


    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    if (kind=='smoothPF-Rank|Presence' or kind=='PF-Rank|Presence'):
        ax1.set_ylabel("P($X = Threshold~~ | ~~Y=Presence$)")
    elif (kind=='smoothPF-Rank' or kind=='PF-Rank'):
        ax1.set_ylabel("P($X = Threshold$)")
    elif (kind=='smoothPF-Presence|Rank' or kind=='PF-Presence|Rank'):
        ax1.set_ylabel("P($Y=Presence|X = Threshold$)")
    ax1.legend(loc = 'center right')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")



def __plotDPF(self,filename = '',title='default',Legend='DPF',kind='DPF-Rank|Presence',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator


    DPF=None
    if (kind=='DPF-Rank|Presence' or kind=='smoothDPF-Rank|Presence'):
        DPF=self.DPF_Rank_given_Presence
        smoothDPF=self.smoothDPF_Rank_given_Presence
        CDF=self.CPF_Rank_given_Presence
    elif(kind=='DPF-Rank' or kind=='smoothDPF-Rank'):
        DPF=self.DPF_Rank
        smoothDPF=self.smoothDPF_Rank
        CDF=self.CPF_Rank
    elif(kind=='DPF-Presence|Rank' or kind=='smoothDPF-Presence|Rank'):
        DPF=self.DPF_Presence_given_Rank
        smoothDPF=self.smoothDPF_Presence_given_Rank
        CDF=self.CPF_Presence_given_Rank
    else:
        print(kind,' is not implemented!')
        return None

    maxDPF=np.max(DPF)
    minDPF=np.min(DPF)
    maxSmoothDPF=np.max(smoothDPF)
    minSmoothDPF=np.min(smoothDPF)
    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    if(title=='default'):
        if (kind[0:9]=='smoothDPF'):
            title = "Smoothed First Derivative of the Probability Density Function"
        else:
            title = "First Derivative of the Probability Density Function"
    if (kind[0:9]=='smoothDPF'):
        ax1.set_ylim(1.01*minSmoothDPF, 1.01*maxSmoothDPF)
    else:
        ax1.set_ylim(1.01*minDPF, 1.01*maxDPF)
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)

    #Ploting the uniform distribution line
    ax1.plot(np.array([self.minr, self.maxr]), np.array([0,0]),'b-.',
    label = "Random classifier")

    if ('vlines' in options):
        if (kind[0:9]=='smoothDPF'):
            ax1.vlines(self.pwRank[smoothDPF>0],smoothDPF[smoothDPF>0],maxSmoothDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.pwRank[smoothDPF<0],smoothDPF[smoothDPF<0],minSmoothDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.pwRank[smoothDPF==0],maxSmoothDPF,minSmoothDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
        else:
            ax1.vlines(self.pwRank[DPF>0],DPF[DPF>0],maxDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.pwRank[DPF<0],DPF[DPF<0],minDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.pwRank[DPF==0],maxDPF,minDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)


    marker='-o'
    markersize=1
    fmt=''
    if (self.kind=='discrete'):
        marker='s'
        fmt='s'
        markersize=3
        if (kind[0:9]=='smoothDPF'):
            ax1.vlines(self.pwRank,0,smoothDPF,colors="#4287f5",alpha=0.95,linewidth = 1)
        else:
            ax1.vlines(self.pwRank,0,DPF,colors="#4287f5",alpha=0.95,linewidth = 1)
        if(title=='default'):
                title="First Difference of the Mass Probability Function"
                if (kind=='smoothDPF'):
                    title="Regularized Difference of the Mass Probability Function"

    ax1.set_title(title,va='baseline')

    #TOC thresholds
    if (kind[0:9]=='smoothDPF'):
        ax1.plot(self.pwRank,smoothDPF,marker='s',markersize = 0.5,label = 'Smoothed DPF', linewidth = 1,color = "#4287f5",alpha=1)
    else:
        ax1.plot(self.pwRank,DPF,marker,markersize = markersize,label = Legend, linewidth = 1,color = "#4287f5")

    if ('quartiles' in options):
        i1=np.argmax(CDF>0.25)
        i2=np.argmax(CDF>0.5)
        i3=np.argmax(CDF>0.75)
        ax1.fill_between(self.pwRank[0:i1+1],DPF[0:i1+1],color='#fc9d03',alpha=0.5)
        ax1.fill_between(self.pwRank[i1:i2+1],DPF[i1:i2+1],color='#fc0703',alpha=0.5)
        ax1.fill_between(self.pwRank[i2:i3+1],DPF[i2:i3+1],color='#ca03fc',alpha=0.5)
        ax1.fill_between(self.pwRank[i3:],DPF[i3:],color='#016e32',alpha=0.5)


    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    if (kind=='smoothDPF-Rank|Presence' or kind=='DPF-Rank|Presence'):
        ax1.set_ylabel(r'$\frac{D}{Dx}~P(X = Threshold~~ | ~~Y=presence$)')
    elif (kind=='smoothDPF-Rank' or kind=='DPF-Rank'):
        ax1.set_ylabel(r'$\frac{D}{Dx}~P(X = Threshold)')
    elif (kind=='smoothDPF-Presence|Rank' or kind=='DPF-Presence|Rank'):
        ax1.set_ylabel(r'$\frac{D}{Dx}~P(Y=presence | X=Threshold$)')

    ax1.legend(loc = 'center right')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")


def __plotJPF(self,filename = '',title='default',Legend='Join PDF Presence-Rank',kind='JPDF',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (kind=='JPF' or kind=='smoothJPF'):
        JPF=self.JPF
        CDF=self.CPF_Rank_given_Presence
    else:
        return 0

    maxpf=np.max(JPF)
    #print(autodpi)
    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    if (title=='default'):
        if (kind[0:8]=='smoothJPF'):
            title = "Smoothed Join Probability Dens. Funct. Rank & Presence"
        else:
            title = "Join Probability Dens. Func. Rank & Presence"

    ax1.set_ylim(0, 1.01*np.max(JPF))
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)

    #Ploting the uniform distribution line
    pprop=(self.np/self.ndata)
    ax1.plot(np.array([self.minr, self.maxr]), np.array([pprop/(self.maxr-self.minr),pprop/(self.maxr-self.minr)]),'b-.',
    label = "Random classifier")
    if ('vlines' in options):
        if (kind[0:8]=='smoothJPF'):
            ax1.vlines(self.pwRank,smoothJPF,maxpf,colors="tab:gray",alpha=0.5,linewidth = 0.2)
            if ('quartiles' in options):
                ax1.vlines(self.pwRank,0,smoothJPF,colors="#2c03fc",alpha=0.05,linewidth = 0.1)
            else:
                ax1.vlines(self.pwRank,0,smoothJPF,colors="#2c03fc",alpha=0.5,linewidth = 0.2)
        else:
            ax1.vlines(self.pwRank,JPF,maxpf,colors="tab:gray",alpha=0.5,linewidth = 0.2)
            if ('quartiles' in options):
                ax1.vlines(self.pwRank,0,JPF,colors="#2c03fc",alpha=0.05,linewidth = 0.1)
            else:
                ax1.vlines(self.pwRank,0,JPF,colors="#2c03fc",alpha=0.5,linewidth = 0.2)


    if ('quartiles' in options):
        i1=np.argmax(CDF>0.25)
        i2=np.argmax(CDF>0.5)
        i3=np.argmax(CDF>0.75)
        ax1.fill_between(self.pwRank[0:i1+1],JPF[0:i1+1],color='#fc9d03',alpha=0.55)
        ax1.fill_between(self.pwRank[i1:i2+1],JPF[i1:i2+1],color='#fc0703',alpha=0.55)
        ax1.fill_between(self.pwRank[i2:i3+1],JPF[i2:i3+1],color='#ca03fc',alpha=0.55)
        ax1.fill_between(self.pwRank[i3:],JPF[i3:],color='#016e32',alpha=0.55) ##fc3d03

    marker='-o'
    markersize=0.5
    if (self.kind=='discrete'):
        marker='h'
        markersize=3
        if (kind[0:8]=='smoothPF'):
            ax1.vlines(self.pwRank,0,smoothJPF,colors="#4287f5",alpha=0.95,linewidth = 3)
        else:
            ax1.vlines(self.pwRank,0,JPF,colors="#4287f5",alpha=0.95,linewidth = 3)
        if (title=='default'):
            title="Mass Probability Function (conditional to presence)"
            if (kind=='smoothPF'):
                title="Regularized Mass Probability Function (conditional to presence)"

    ax1.set_title(title,va='baseline')
    #TOC thresholds
    if (kind[0:8]=='smoothJPF'):
        ax1.plot(self.pwRank,smoothJPF,marker,markersize = markersize,label = 'Smoothed JPF Rank & Presence', linewidth = 1,color = "#4287f5")
    else:
        ax1.plot(self.pwRank, JPF,marker,markersize = markersize,label = Legend, linewidth = 1,color = "#4287f5")


    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    if (kind=='smoothJPF-Rank & Presence' or kind=='JPF-Rank & Presence'):
        ax1.set_ylabel("P($X = Threshold~~ & ~~Y=Presence$)")
    ax1.legend(loc = 'center right')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")



def __plotRaster(self,filename = '',title='default',Legend='Raster',kind='raster',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator


    cmap = mpl.colormaps['RdBu']  # viridis is the default colormap for imshow
    if ('binary' in options):
        cmap = mpl.colors.ListedColormap(['red', '#053061'])

    cmap.set_bad(color='gray')
    plot=plt.imshow(self.raster, cmap=cmap)
    plt.colorbar()
    plt.minorticks_on()
    tlabel=Legend
    if (Legend=='Raster'):
        tlabel='$Rank=$'+self.featureName

    plt.title(tlabel,loc='center')

    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")




##########################################BEGIN METHOD plot#####################################################

#This function plots the TOC to the terminal or to a file
def plot(self,filename = '',title='default',Legend='default',kind='None',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    """
    A generic plot function for all the kind of TOCs.  All the parameters are optional. If ``filename`` is not given it plots to a window, otherwise it is a png file.

    :param filename: Optional. If given it must be a png filename, otherwise the TOC is plotted to a window.

    :param title: Optional, title of the plot.

    :param kind: Optional, a standard TOC can be plotted normalized or in the original axis values.

    :param height: pixels of the height. 1800 by default.

    :param width: pixels of the width. 1800 by default.

    :param dpi: resolution. 300 by default.

    :param xlabel: string.

    :param ylabel: string.

    :return: it does not return anything.

    """
    if (kind=='None'):
        kind=self.kind
    if (kind=='continuous' or kind=='semicontinuous' or kind=='forcedContinuous' or kind=='discrete'):
        if (Legend=='default'):
            Legend=kind+" TOC"
        self.__plotTOC(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    elif (kind=='piecewise'):
        if (Legend=='default'):
            Legend="Discrete approx. of the TOC"
        self.__plotPiecewiseTOC(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    elif (kind[0:3]=='CPF' or kind[0:8]=='smoothCPF'):
        if (Legend=='default'):
            Legend="Cumulative Distribution Function"
        self.__plotCDF(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    elif (kind[0:2]=='PF' or kind[0:8]=='smoothPF'):
        if (Legend=='default'):
            Legend=kind
        self.__plotPF(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    elif (kind[0:3]=='DPF' or kind[0:9]=='smoothDPF'):
        if (Legend=='default'):
            Legend=kind
        self.__plotDPF(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    elif (kind[0:3]=='JPF' or kind[0:9]=='smoothJPF'):
        if (Legend=='default'):
            Legend=kind
        self.__plotJPF(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    elif (kind=='Raster'):
        if (Legend=='default'):
            Legend='Raster'
        self.__plotRaster(filename,title,Legend,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    else:
        print("Error: the plot of kind:",kind," is not defined!")



##########################################END METHOD plot#######################################################



PATOC.discretization=discretization
PATOC.continuity=continuity
PATOC.areaComputation=areaComputation
PATOC.tickPositions=tickPositions
PATOC.areaDComputation=areaDComputation
PATOC.computePF=computePF
PATOC.centeredDF=centeredDF
PATOC.PFsmoothing=PFsmoothing
PATOC.fitNN=fitNN
PATOC.meanWindowSmoothing=meanWindowSmoothing
PATOC.rank2prob=rank2prob
PATOC.rasterize=rasterize
PATOC.simulate=simulate
PATOC.__plotTOC=__plotTOC
PATOC.__plotCDF=__plotCDF
PATOC.__plotPF=__plotPF
PATOC.__plotJPF=__plotJPF
PATOC.__plotDPF=__plotDPF
PATOC.__plotPiecewiseTOC=__plotPiecewiseTOC
PATOC.__plotRaster=__plotRaster
PATOC.plot=plot
