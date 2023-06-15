import numpy as np

#code from Walter Del Pozzo
def FindHeightForLevel(inArr, adLevels):
   """
   Computes the height of a :math:`2D` function for given levels

   :param inArr: function values
   :type inArr: array
   :param adLevels: levels
   :type adLevels: list or array

   :return: function values with levels closest to *levels*
   :rtype: array
   """

   # flatten the array
   oldshape = np.shape(inArr)
   adInput  = np.reshape(inArr,oldshape[0]*oldshape[1])
   #adInput=inArr

   # get array specifics
   nLength  = np.size(adInput)

   # create reversed sorted list
   adTemp   = -1.0 * adInput
   adSorted = np.sort(adTemp)
   adSorted = -1.0 * adSorted

   # create the normalised cumulative distribution
   adCum    = np.zeros(nLength)
   adCum[0] = adSorted[0]

   for i in range(1,nLength):
       adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])

   adCum    = adCum - adCum[-1]

   # find the values closest to levels
   adHeights = []
   for item in adLevels:
       idx = (np.abs(adCum-np.log(item))).argmin()
       adHeights.append(adSorted[idx])

   adHeights = np.array(adHeights)
   return np.sort(adHeights)