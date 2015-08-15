# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:34:16 2015

@author: d
"""

import statsmodels.api as sm 
import statsmodels.formula.api as smf

#data = sm.datasets.get_rdataset("dietox", "geepack").data

md = smf.mixedlm("i ~ deprivation", meta, groups=meta["Genre"]) 
mdf = md.fit() 

print mdf.summary()