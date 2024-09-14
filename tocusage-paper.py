#!/usr/bin/env python3

#Test for the computation of the Total Operating Charcateristic Curve and
#TOC=
#
#Author:S. Ivvan Valdez 
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.


# if __name__ == '__main__':
import patoc as toc
import pandas as pd
import importlib
import os
import time
#Loading training data,  
dataY=pd.read_csv('~/Documents/data/Morelia_train_Y.csv')
dataX=pd.read_csv('~/Documents/data/Morelia_train_X.csv')
lat=dataX.lon.to_numpy()
lon=dataX.lat.to_numpy()
label=dataY.incremento_urbano.to_numpy()


plotwidth=2000
plotheight=2000
plotdpi=300

#Distance to urban land
feature=dataX.dist_urbano.to_numpy()
label=dataY.incremento_urbano.to_numpy()
importlib.reload(toc)
T=toc.PATOC(rank=feature,groundtruth=label,smoothingMethod="ANN")
T.featureName='Distance to Urban Land'

#TOC
T.plot(filename='TOC-dist2UL.png',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['intersections'])
T.plot(kind='piecewise',filename='PWTOC-dist2UL.png',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['intersections'])

#Cumulative distribution Functions
T.plot(kind='CPF-Rank|Presence',filename='CPF-Rank2Pres-dist2UL.png',title='Cum. Dist. Funct. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='CPF-Rank',filename='CPF-Rank-dist2UL.png',title='Cum. Dist. Funct. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='CPF-Presence|Rank',filename='CPF-Pres2Rank-dist2UL.png',title='Cum. Dist Funct. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])

#Density distribution Functions
T.plot(kind='PF-Rank|Presence',filename='PF-Rank2Pres-dist2UL.png',title='Prob. Density Func. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='PF-Rank',filename='PF-Rank-dist2UL.png',title='Prob. Density Func. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='PF-Presence|Rank',filename='PF-Pres2Rank-dist2UL.png',title='Prob. Density Func. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])

#Derivative of the density distribution Functions
T.plot(kind='DPF-Rank|Presence',filename='DPF-Rank2Pres-dist2UL.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])
T.plot(kind='DPF-Rank',filename='DPF-Rank-dist2UL.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])
T.plot(kind='DPF-Presence|Rank',filename='DPF-Pres2Rank-dist2UL.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])

#Join probability function Rank & Presence
T.plot(kind='JPF',filename='JPF-PresRank-dist2UL.png',title="Join prob. Rank & Presence",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])

#Smoothed versions of the probability functions
T.plot(kind='smoothPF-Rank|Presence',filename='smoothPF-Rank2Pres-dist2UL.png',title='Smoothed Prob. Funct. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothPF-Rank',filename='smoothPF-Rank-dist2UL.png',title='Smoothed Prob. Funct. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothPF-Presence|Rank',filename='smoothPF-Pres2Rank-dist2UL.png',title='Smoothed Prob. Funct. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])

#Smoothed versions of the Derivative of the (density) probability functions
T.plot(kind='smoothDPF-Rank|Presence',filename='smoothDPF-Rank2Pres-dist2UL.png',title='Smoothed Prob. Funct. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothDPF-Rank',filename='smoothDPF-Rank-dist2UL.png',title='Smoothed Prob. Funct. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothDPF-Presence|Rank',filename='smoothDPF-Pres2Rank-dist2UL.png',title='Smoothed Prob. Funct. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])

#Probability of Presence|Rank to spatial coordinates
prob=T.rank2prob(feature)
T.rasterize(prob,lat,lon)
T.plot(kind='Raster',title='Prob. Presence of ULC given Distance to UL',filename='rasterP-dist2UL.png',height=plotheight,width=plotwidth,dpi=plotdpi)

#Simulation of land change
sim=T.simulate(feature,int(T.np))
T.rasterize(sim,lat,lon)
T.plot(kind='Raster',title='Simulation of ULC using Prob. of Dist. to UL',filename='rasterSim-dist2UL.png',height=plotheight,width=plotwidth,dpi=plotdpi,options=['binary'])
probDist2UL=prob


#Verification, computing TOC and PF from the simulation, it must approximate the original TOC
label=sim
importlib.reload(toc)
Tsim=toc.PATOC(rank=feature,groundtruth=label,smoothingMethod="wmeans")
Tsim.plot(filename='sim-TOC-dist2UL.png',title='Simulation TOC',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['intersections'])
#Density distribution Functions
Tsim.plot(kind='PF-Rank|Presence',filename='sim-PF-Rank2Pres-dist2UL.png',title='Simulation Density Func. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
Tsim.plot(kind='PF-Rank',filename='sim-PF-Rank-dist2UL.png',title='Simulation Prob. Density Func. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
Tsim.plot(kind='PF-Presence|Rank',filename='sim-PF-Pres2Rank-dist2UL.png',title='Simulation Prob. Density Func. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])






#Slope
feature=dataX.pendiente.to_numpy()
importlib.reload(toc)
T=toc.PATOC(rank=feature,groundtruth=label)
T.featureName='Terrain slope'



T.plot(filename='TOC-slope.png',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['intersections'])

#Cumulative distribution Functions
T.plot(kind='CPF-Rank|Presence',filename='CPF-Rank2Pres-slope.png',title='Cum. Dist. Funct. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='CPF-Rank',filename='CPF-Rank-slope.png',title='Cum. Dist. Funct. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='CPF-Presence|Rank',filename='CPF-Pres2Rank-slope.png',title='Cum. Dist Funct. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])

#Density distribution Functions
T.plot(kind='PF-Rank|Presence',filename='PF-Rank2Pres-slope.png',title='Mass Prob. Func. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='PF-Rank',filename='PF-Rank-slope.png',title='Mass Prob. Func. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='PF-Presence|Rank',filename='PF-Pres2Rank-slope.png',title='Mass Prob. Func. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])


#Derivative of the density distribution Functions
T.plot(kind='DPF-Rank|Presence',filename='DPF-Rank2Pres-slope.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])
T.plot(kind='DPF-Rank',filename='DPF-Rank-slope.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])
T.plot(kind='DPF-Presence|Rank',filename='DPF-Pres2Rank-slope.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])

#Join probability function Rank & Presence
T.plot(kind='JPF',filename='JPF-PresRank-slope.png',title="Join prob. Rank & Presence",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])

#Smoothed versions of the probability functions
T.plot(kind='smoothPF-Rank|Presence',filename='smoothPF-Rank2Pres-slope.png',title='Smoothed Prob. Funct. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothPF-Rank',filename='smoothPF-Rank-slope.png',title='Smoothed Prob. Funct. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothPF-Presence|Rank',filename='smoothPF-Pres2Rank-slope.png',title='Smoothed Prob. Funct. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])

#Smoothed versions of the Derivative of the (density) probability functions
T.plot(kind='smoothDPF-Rank|Presence',filename='smoothDPF-Rank2Pres-slope.png',title='Smoothed Prob. Funct. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothDPF-Rank',filename='smoothDPF-Rank-slope.png',title='Smoothed Prob. Funct. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
T.plot(kind='smoothDPF-Presence|Rank',filename='smoothDPF-Pres2Rank-slope.png',title='Smoothed Prob. Funct. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])


prob=T.rank2prob(feature)
T.rasterize(prob,lat,lon)
T.plot(kind='Raster',title='Prob. of Presence given Terrain Slope',filename='rasterP-slope.png',height=plotheight,width=plotwidth,dpi=plotdpi)
sim=T.simulate(feature,int(T.np/2))
T.rasterize(sim,lat,lon)
T.plot(kind='Raster',title='Simulation of ULC using prob. of Terrain Slope',filename='rasterSim-slope.png',height=plotheight,width=plotwidth,dpi=plotdpi,options=['binary'])
probSlope=prob




#Verification, computing TOC and PF from the simulation, it must approximate the original TOC
label=sim
importlib.reload(toc)
Tsim=toc.PATOC(rank=feature,groundtruth=label,smoothingMethod="wmeans")
Tsim.plot(filename='sim-TOC-slope.png',title='Simulation TOC',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['intersections'])
#Density distribution Functions
Tsim.plot(kind='PF-Rank|Presence',filename='sim-PF-Rank2Pres-slope.png',title='Simulation Density Func. of Rank given Presence', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
Tsim.plot(kind='PF-Rank',filename='sim-PF-Rank-slope.png',title='Simulation Prob. Density Func. of Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])
Tsim.plot(kind='PF-Presence|Rank',filename='sim-PF-Pres2Rank-slope.png',title='Simulation Prob. Density Func. of Presence given Rank', height=plotheight, width=plotwidth, dpi=plotdpi, xlabel="default", ylabel="default", autodpi=False, labelsize=8, options=['vlines','quartiles'])











importlib.reload(toc)
feature=dataX.costo.to_numpy()
T=toc.TOCPF(rank=feature,groundtruth=label)
T.featureName='Traveling time to the city center'
T.plot(filename='TOC-ttime.png',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['intersections'])
#T.plot(kind='PF',filename='PF-ttime-before-smoothing.png',title='Probability Density Function cond. to Presence',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines','quartiles'])
#T.plot(kind='DPF',filename='DPF-ttime-before-smoothing.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])
T.plot(kind='CDF',filename='CDF-ttime.png',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8)
T.plot(kind='PF',filename='PF-ttime.png',title='Probability Density Function cond. to Presence',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines','quartiles'])
T.plot(kind='DPF',filename='DPF-ttime.png',title="First Derivative of the PDF",height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines'])
T.plot(kind='smoothPF',filename='smoothPF-ttime.png',title='Density Probability Function cond. to Presence',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines','quartiles'])
T.plot(kind='smoothDPF',filename='smoothDPF-ttime.png',title='Difference of the DPF cond. to Presence',height=plotheight,width=plotwidth,dpi=plotdpi,xlabel="default",ylabel="default",autodpi=False,labelsize=8,options=['vlines','quartiles'])


prob=T.rank2prob(feature)
T.rasterize(prob,lat,lon)
T.plot(kind='raster',TOCname='Cond. Prob. of Traveling Time given Presence of ULC',filename='rasterP-ttime.png',height=plotheight,width=plotwidth,dpi=plotdpi)
sim=T.simulate(feature,T.np)
T.rasterize(sim,lat,lon)
T.plot(kind='raster',TOCname='Simulation of ULC using prob. of Traveling Time',filename='rasterSim-ttime.png',height=plotheight,width=plotwidth,dpi=plotdpi,options=['binary'])
probTravTime=prob

prob=probDist2UL*probTravTime*probSlope
T.rasterize(prob,lat,lon)
T.plot(kind='raster',TOCname='',filename='rasterP-combined.png',height=plotheight,width=plotwidth,dpi=plotdpi)
sim=T.simulate(feature,T.np)
T.rasterize(sim,lat,lon)
T.plot(kind='raster',TOCname='Simulation of ULC using combined prob',filename='rasterSim-combined.png',height=plotheight,width=plotwidth,dpi=plotdpi,options=['binary'])













####################################################################
##--Time computation  wmeans--##
feature=dataX.costo.to_numpy()
importlib.reload(toc)
tstart = time.time()
T=toc.TOCPF(rank=feature,groundtruth=label,smoothingMethod='wmeans')
tend = time.time()
print("Elapsed time for computing the TOC, find an adequate discretization, computing CDF,PF, DPF and their smoothed versions with moving windows:",tend-tstart, "seconds")



####################################################################
##--Time computation ANN--##
tstart = time.time()
T=toc.TOCPF(rank=feature,groundtruth=label,smoothingMethod='ANN')
tend = time.time()
print("Elapsed time for computing the TOC, find an adequate discretization, computing CDF,PF, DPF and their smoothed versions with ANN:",tend-tstart, "seconds")


#####################################################################
##Main variables and methods###
T.plot()
T.kind #Discrete or continuous
T.np #Number of data labeled 1
T.ndata #Data size of the orginal data set and the TOC
