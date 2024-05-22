###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#-----------------------------------------------------------------------------------------------
#Section: Correlation
#-----------------------------------------------------------------------------------------------

def CorrelationHeatmap(ds,MetricNames, scoreColumns):

    #To Numpy
    Data = ds[MetricNames].to_numpy()

    #Compute correlation
    correlations = []
    for al in scoreColumns:
        correlations.append(np.abs(np.corrcoef(Data, ds[al].to_numpy(), rowvar=False)[-1, :-1]))
    #As numpy array
    correlations = np.round(np.array(correlations),3)


    #Prepare for plotting
    fig, ax = plt.subplots()
    heatmap = ax.imshow(correlations)

    #Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(MetricNames)), labels=MetricNames)
    ax.set_yticks(np.arange(len(scoreColumns)), labels=scoreColumns)

    #Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    #Loop over data dimensions and create text annotations
    for i in range(len(scoreColumns)):
        for j in range(len(MetricNames)):
            _ = ax.text(j, i, correlations[i, j], ha="center", va="center", color="w")

    ax.set_title("Correlation heatmap for " + str(ds.shape[0]) + " circuits")
    fig.tight_layout()
    fig.colorbar(heatmap)
    plt.show()

#-----------------------------------------------------------------------------------------------
#Section: Compare CrossEntropy of best model (per gate count) to model with best (given) metric
#-----------------------------------------------------------------------------------------------
def CrossEntropyCompare(ds, mLbl, avgCount = 5):
        
    #Labels we will be using
    gLbl = "Number of gates"
    tLbl = "Cross Entropy"    

    #Create temp dataframe
    dfplot = pd.DataFrame({
        gLbl: ds[gLbl].tolist(),
        tLbl: ds[tLbl].tolist(),
        mLbl: ds[mLbl].tolist()        
    })

    #Group by (discrete) gate count
    gateGrouped = dfplot.groupby(gLbl)

    #Get indexes of highest score per (discrete) gate count
    gBest = dfplot.iloc[gateGrouped[tLbl].idxmin(),[0,1]].to_numpy()

    #Get indexes of highest metric per (discrete) gate count
    mBest = dfplot.iloc[gateGrouped[mLbl].idxmin(),[0,1]].to_numpy()

    #Get average of score for top models, based on metrics, per gate count
    topMetricAvg = []
    for key, item in gateGrouped:
        topMetricAvg.append(gateGrouped.get_group(key).nsmallest(avgCount, mLbl)[tLbl].mean())

    #Plot
    dfplot.plot.scatter(gLbl, tLbl,     label = 'All models', c=dfplot[mLbl])
    plt.plot(gBest[:,0], gBest[:,1],    label = 'Model with best score',  c='red')
    plt.plot(mBest[:,0], mBest[:,1],    label = 'Model with best metric', c='blue')
    plt.plot(mBest[:,0], topMetricAvg,  label = 'Average of top ' + str(avgCount) + ' models', c='green')

    #Labels
    plt.title(tLbl + " comparison, per '" + gLbl + "', for metric: '" + mLbl + "'")
    plt.xlabel(gLbl)
    plt.ylabel(tLbl)
    plt.legend()

    #Show
    plt.show()



#-----------------------------------------------------------------------------------------------
#Section: Scatter plots with colorbar & fittted polynomial
#-----------------------------------------------------------------------------------------------

#How to use:

#   w = "Parameterized gates"   #X-axis
#   t = "Training time"         #Y-axis
#   c = "Number of gates"       #Make colorchart from this column

#   Optional: 
        #PolyFit                Fit a polynomial to data
        #bin = None             Split target into chunks

#Call        
#   cpPlot(ds, w, t, c)

#Plot function definition
def cpPlot(ds, w, t, c = None, PolyFit = True, bin = None):
  
    if(bin is None):
        target = ds[t]
    else:
        target = ds[t] // bin

    if(c is None):
        __innerPlot(ds[w], target, w, t, PolyFit, None, "")
    else:
        __innerPlot(ds[w], target, w, t, PolyFit, ds[c],c)

def __innerPlot(x, y, xLabel, yLabel, PolyFit, colorData, colorTitle):

    #Fit polynomial
    if(PolyFit):
        a, b = np.polyfit(x, y, 1)
        plt.plot(x, a*x+b)

    #Scatter points
    if(colorData is None):
        plt.scatter(x, y)        
    else:
        plt.scatter(x, y, c = colorData)
        plt.colorbar(label=colorTitle)
    
    #Labels
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    #Display
    plt.show()
