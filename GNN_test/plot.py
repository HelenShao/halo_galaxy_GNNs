import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"


def denormalize(true, pred, errors, min, max):

    true = min + true*(max - min)
    pred = min + pred*(max - min)
    errors = errors*(max - min)
    return true, pred, errors

slurm, trial = 2, 20
pred  = np.loadtxt("slurm%d/predicted_%d.txt"%(slurm, trial))[:,0]
true  = np.loadtxt("slurm%d/true_%d.txt"%(slurm, trial))
errors = np.loadtxt("slurm%d/predicted_%d.txt"%(slurm, trial))[:,1]

# De-normalize pred/true using Omega_M min and max values
min, max = 0.1, 0.5
true, pred, errors = denormalize(true, pred, errors, min, max)

mse = np.mean((true - pred)**2)

true_sort = np.sort(true)

# Remove normalization of cosmo parameters
def denormalize(true, pred, errors, min, max):

    true = min + true*(max - min)
    pred = min + pred*(max - min)
    errors = errors*(max - min)
    return true, pred, errors

# Scatter plot of true vs predicted cosmological parameter
def scatterplot(slurm, trial):

    figscat, axscat = plt.subplots(figsize=(6,5), dpi=1200)
    # suite, simset = hparams.simsuite, hparams.simset
    col = "forestgreen"

    # Load true values and predicted means and standard deviations
    pred  = np.loadtxt("slurm%d/predicted_%d.txt"%(slurm, trial))[:,0]
    true  = np.loadtxt("slurm%d/true_%d.txt"%(slurm, trial))
    errors = np.loadtxt("slurm%d/predicted_%d.txt"%(slurm, trial))[:,1]
    
    sort_ind = np.argsort(true)
    pred = pred[sort_ind][::10]
    errors = errors[sort_ind][::10]
    true = true[sort_ind][::10]

    """# There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    errors = errors[1:]"""

    # De-normalize pred/true using Omega_M min and max values
    min, max = 0.1, 0.5
    true, pred, errors = denormalize(true, pred, errors, min, max)

    # Compute the number of points lying within 1 or 2 sigma regions from their uncertainties
    cond_success_1sig, cond_success_2sig = np.abs(pred-true)<=np.abs(errors), np.abs(pred-true)<=2.*np.abs(errors)
    tot_points = 1
    successes1sig, successes2sig = pred[cond_success_1sig].shape[0], pred[cond_success_2sig].shape[0]
    
    mse = np.mean((true - pred)**2)
   
    # Compute the accuracy metrics
    r2 = r2_score(true, pred)
    err_rel = np.mean(np.abs((true - pred)/(true)), axis=0)
    chi2s = (pred-true)**2./errors**2.
    chi2 = chi2s[chi2s<1.e1].mean()    # Remove some outliers which make explode the chi2
    print("R^2={:.2f}, Relative error={:.2e}, Chi2={:.2f}".format(r2, err_rel, chi2))
    print("A fraction of succeses of", successes1sig/tot_points, "at 1 sigma,", successes2sig/tot_points, "at 2 sigmas")
    
    mse = np.mean((true - pred)**2)
    
    # Sort by true value
    indsort = true.argsort()
    pred, true, errors = pred[indsort], true[indsort], errors[indsort]

    # Compute mean and std region within several bins
    truebins, binsize = np.linspace(true[0], true[-1], num=10, retstep=True)
    means, stds = [], []
    for i, bin in enumerate(truebins[:-1]):
        cond = (true>=bin) & (true<bin+binsize)
        outbin = pred[cond]
        if len(outbin)==0:
            outmean, outstd = np.nan, np.nan    # Avoid error message from some bins without points
        else:
            outmean, outstd = outbin.mean(), outbin.std()
        means.append(outmean); stds.append(outstd)
    means, stds = np.array(means), np.array(stds)

    # Plot predictions vs true values
    truemin, truemax = min-0.05, max+0.05
    #axscat.plot([truemin, truemax], [0., 0.], "r-")
    #axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    axscat.plot([truemin, truemax], [truemin, truemax], "k-")
    axscat.errorbar(true, pred, yerr=errors, color=col, marker="o", ls="none", markersize=1, elinewidth=0.8, zorder=10)

    # Legend
    par = "\t"+r"$\Omega_{\rm m}$"
    #par = "\t"+r"$\sigma_8$
    leg = "$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.2f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)+"\n"+"RMSE={:.2e}".format(np.sqrt(mse))
    #leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.2e}".format(err_rel)
    #leg = par+"\n"+"$R^2$={:.2f}".format(r2)
    #leg = par
    at = AnchoredText(leg, frameon=True, loc="lower right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    # Labels etc
    axscat.set_xlim([truemin, truemax])
    axscat.set_ylim([truemin, truemax])
    """if testsuite:
        axscat.set_ylim([truemin, truemax+0.1])"""
    axscat.set_ylabel(r"Prediction")
    axscat.set_xlabel(r"Truth")
    axscat.grid()
    
    axscat.set_title("Gadget: GNN")
    
    fname = "GNN_2channels.pdf"
    plt.savefig(fname, dpi=1200, pad_inches=0.1, bbox_inches="tight")
    #plt.close(figscat)


