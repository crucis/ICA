import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.cluster.vq import whiten
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import entropy, chi2_contingency, shapiro, norm

def kde_entropy(x, bandwidth = 'silverman', **kwargs):
    from statsmodels.nonparametric.kde import KDEUnivariate
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(np.reshape(x, (-1,1)))
    kde.fit(bw = bandwidth, kernel = 'gau', fft = True, **kwargs)
    
    return kde.entropy

def KLdivergence(x, n_bins = None):
 
    if n_bins is None:
        bins_gaussian = len(np.histogram(np.random.normal(loc = np.mean(x), scale = np.std(x), 
                                                          size = len(x)), bins = 'fd')[0])
        bins_x = len(np.histogram(x, bins = 'fd')[0])
        n_bins = min(bins_gaussian, bins_x)
    #gaussian_hist = norm.pdf(np.linspace(-5, 5, n_bins), loc = np.mean(x), scale = np.std(x))
    gaussian_hist = np.histogram(np.random.normal(loc = np.mean(x), scale = np.std(x), 
                                                          size = len(x)), bins = n_bins)[0]
    x_hist = np.histogram(x, bins = n_bins)[0]
    
    return entropy(x_hist, gaussian_hist+1e-40)

def calculateNegentropy(x, kindOfNegentropy = 'empirical', n_bins = None):
   
    if kindOfNegentropy == 'KDE':
        return np.log(np.std(x)*np.sqrt(2*np.pi*np.exp(1))) - kde_entropy(x)
    
    elif kindOfNegentropy == 'empirical':
        if n_bins is None:
            bins_gaussian = len(np.histogram(np.random.normal(loc=np.mean(x), scale=np.std(x), 
                                                              size = len(x)), bins = 'fd')[0])
            bins_x = len(np.histogram(x, bins = 'fd')[0])
            n_bins = min(bins_gaussian, bins_x)
        gaussian_hist = np.histogram(np.random.normal(loc=np.mean(x), scale=np.std(x), 
                                                      size = len(x)), bins = n_bins)[0]
        x_hist = np.histogram(x, bins = n_bins)[0]
        negentropy = entropy(gaussian_hist) - entropy(x_hist)
        if negentropy < 0:
            negentropy = 0
        return negentropy
    
    else:
        print('Not implemented')
        return None
    
def resultsTable(y, n_bins = None, negentropyType = 'empirical'):
    import tabulate
    from IPython.display import HTML, display
    
    data = np.array([["Data", "Negentropy", "KL Divergence", "Shapiro-Wilk test W", "Shapiro-Wilk test P_value"]])
    for i, y_i in enumerate(y):
        shapiro_yi = shapiro(y_i)
        new_row = np.array([["%d"%i, "%.04f"%calculateNegentropy(y_i, n_bins = n_bins, 
                                                                 kindOfNegentropy = negentropyType),
                             "%0.4f"%KLdivergence(y_i, n_bins = n_bins),
                             "%.04f"%shapiro_yi[0], "%.04E"%shapiro_yi[1]]])
        data = np.concatenate((data, new_row))
    display(HTML(tabulate.tabulate(data, tablefmt = 'html', headers = 'firstrow')))
    return None

def mutualInformation_matrix(signal, kde=False, n_bins=None):
    from statsmodels.sandbox.distributions.mv_measures import mutualinfo_kde, mutualinfo_binned
    rows, cols = signal.shape
    mat = np.zeros((rows, rows))
    np.fill_diagonal(mat, 1)
    # Upper diagonal
    for r in range(rows):
        for c in range(r, rows):
            if r == c:
                continue
            p = signal[r].flatten() + 1e-12
            q = signal[c].flatten() + 1e-12
            if kde:
                mi = mutualinfo_kde(p, q)
            else:
                if n_bins is None:
                    p_bins = len(np.histogram(p, bins='fd')[0])
                    q_bins = len(np.histogram(q, bins='fd')[0])
                    n_bins = min(p_bins, q_bins)
                elif n_bins == 'auto':
                    qs = np.sort(q)
                    ps = np.sort(p)
                    qbin_sqr = np.sqrt(5./cols)
                    quantiles = np.linspace(0, 1, 1./qbin_sqr)
                    quantile_index = ((cols-1)*quantiles).astype(int)
                    #move edges so that they don't coincide with an observation
                    shift = 1e-6 + np.ones(quantiles.shape)
                    shift[0] -= 2*1e-6
                    q_bins = qs[quantile_index] + shift
                    p_bins = ps[quantile_index] + shift
                    n_bins = min(p_bins, q_bins)

                #mi = mutualinfo_binned(p, q, n_bins)[0]
                
                fx, binsx = np.histogram(p, bins = p_bins)
                fy, binsy = np.histogram(q, bins = q_bins)
                fyx, binsy, binsx = np.histogram2d(q, p, bins = (binsy, binsx))

                pyx = fyx * 1. / cols
                px = fx * 1. / cols
                py = fy * 1. / cols


                mi_obs = pyx * (np.log(pyx + 1e-12) - np.log(py + 1e-12)[:,None] - np.log(px + 1e-12))
                mi_obs[np.isnan(mi_obs)] = 0
                mi_obs[np.isinf(mi_obs)] = 0
                mi = mi_obs.sum()
            if np.isnan(mi):
                mi = 1
            elif mi > 1:
                mi = 1
            mat[r][c] = mi
            mat[c][r] = mi
            
    return mat

def plot_MutualInformation(mixtures, y, KDE = False, nbins = None):
    import seaborn as sb

    
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    
    sb.heatmap(mutualInformation_matrix(mixtures, kde=KDE, n_bins = nbins), ax=axs[0], annot=True, cmap = 'YlGnBu', vmin = 0, vmax = 1)
    axs[0].set_title('Mutual Information: mixed signals')
    
    sb.heatmap(mutualInformation_matrix(y, kde = KDE, n_bins = nbins), ax=axs[1], annot=True, cmap = 'YlGnBu', vmin = 0, vmax = 1)
    axs[1].set_title('Mutual Information: outputs')
    
    return None



