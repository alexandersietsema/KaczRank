import numpy as np
from scipy.stats import rankdata
from scipy.stats import gmean
from tqdm.auto import tqdm

def get_ranking(vec):
    """Return the ranking associated with a vector of ratings."""
    return (rankdata(vec)-1).astype(int)

def hamdist(v,w):
    """Returns the Hamming distance between two vectors."""
    return np.sum(np.abs(v-w) > 1e-10)

def mode_rows(a):
    """Returns the most frequent row in an array."""
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return count.max(), most_frequent_row

def kaczrank(A,eps,x_init,n_iters = 1000, verbose=True, check_convergence=True, return_iters=True):
    """
    The Kaczrank method as defined in Algorithm 1.
    
    Parameters:
    A: array, array of pairwise comparisons
    eps: float, relaxation parameter as in Section 5
    x_init: vector, initial iterate
    n_iters: int, number of iterations to run
    verbose: bool, if true, returns all iterates and rankings across all iterations
    check_convergence: if true, stop iteration once true ranking (0,1,... in order) has been reached
    return_iters: bool, if true, return final iterate and number of iterations
     
    Returns:
    x: vector, final iterate
    
    if verbose:
    iterates: array, array of iterates
    rankings: array, array of rankings for each iterate
    
    if return_iters:
    x: vector, final iterate
    i: int, number of iterations
    """
    
    m,n = np.shape(A)
    x = x_init
    
    if verbose:
        iterates = np.zeros((n_iters+1, n))
        rankings = np.zeros((n_iters+1, n))
        iterates[0,:] = x
        rankings[0,:] = get_ranking(x)
    
    
    for i in range(n_iters):
        j = np.random.randint(0,m)
        
        res = np.dot(A[j,:], x)
        if res >= 0:
            x = x - 0.5*(res+eps)*A[j,:]
        
        if verbose:
            iterates[i+1,:] = x
            rankings[i+1,:] = get_ranking(x)
            
        if check_convergence:
            if np.all(x[1:] > x[:-1]):
                break
    
    if verbose:
        return iterates, rankings
    if return_iters:
        return x, i
    return x



def kr_itercount(A, eps, x_init):
    """Counts the number of iterations until convergence for Kaczrank.
    
    Parameters:
    A: array, array of pairwise comparisons
    eps: float, relaxation parameter as in Section 5
    x_init: vector, initial iterate
    
    Returns:
    ct: int, number of iterations to convergence.
    """
    m,n = np.shape(A)
    x = x_init
    ct = 0
    while np.sum(A @ x > 0) > 0:
        j = np.random.randint(0, m)
        res = np.dot(A[j, :], x)
        if res > 0:
            x = x - 0.5 * (res + eps) * A[j, :]
        ct += 1
    return ct

def noisykaczrank(A,p,eps,x_init,n_iters = 1000):
    """
    The KaczRank method as defined in Algorithm 1 applied to noisy data.
    
    Parameters:
    A: array, array of pairwise comparisons
    p: float, probability of flipped comparison as in Section 4. Note that we assume that A is input without noise, so we add
       noise to the observations as part of the algorithm rather than when constructing A.
    eps: float, relaxation parameter as in Section 5
    x_init: vector, initial iterate
    n_iters: int, number of iterations to run
   
    Returns:
    iterates: array, array of iterates
    rankings: array, array of rankings for each iterate
    """
    m,n = np.shape(A)
    x = x_init
    iterates = np.zeros((n_iters+1, n))
    rankings = np.zeros((n_iters+1, n))
    iterates[0,:] = x
    rankings[0,:] = get_ranking(x)
    for i in tqdm(range(n_iters)):
        j = np.random.randint(0,m)
        u = np.random.random()
        if u <= p:
            t = -1
        else:
            t = 1
        res = np.dot(t*A[j,:], x)
        if res > 0:
            x = x - 0.5*(res + eps)*t*A[j,:]
        iterates[i+1,:] = x
        rankings[i+1,:] = get_ranking(x)
    return iterates, rankings


def cautiousrank(A,p,eps,x_init,alpha, n_iters = 1000, verbose=True):
    """
    The CautiousRank method as defined in Algorithm 2.
    
    Parameters:
    A: array, array of pairwise comparisons
    p: float, probability of flipped comparison as in Section 4. Note that we assume that A is input without noise, so we add
       noise to the observations as part of the algorithm rather than when constructing A.
    eps: float, relaxation parameter as in Section 5
    x_init: vector, initial iterate
    n_iters: int, number of iterations to run
    verbose: bool, if true, returns all iterates and rankings across all iterations
    
    Returns:
    x: vector, final iterate
    
    if verbose:
    iterates: array, array of iterates
    rankings: array, array of rankings for each iterate
    """
    m,n = np.shape(A)
    x = x_init
    
    if verbose:
        iterates = np.zeros((n_iters+1, n))
        rankings = np.zeros((n_iters+1, n))
        iterates[0,:] = x
        rankings[0,:] = get_ranking(x)
    
    for i in range(n_iters):
        j = np.random.randint(0, m)
        u = np.random.random()
        if u <= p:
            t = -1
        if u > p:
            t = 1
        res = np.dot(t * A[j, :], x)
        if res > 0:
            y = x - 0.5*(res + eps)*t*A[j,:]
            if hamdist(get_ranking(y), get_ranking(rankings[i, :])) < alpha:
                x = y
        
        if verbose:
            iterates[i+1,:] = x
            rankings[i+1,:] = get_ranking(x)
    if verbose:
        return iterates, rankings
    
    return x

#plotting code
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from zlib import crc32
from itertools import cycle
from importlib import reload
reload(matplotlib)
matplotlib.use('Agg')
sns.set_style("whitegrid")
# sns.set_context("talk")

"""
For details on the params below, see the matplotlib docs:
https://matplotlib.org/users/customizing.html
"""

rcParams["axes.edgecolor"] = "0.6"
# rcParams['figure.dpi'] = 200
# rcParams['figure.figsize'] = [12,4]
rcParams['figure.figsize'] = [6,4]
rcParams['font.family'] = 'serif'
rcParams["mathtext.fontset"] = "dejavuserif"
# rcParams['font.size'] = 16.0
rcParams["grid.color"] = "0.85"
# rcParams['savefig.dpi'] = 300
# rcParams["legend.columnspacing"] *= 0.8
rcParams["legend.edgecolor"] = "0.6"
rcParams["legend.framealpha"] = "1"
rcParams["legend.frameon"] = True
rcParams["legend.handlelength"] *= 1.5
# rcParams['legend.numpoints'] = 2
#rcParams['text.usetex'] = True
rcParams['xtick.major.pad'] = 3
rcParams['ytick.major.pad'] = 2

# plt.rc('text.latex', preamble=[r'\usepackage{amsmath}',r'\usepackage{amssymb}'])

COLORS = list(sns.color_palette())
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ['D', 'o', 'X', '*', '<', 'd', '>', 's', 'v']

def _compute_markevery(n_points, n_lines=1, line_i=0, n_markers=10):
    """Compute the markevery parameter for ``plt.plot``.

    Try to avoid stacking markers on top of eachother, don't use too many markers, etc.

    n_points : int
        How many datapoints are being plotted in the line.
    n_lines : int
        How many lines are there going to be on your figure?
    line_i : int
        Of the ``n_lines`` lines on your figure, which one is this?
    n_markers : int
        How many markers would you like on each line?
    """
    # TODO: compute offset from the number of things to plot instead
    if n_points > n_markers:
        markevery = int(n_points / n_markers)
        return (int(markevery * line_i / n_lines), markevery)
    else:
        return 1

def _stack_data(data_list):
    """If data is not same length, chop to shortest data, then stack."""
    truncate_at = min(len(data) for data in data_list)
    return np.stack([data[:truncate_at] for data in data_list])

# TODO: Automate the color, linestyle, and marker stuff.

def plot_average_w_ci(data, x=None, median=True, n_lines=3, line_i=0, n_markers=10, label=None,
                      color=None, linestyle=None, marker=None, percentile = 25, ax=None):
    """Plot an averaged line with a shaded confidence interval using data from multiple trials.

    data: list[list[float]]
        The data to be plotted. One element for each trial.
        Trials will be truncated to the shortest number of iterations.
    x: list[int]
        x-positions of each datapoint to plot. Default range(n_iters)
    n_lines: int
        The number of lines in the plot.
    line_i: int
        If you are plotting one of several things, which one is this?
    label: string
        Label for the line to be plotted. Passed to `plt.plot`.
    color: string
        Color of the line to be plotted. Defaults to picking from a static array of colors based on line_i.
    linestyle: string
        Linestyle of the line to be plotted. Defaults to picking from a static array of linestyles based on line_i.
    marker: string
        Marker of the line to be plotted. Defaults to picking from a static array of markers based on line_i.
    n_markers: int
        Number of markers to use.
    percentile: int
        50 +- percentile for shading
    """
    data = _stack_data(data)
    n_iters = data.shape[1]

    if median:
        means = np.median(data, axis=0)
    else:
        means = np.mean(data, axis=0)

    markevery = _compute_markevery(n_points=len(means),
                                  n_lines=n_lines,
                                  line_i=line_i,
                                  n_markers=n_markers)

    if color is None:
        color = COLORS[line_i % len(COLORS)]
    if linestyle is None:
        linestyle = LINESTYLES[line_i % len(LINESTYLES)]
    if marker is None:
        marker = MARKERS[line_i % len(MARKERS)]

    if x is None:
        x = range(n_iters)
        
    if ax:
        ax.plot(x,
             means,
             label=label,
             color=color,
             linestyle=linestyle,
             marker=marker,
             markevery=markevery)
    else:
        plt.plot(x,
             means,
             label=label,
             color=color,
             linestyle=linestyle,
             marker=marker,
             markevery=markevery)

    if percentile > 0:
        lower_ci = []
        upper_ci = []
        for mean,i in zip(means,range(n_iters)):
            lower_ci.append(np.percentile(data[:,i],50-percentile))
            upper_ci.append(np.percentile(data[:,i],50+percentile))
        if ax:
            ax.fill_between(x, lower_ci, upper_ci,
                         alpha=0.25, color=color)
        else:
        
            plt.fill_between(x, lower_ci, upper_ci,
                         alpha=0.25, color=color)





