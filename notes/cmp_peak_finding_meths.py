#! /usr/bin/env python

import matplotlib.pyplot as plt
import ibdutils.utils.ibdutils as ibdutils

fn1 = "../esea_ibd_unrel_diploid_xirsfilt.ibdobj.gz"
fn2 = "../waf_ibd_unrel_diploid_xirsfilt.ibdobj.gz"
# fn1 = "../../posseleff_empirical/run3/results/05_ibdanalysis/ESEA/ifm_input/ESEA_ifm_orig.ibdobj.gz"
# fn2 = "../../posseleff_empirical/run3/results/05_ibdanalysis/WAF/ifm_input/WAF_ifm_orig.ibdobj.gz"
# fn = "../posseleff_empirical/run3/results/05_ibdanalysis/WAF/ifm_input/WAF_ifm_orig.ibdobj.gz"


def proc_ibd(fn, pf_method='std', xc_method='fdr_bh', min_xirs_hits=0):
    ibd0 = ibdutils.IBD.pickle_load(fn)
    ibd0._peaks_df = None
    ibd0._peaks_df_bk = None
    ibd0.find_peaks(method=pf_method)
    ibd0._xirs_df.rename(columns={'Pvalue': 'RawPvalue'}, inplace=True)
    ibd0.calc_xirs([], multitest_correction=xc_method, only_adj_pvalues=True)

    ibd0._peaks_df_bk = ibd0._peaks_df.copy()
    if min_xirs_hits != 0:
        ibd0.filter_peaks_by_xirs(ibd0._xirs_df, min_xirs_hits=min_xirs_hits)
    ibd0._xirs_df = (ibd0._xirs_df.merge(
        ibd0._genome._chr_df[['Chromosome', 'GwChromStart']],
        on='Chromosome', how='left')
        .assign(GwPos=lambda x: x.Pos + x.GwChromStart)
        .drop(labels='GwChromStart', axis=1)
    )
    return ibd0


def plot_cmp_std_iqr(fn, label='esea'):
    # compare peak finding methods
    fig, ax = plt.subplots(figsize=(12, 4), nrows=2, constrained_layout=True)

    ibd = proc_ibd(fn, pf_method='std', xc_method='fdr_bh', min_xirs_hits=0)
    # ibd._peaks_df = ibd._peaks_df_bk.copy()
    ibd.plot_coverage(ax=ax[0], which='unfilt')
    ax[0].set_ylim(-1000, 20000)
    ax[0].set_title("peak identification: std")
    ibd.plot_drg_annotation(ax=ax[0])

    ibd = proc_ibd(fn, pf_method='iqr', xc_method='fdr_bh', min_xirs_hits=0)
    # ibd._peaks_df = ibd._peaks_df_bk.copy()
    ibd.plot_coverage(ax=ax[1], which='unfilt')
    ax[1].set_ylim(-1000, 20000)
    ax[1].set_title("peak identification: iqr")
    ibd.plot_drg_annotation(ax=ax[1])

    fig.savefig(f"./imgs/std_vc_iqr_{label}.pdf")


def plot_cmp_xirs_correction(fn, label='esea', min_xirs_hits=3):
    # compare peak finding methods
    fig, axes = plt.subplots(figsize=(12, 6), nrows=3, constrained_layout=True)

    ax, xc = axes[0], 'bonferroni'
    ibd = proc_ibd(fn, pf_method='std', xc_method=xc,
                   min_xirs_hits=min_xirs_hits)
    ibd.plot_coverage(ax=ax, which='xirsfilt')
    ax.set_title(f'peaks with >= {min_xirs_hits} Xirs hits, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax)

    ax, xc = axes[1], 'bonferroni_cm'
    ibd = proc_ibd(fn, pf_method='std', xc_method=xc,
                   min_xirs_hits=min_xirs_hits)
    ibd.plot_coverage(ax=ax, which='xirsfilt')
    ax.set_title(f'peaks with >= {min_xirs_hits} Xirs hits, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax)

    ax, xc = axes[2], 'fdr_bh'
    ibd = proc_ibd(fn, pf_method='std', xc_method=xc,
                   min_xirs_hits=min_xirs_hits)
    ibd.plot_coverage(ax=ax, which='xirsfilt')
    ax.set_title(f'peaks with >= {min_xirs_hits} Xirs hits, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax)

    fig.savefig(f"./imgs/xirs_correction_{label}_{min_xirs_hits}_hits.pdf")


def plot_cmp_xirs_hits(fn, label='esea', xc_method='fdr_bh'):
    # compare peak finding methods
    fig, axes = plt.subplots(figsize=(12, 8), nrows=4, constrained_layout=True)

    ax, xc, min_xirs_hits = axes[0], xc_method, 1
    ibd = proc_ibd(fn, pf_method='std', xc_method=xc,
                   min_xirs_hits=min_xirs_hits)
    ibd.plot_coverage(ax=ax, which='unfilt')
    ax.set_title(f'all peaks unfilt, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax)

    ax = axes[1]
    ibd.plot_coverage(ax=ax, which='xirsfilt')
    ax.set_title(f'peaks with >= {min_xirs_hits} Xirs hits, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax, plot_drg_label=False)

    ax, xc, min_xirs_hits = axes[2], xc_method, 3
    ibd = proc_ibd(fn, pf_method='std', xc_method=xc,
                   min_xirs_hits=min_xirs_hits)
    ibd.plot_coverage(ax=ax, which='xirsfilt')
    ax.set_title(f'peaks with >= {min_xirs_hits} Xirs hits, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax, plot_drg_label=False)

    ax, xc, min_xirs_hits = axes[3], xc_method, 6
    ibd = proc_ibd(fn, pf_method='std', xc_method=xc,
                   min_xirs_hits=min_xirs_hits)
    ibd.plot_coverage(ax=ax, which='xirsfilt')
    ax.set_title(f'peaks with >= {min_xirs_hits} Xirs hits, {xc}')
    ax.set_ylim(-1000, 20000)
    ibd.plot_drg_annotation(ax=ax, plot_drg_label=False)

    fig.savefig(f"./imgs/xirs_hits_{label}_m_{xc}.pdf")


plot_cmp_std_iqr(fn1, 'esea')
plot_cmp_std_iqr(fn2, 'waf')
plot_cmp_xirs_correction(fn1, 'esea', min_xirs_hits=1)
plot_cmp_xirs_correction(fn2, 'waf', min_xirs_hits=1)
plot_cmp_xirs_hits(fn1, label='esea', xc_method='fdr_bh')
plot_cmp_xirs_hits(fn2, label='waf', xc_method='fdr_bh')


"""
fig, ax = plt.subplots(figsize=(12, 6), nrows=3, constrained_layout=True)
ibd.plot_coverage(ax=ax[0], which='unfilt')
print(ibd._peaks_df)
ax[0].set_title('all peak candidates')
ax[0].set_ylim(-1000, 20000)
ibd.plot_drg_annotation(ax=ax[0])
ibd.plot_coverage(ax=ax[1], which='xirsfilt')
ax[1].set_title('peaks with >= 1 Xirs hits')
ax[1].set_ylim(-1000, 20000)
ibd.plot_drg_annotation(ax=ax[1])
ibd.plot_xirs(ax=ax[2])
ax[2].set_title('Xirs pvalues')
ibd.plot_drg_annotation(ax=ax[2])
fig.savefig("tmp2.pdf")
"""
