# ibdutils
## Introduction

`ibdutils` is a small Python package designed to facilitate identity-by-descent
(IBD) analysis. It offers a set of tools for genetic researchers and
bioinformaticians working with IBD data.

## Features

### IBD Filteration and Processing
- Filter IBD segments by TMRCA, mutation, IBD segment length and samples
- Allow haploid-to-diploid IBD conversion and flattening of diploid IBD segments
- Remove highly related samples/isolates biased on pairwise total IBD network

### Selection Correction
- Calculate and visualize IBD coverage over sampling sites
- Calculate and visualize `Xir,s` statistics to scan for selection signals
- Identify (via IBD coverage threholding method) and validate (via `Xir,s`
statistics) IBD peak regions
- Remove IBD located within identified IBD peak regions for selection correction
- Split genomes into contigs of non-zero IBD coverage, important for preparing
IBDNe input after removing IBD from peak regions

### IBD-based Downstream Analyses
- Prepare pairwise total IBD sharing matrix and run hierarchical clustering over it
- Provide wrappers for IBD-based downstream analyses, including:
    - `igraph-python` Infomap community detection algorithm for population
    structure analysis
    - `IBDNe` to infer the trajectory of effective population size for recent times

### IBD Benchmarking
- Provide an IbdComparator class to allow benchmarking inferred IBD set against
a true IBD set

### Helper Classes and Additional Utilities
- Provide helper classes like GeneticMap (base pair and centimorgan coordinate
conversion) and Genome (chromosome sizes, and annotations like drug resistance
genes, multigene family)
- Fast dump and load of IBD objects



## Installation

The package can be easily installed via `pip`:
```sh
git clone git@github.com:bguo068/ibdutils.git
cd ibdutils
# optional: git checkout [specific version/branch/commit]
pip install .
```

## Usage examples

0. Prepare input data
```sh
#! /usr/bin/env bash
tar xf testdata.tgz
```

1. Example 1

```py
import ibdutils.utils.ibdutils as ibdutils
import ibdutils.runner.ibdne as ibdne

# ---------------- prepare IBDNe input files ------------
# input files (simulated from single population model with selection)
sp_ibd_files = [f"testdata/single_pop/tskibd/{i}.ibd" for i in range(1, 15)]
sp_vcf_files = [f"testdata/single_pop/vcf/{i}.vcf.gz" for i in range(1, 15)]

# parameters
label_str = "tmp_sp_sim"
ibdne_flatmeth = "none"
ibdne_mincm = 2

# read ibd
genome_14_100 = ibdutils.Genome.get_genome("simu_14chr_100cm")
ibd = ibdutils.IBD(genome=genome_14_100, label=f"{label_str}_orig")
ibd.read_ibd(ibd_fn_lst=sp_ibd_files)

# remove highly related samples
mat = ibd.make_ibd_matrix()
unrelated_samples = ibd.get_unrelated_samples(ibdmat=mat)
ibd.subset_ibd_by_samples(subset_samples=unrelated_samples)

# prepare input for IBDNe
#  remove ibd with tmrca < 1.5 (required according to IBDNe paper)
ibd.filter_ibd_by_time(min_tmrca=1.5)
ibd.filter_ibd_by_length(min_seg_cm=ibdne_mincm)

# calculate XiR,s (this step can be pretty slow)
xirs_df = ibd.calc_xirs(vcf_fn_lst=sp_vcf_files, min_maf=0.01)

# convert to heterzygous diploids (IBDNe works with diploid data)
ibd.convert_to_heterozygous_diploids(remove_hbd=True)

# flattening function can be used
if ibdne_flatmeth != "none":
    ibd.flatten_diploid_ibd(method=ibdne_flatmeth)

# calculate IBD coverage and find peaks
ibd.calc_ibd_cov()
ibd.find_peaks()

# only keep peaks that contain a xirs hit
ibd.filter_peaks_by_xirs(xirs_df)

# save IBD before remove peaks
# of_orig_ibdne_obj = f"{label_str}_orig.ibdne.ibdobj.gz"
# ibd.pickle_dump(of_orig_ibdne_obj)

# remove peaks
ibd2 = ibd.duplicate(f"{label_str}_rmpeaks")
ibd2.remove_peaks()
ibd2._df = ibd2.cut_and_split_ibd()

# save IBD after remove peaks
# of_rmpeaks_ibdne_obj = f"{label_str}_rmpeaks.ibdne.ibdobj.gz"
# ibd2.pickle_dump(of_rmpeaks_ibdne_obj)

ibdne_runner1 = ibdne.IbdNeRunner(ibd, ".", ".")
ibdne_runner1.run(dry_run=True)
# This generate three files *.ibd.gz/*.map/*.sh for running IBDNe

ibdne_runner2 = ibdne.IbdNeRunner(ibd2, ".", ".")
ibdne_runner2.run(dry_run=True)
# This generate three files *.ibd.gz/*.map/*.sh for running IBDNe

```

2. Example 2
```py
import ibdutils.utils.ibdutils as ibdutils
import ibdutils.runner.ibdne as ibdne
import pandas as pd
import numpy as np

# ---------------- call infomap ------------
# input files
mp_ibd_files = [f"testdata/multiple_pop/tskibd/{i}.ibd" for i in range(1, 15)]
mp_vcf_files = [f"testdata/multiple_pop/vcf/{i}.vcf.gz" for i in range(1, 15)]

# parameters
label_str = "tmp_mp_sim"
nsam = 200  # no. of isolates per subpopulation
npop = 5  # no. of subpopulations
transform = "square"  # transformation of IBD matrix
ntrials = 1000  # parameter of infomap algorithm

# meta information with optional true labels (for purpose of comparison)
meta = pd.DataFrame(
    {
        "Sample": np.arange(nsam * npop),  # use haploid here
        "Population": np.repeat(np.arange(npop), nsam),
    }
)

# read ibd from files
genome_14_100 = ibdutils.Genome.get_genome("simu_14chr_100cm")
ibd = ibdutils.IBD(genome=genome_14_100, label=f"{label_str}_orig")
ibd.read_ibd(ibd_fn_lst=mp_ibd_files)


# remove highly relatedness samples
mat = ibd.make_ibd_matrix()
unrelated_samples = ibd.get_unrelated_samples(ibdmat=mat)
ibd.subset_ibd_by_samples(subset_samples=unrelated_samples)

# calculate IBD coverage and find peaks
ibd.calc_ibd_cov()
ibd.find_peaks()

# calculate XiR,s and filter peaks
xirs_df = ibd.calc_xirs(vcf_fn_lst=mp_vcf_files, min_maf=0.01)
ibd.filter_peaks_by_xirs(xirs_df)

# make a copy of the IBD object and remove IBD within peaks on the copy
ibd2 = ibd.duplicate("rmpeak")
ibd2.remove_peaks()

# run infomap on IBD object without peak removal
mat = ibd.make_ibd_matrix()
member_df = ibd.call_infomap_get_member_df(
    mat, meta, trials=ntrials, transform=transform
)

# run infomap on IBD object WITH peak removal
mat2 = ibd2.make_ibd_matrix()
member_df2 = ibd2.call_infomap_get_member_df(
    mat2, meta, trials=ntrials, transform=transform
)

# Infomap identifies 2 main groups
member_df.Rank.value_counts().iloc[:6]
# 0    640
# 1    120
# 2     15
# 3     11
# 4      9
# 5      9
# Name: Rank, dtype: int64

# Infomap identifies 4 main groups when selection correction is performed
member_df2.Rank.value_counts().iloc[:6]
# 0    203
# 1    150
# 2    149
# 3    108
# 4     13
# 5      9
# Name: Rank, dtype: int64

```

## Caveats and Related Ongoing Work

1. The documentation for each function or method is currently only partially
complete.
2. The existing implementation is based solely on Python. A separate
implementation in Rust is in progress. The Rust version is expected to offer
enhanced computational efficiency, particularly for the Xirs calculation,
coverage calculation, and peak removal steps.

## Citation

If you find this tool useful, please cite our preprint:
> Guo, B., Borda, V., Laboulaye, R., Spring, M. D., Wojnarski, M., Vesely, B.
A., Silva, J. C., Waters, N. C., O'Connor, T. D., & Takala-Harrison, S. (2023).
Strong Positive Selection Biases Identity-By-Descent-Based Inferences of Recent
Demography and Population Structure in Plasmodium falciparum. bioRxiv : the
preprint server for biology, 2023.07.14.549114.
https://doi.org/10.1101/2023.07.14.549114

Other citations:

- `Xir,s` statistics: 
> Henden, L., Lee, S., Mueller, I., Barry, A., & Bahlo, M. (2018).
Identity-by-descent analyses for measuring population dynamics and selection in
recombining pathogens. PLoS genetics, 14(5), e1007279.
https://doi.org/10.1371/journal.pgen.1007279
- `IBDNe`
> Browning, S. R., & Browning, B. L. (2015). Accurate Non-parametric Estimation
of Recent Effective Population Size from Segments of Identity by Descent.
American journal of human genetics, 97(3), 404–418.
https://doi.org/10.1016/j.ajhg.2015.07.012

- `Infomap` algorithm
> Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex
networks reveal community structure. Proceedings of the National Academy of
Sciences of the United States of America, 105(4), 1118–1123.
https://doi.org/10.1073/pnas.0706851105

##  Resources


For packaging: 
- basic: https://packaging.python.org/en/latest/tutorials/packaging-projects/
- include files: https://packaging.python.org/en/latest/guides/using-manifest-in/


