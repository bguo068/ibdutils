#! /usr/bin/env python3
import allel
import pandas as pd
import numpy as np
from scipy.stats import chi2
from typing import List, Tuple


def get_afreq_from_vcf_files(
    vcf_fn_lst: List,
    samples=None,
    fix_pf3d7_chrname=False,
    fix_tsk_samplename=False,
    rm_sample_name_suffix=False,
    check_pseudo_diploid=False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Get frquency dataframe for all included chromosomes that is sorted by
    chromosome and bp position.

    Parameters
    ----------
    - `vcf_fn_lst` : a list of str/path
        A list of vcf file names. The length can be one or more than one. The
        vcf file is assumed to pesudo diploid samples.
    - `samples` : pd.Series of str type, optional
        If not None, only samples in this parameter will be used to calculate
        allele frequency. For instance, this parameter can be used to calculate
        allele frequeny on unrelated samples.
    - `fix_pf3d7_chrname`: boolean, optional
        If True, convert "Pf3D7_XX_v3" chromosome names to "XX" integers.
    - `fix_tsk_samplename`: boolean, optional
        If True, remove 'tsk_' prefix from the sample names, and convert them
        to intergers.
    - `rm_sample_name_suffix`: boolean, optional
        If True, remove '~.*$' prefix from the sample names, and convert them
        to intergers.
    - `check_pseudo_diploid`: boolean, optional
        If True, assert homozyogsity for each genotype call.

    Returns
    --------
        A tuple of alelle frequence (pd.DataFrame) and sorted samples (pd.Series)


    """
    assert type(vcf_fn_lst) == list
    fields = fields = ["variants/POS", "calldata/GT", "variants/CHROM"]

    pos_lst = []
    chr_lst = []
    frq_lst = []
    samples = None

    for vcf_fn in vcf_fn_lst:
        samples2 = pd.Series(allel.read_vcf_headers(vcf_fn).samples)
        if rm_sample_name_suffix:
            samples2 = samples2.str.replace("~.*$", "", regex=True)
        if samples is not None:
            samples2 = samples2[samples2.isin(samples)]
        if samples is not None:
            # make sure all vcf have samples in the same order
            assert np.all(samples == samples2)
        else:
            samples = samples2

        vcf = allel.read_vcf(vcf_fn, fields=fields)
        pos = vcf["variants/POS"]
        gt = vcf["calldata/GT"]
        if check_pseudo_diploid:
            # assume vcf are pseudo homozygote for all sites
            assert np.all(gt[:, :, 0] == gt[:, :, 1])
        gt = gt[:, :, 0]
        freq = gt.mean(axis=1).copy()
        chrs = vcf["variants/CHROM"]

        pos_lst.extend(pos)
        chr_lst.extend(chrs)
        frq_lst.extend(freq)

    df = pd.DataFrame({"Chromosome": chr_lst, "Pos": pos_lst, "Freq": frq_lst})

    if fix_pf3d7_chrname and not pd.api.types.is_integer_dtype(df.Chromosome):
        df["Chromosome"] = df.Chromosome.str.replace(
            "Pf3D7_", "", regex=False
        ).str.replace("_v3", "", regex=False)

    df["Chromosome"] = df.Chromosome.astype(int)

    df = df.sort_values(["Chromosome", "Pos"])

    if fix_tsk_samplename and samples.str.contains("tsk_").any():
        samples = samples.str.replace("tsk_", "", regex=False).astype(int)

    return df, samples.sort_values().reset_index(drop=True)


def get_ibd_status_matrix(ibd_df_chr: pd.DataFrame, pos_chr: np.ndarray) -> np.ndarray:
    """
    Create a matrix of binary IBD status with rows corresponding to SNPs and
    columns corresponding to isolate pairs.

    Parameters
    ----------
    - `ibd_all_df`: pd.DataFrame
        A dataframe contains "Id1", "Id2", "Chromosome", "Start" and "End"
        columns. NOTE: the dataframe should only contain segments of the SAME
        chromosome, that is, Chromosome values are the same across rows. "Id1"
        and "Id2" should be already encoded into integers from 0 to num_samples
        - 1.
    - `pos_chr`: np.ndarray
        The array contains SORTED SNP positions in base pair.

    Returns
    ----------
    np.ndarray, with rows being SNPs and columns being sample/genome pairs

    """
    ibd = ibd_df_chr
    pos = pos_chr

    # assert position is sorted
    np.all(pos_chr[:-1] <= pos_chr[1:])
    assert pd.api.types.is_integer_dtype(ibd.Id1)
    assert pd.api.types.is_integer_dtype(ibd.Id2)

    # ensure the id1, Id2 order
    r = ibd[["Id1", "Id2"]].max(axis=1).to_numpy()
    c = ibd[["Id1", "Id2"]].min(axis=1).to_numpy()

    start = np.searchsorted(pos, ibd.Start, side="left")
    end = np.searchsorted(pos, ibd.End, side="right")

    ibd = pd.DataFrame({"Pair": (r * r - r) // 2 + c, "Start": start, "End": end})

    M = np.zeros(shape=(pos.size, ibd.Pair.max() + 1))

    for p, s, e in ibd.itertuples(index=False):
        M[s:e, p] = 1

    return M


def calc_xirs_raw_stats_per_chr(M: np.ndarray, frq: np.ndarray):
    """
    Calculate unnormalized Raw xirs stats for each chromosome.

    Parameters
    ----------
    - `M`: np.ndarray, two dimentional
        A matrix of binary IBD status. np.ndarray, with rows being SNPs and
        columns being sample/genome pairs. See `get_ibd_status_matrix`
        function.
    - `frq`: np.ndarray, one dimentional
        An array of allele frequency with orders corresponding to SNPs of `M` (rows).

    Returns
    ----------Benjamini-Hochberg
        np.ndarray, one dimentional
        An array of unnormalized XiR,s stats for each SNP corresponding to
        rows of `M` matrix

    """

    # For each column, we subtract the column mean from all rows to account for
    # the amount of relatedness between each pair. Note: use for loop to avoid
    # running out of RAM as using numpy matrix calculation directly need more
    # RAM.
    for c in range(M.shape[1]):
        col_mean = M[:, c].mean()
        M[:, c] = M[:, c] - col_mean

    # Following this we subtract the row mean from each row and divide by the
    # square root of pi(1-pi), where pi is the population allele frequency of
    # SNP i. This adjusts for differences in SNP allele frequencies, which can
    # affect the ability to detect IBD.
    for r in range(M.shape[0]):
        row_mean = M[r, :].mean()
        p = frq[r]
        M[r, :] = (M[r, :] - row_mean) / np.sqrt(p * (1 - p))

    # Next we calculate row sums and divide these values by the square root of
    # the number of pairs.
    stats = M.sum(axis=1) / np.sqrt(M.shape[1])
    return stats


def calc_xirs(freq_all_df, ibd_all_df):
    """
    Calculate XiR,s stats across all chromosomes by calling
    `get_ibd_status_matrix` and `calc_xirs_raw_stats_per_chr`.

    Parameters
    ----------
    - `freq_all_df`: pd.DataFrame
        A dataframe containing "Chromosome", "Pos" and "Freq" columns and is
        sorted by "Chromosome" and "Pos". See `get_afreq_from_vcf_files`.
    - `ibd_all_df`: pd.DataFrame
        A dataframe contains "Id1", "Id2", "Chromosome", "Start" and "End" columns. It
        can contains IBD segments from multiple chromosomes. "Id1" and "Id2" should
        be already encoded into integers from 0 to num_samples - 1.

    Returns
        pd.DataFrame. It contains the following columns:
            - Chromosome
            - Pos
            - Freq
            - RawStat
            - Bin
            - RawStatMean
            - RawStatStd
            - Zscore
            - ChisqStat
            - Pvalue: based on ChisqStat of a degree = 1
            - NegLogP: -np.log10(pvalue)
    -------

    """

    # make sure ibd_all_df Id1 and Id2 are integers
    assert pd.api.types.is_integer_dtype(ibd_all_df.Id1)
    assert pd.api.types.is_integer_dtype(ibd_all_df.Id2)

    df_list = []
    for chrno in ibd_all_df.Chromosome.unique():
        ibd_df_chr = ibd_all_df[lambda df: df.Chromosome == chrno]
        pos_chr = freq_all_df.loc[lambda df: df.Chromosome == chrno, "Pos"].to_numpy()
        frq_chr = freq_all_df.loc[lambda df: df.Chromosome == chrno, "Freq"].to_numpy()

        M = get_ibd_status_matrix(ibd_df_chr, pos_chr)
        stats = calc_xirs_raw_stats_per_chr(M, frq_chr)
        df = pd.DataFrame(
            {"Chromosome": chrno, "Pos": pos_chr, "Freq": frq_chr, "RawStat": stats}
        )
        df_list.append(df)
    df = pd.concat(df_list, axis=0)

    # These summary statistics are normalized genome-wise by binning all SNPs
    # into 100 equally sized bins partitioned on allele frequencies and then we
    # subtracted the mean and divided by the standard deviation of all values
    # within each bin.
    df = df.sort_values("Freq")

    npos = df.shape[0]
    binsize = npos // 100 + ((npos % 100) > 0)
    df["Bin"] = np.arange(npos) // binsize

    bin_mean = df.groupby("Bin")["RawStat"].mean()
    bin_mean.name = "RawStatMean"
    bin_mean = bin_mean.reset_index()

    bin_std = df.groupby("Bin")["RawStat"].std()
    bin_std.name = "RawStatStd"
    bin_std = bin_std.reset_index()

    df = df.merge(bin_mean, how="left", on="Bin").merge(bin_std, how="left", on="Bin")

    df = df.sort_values(["Chromosome", "Pos"])
    df["Zscore"] = (df.RawStat - df.RawStatMean) / df.RawStatStd

    # Negative z-scores are difficult to interpret when investigating positive
    # selection; therefore we square the z-scores such that the new summary
    # statistics follow a chi-squared distribution with 1 degree of freedom.
    # This produces a set of genome wide test statistics (XiR,s), where XiR,s
    # is the chisquare distributed test statistic for IBD sharing from
    # isoRelate at SNPs.
    df["ChisqStat"] = df.Zscore * df.Zscore
    df["Pvalue"] = chi2.sf(df.ChisqStat, df=1)

    # p = df.Pvalue.to_numpy()
    # rank = np.argsort(p) + 1
    # ntests = p.size
    # df["AdjPvalue"] = p * ntests / rank
    df["NegLogP"] = -np.log10(df.Pvalue)

    return df
