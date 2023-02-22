#! /usr/bin/env python3

import io
import random
import re
from typing import List, Tuple, Union

import allel
import igraph
import numpy as np
import pandas as pd
import pybedtools as pb
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

import numpy as np


class GeneticMap:
    def __init__(self):
        # gmap is datafram with columns: "Chromosome", "Bp", "Cm", "CmPerBp"
        self.gmap: pd.DataFrame = None

    @staticmethod
    def from_plink_maps(
        fn_lst: List[str],
        sep=" ",
    ) -> "GeneticMap":

        # read files
        df_lst = []
        columns = ["Chromosome", "VarId", "Cm", "Bp"]
        for fn in fn_lst:
            df_lst.append(pd.read_csv(fn, sep=sep, names=columns))
        # combines
        df: pd.DataFrame = pd.concat(df_lst, axis=0)

        # add bp = 0 rows
        rows_to_add = {"Chromosome": [], "VarId": [], "Cm": [], "Bp": []}
        for chr, df_chr in df.groupby("Chromosome"):
            if 0 not in df_chr.Bp.values:
                rows_to_add["Chromosome"].append(chr)
                rows_to_add["VarId"].append(".")
                rows_to_add["Cm"].append(0.0)
                rows_to_add["Bp"].append(0)
        rows_to_add = pd.DataFrame(rows_to_add)

        df = (
            pd.concat([df, rows_to_add])
            .sort_values(["Chromosome", "Bp"])
            .reset_index(drop=True)
        )

        # remove VarId columns and add 'cm_per_bp' columns
        df = df.rename(columns={"VarId": "CmPerBp"})[
            ["Chromosome", "Bp", "Cm", "CmPerBp"]
        ]

        # calculate the slope
        Cm = df.Cm.values  # NOTE: use ndarray for easier offset subtraction
        Bp = df.Bp.values
        CmPerBp = np.zeros_like(Cm)
        for chr, df_chr in df.groupby("Chromosome"):
            assert df_chr.shape[0] >= 2
            first, last = df_chr.index.values[0], df_chr.index.values[-1]

            diff_Cm = Cm[(first + 1) : (last + 1)] - Cm[first:last]
            diff_Bp = Bp[(first + 1) : (last + 1)] - Bp[first:last]

            CmPerBp[first:last] = diff_Cm / diff_Bp
            CmPerBp[last] = CmPerBp[last - 1]
        df["CmPerBp"] = CmPerBp

        # gmap obj
        gmap = GeneticMap()
        gmap.gmap = df

        return gmap

    def to_plink_maps(self, fn_prefix="", sep=" ", per_chr: bool = True):
        # NOTE: before writing to plink format, remove the bp = 0 rows
        df = self.gmap[lambda df: df.Bp != 0].sort_values(["Chromosome", "Bp"]).copy()
        df["VarId"] = "."
        df = df[["Chromosome", "VarId", "Cm", "Bp"]]

        # write
        if per_chr:
            for chr, df_chr in df.groupby("Chromosome"):
                out_fn = fn_prefix + chr + ".map"
                df_chr.to_csv(out_fn, sep=sep, index=None, header=None)
        else:
            out_fn = fn_prefix + ".map"
            df.to_csv(out_fn, sep=sep, index=None, header=None)
        pass

    @staticmethod
    def from_const_rate(
        r=None,
        bp_per_cm=None,
        chrid_lst: List[Union[int, str]] = None,
        chrlen_bp_lst: List[int] = None,
        chrlen_cm_lst: List[float] = None,
    ) -> "GeneticMap":

        # one and only one of r and bp_per_cm can be specified
        assert ((r is None) + (bp_per_cm is None)) == 1
        # one and only one of  chrlen_bp_lst and chrlen_cm_lst can be specified
        assert (chrlen_bp_lst is None) + (chrlen_cm_lst is None) == 1

        if bp_per_cm is None:
            bp_per_cm = int(0.01 / r)

        if chrlen_bp_lst is None:
            chrlen_bp_lst = [cm * bp_per_cm for cm in chrlen_cm_lst]

        # chrlen_cm_lst is should be either None or a list with same size as chrlen list
        assert (chrid_lst is None) or (len(chrid_lst) == len(chrlen_bp_lst))
        if chrid_lst is None:
            chrid_lst = list(range(1, 1 + len(chrlen_bp_lst)))

        # create the geneticmap dataframe
        chroms = []
        bps = []
        cms = []
        cm_per_bp_lst = []
        for chrid, len_bp in zip(chrid_lst, chrlen_bp_lst):
            chroms.extend([chrid] * 3)
            bps.extend([0, 1, len_bp])
            cms.extend([0, 1.0 / bp_per_cm, 1.0 * len_bp / bp_per_cm])
            cm_per_bp_lst.extend([1.0 / bp_per_cm] * 3)
        gmap = pd.DataFrame(
            {"Chromosome": chroms, "Bp": bps, "Cm": cms, "CmPerBp": cm_per_bp_lst}
        )

        # make geneticmap object
        gmap_obj = GeneticMap()
        gmap_obj.gmap = gmap

        return gmap_obj

    def get_bp(self, chrom: np.ndarray, cm: np.ndarray) -> np.ndarray:

        assert chrom.shape == cm.shape
        assert len(chrom.shape) == 1
        df = pd.DataFrame(
            {"Chromosome": chrom, "Cm": cm, "InputOrder": np.arange(chrom.shape[0])}
        )
        df = df.sort_values(["Chromosome", "Cm"]).reset_index(drop=True)
        gmap_grouped = self.gmap.groupby("Chromosome")

        bp = np.zeros(shape=df.Cm.shape, dtype=int)
        for chr, df_chr in df.groupby("Chromosome"):
            s = df_chr.index.values[0]
            e = df_chr.index.values[-1] + 1
            gg = gmap_grouped.get_group(chr)

            idx = np.searchsorted(gg.Cm, df_chr.Cm.values, side="right") - 1
            bp[s:e] = gg.Bp.values[idx] + np.round(
                (df_chr.Cm.values - gg.Cm.values[idx]) / gg.CmPerBp.values[idx]
            )

        # sort mapped result according the input the order
        df["Bp"] = bp
        bp_out = df.sort_values("InputOrder")["Bp"].values.copy()

        return bp_out

    def get_cm(self, chrom: np.ndarray, bp: np.ndarray) -> np.ndarray:
        assert chrom.shape == bp.shape
        assert len(chrom.shape) == 1
        df = pd.DataFrame(
            {"Chromosome": chrom, "Bp": bp, "InputOrder": np.arange(chrom.shape[0])}
        )
        df = df.sort_values(["Chromosome", "Bp"]).reset_index(drop=True)
        gmap_grouped = self.gmap.groupby("Chromosome")

        cm = np.zeros(shape=df.Bp.shape, dtype=np.float64)
        for chr, df_chr in df.groupby("Chromosome"):
            # get position in full array
            s = df_chr.index.values[0]
            e = df_chr.index.values[-1] + 1
            gg = gmap_grouped.get_group(chr)

            idx = np.searchsorted(gg.Bp, df_chr.Bp.values, side="right") - 1
            cm[s:e] = (
                gg.Cm.values[idx]
                + (df_chr.Bp.values - gg.Bp.values[idx]) * gg.CmPerBp.values[idx]
            )

        # sort mapped result according the input the order
        df["Cm"] = cm
        cm_out = df.sort_values("InputOrder")["Cm"].values.copy()

        return cm_out

    def get_length_in_cm(
        self, chrom: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        right_cm = self.get_cm(chrom, right)
        left_cm = self.get_cm(chrom, left)
        dist_cm = right_cm - left_cm
        return dist_cm


class Genome:
    def __init__(
        self,
        chr_df=None,
        drg_df=None,
        bp_per_cm=None,
        gmap: GeneticMap = None,
        label=None,
    ):
        self._chr_df = chr_df
        self._drg_df = drg_df
        self._bp_per_cm = bp_per_cm
        self._gmap: GeneticMap = gmap
        self._label = label

    def get_genome_wide_coords(
        self, chromosome: np.ndarray, bp: np.ndarray
    ) -> np.ndarray:

        assert chromosome.shape == bp.shape and len(chromosome.shape) == 1
        assert pd.Series(chromosome).isin(self._chr_df.Chromosome).all()

        gw_bp = np.zeros_like(bp)

        chr_map = pd.Series(
            self._chr_df.GwChromStart.values, index=self._chr_df.Chromosome.values
        )
        gw_bp = bp + chr_map.loc[chromosome].values

        return gw_bp

    def get_chromosome_boundaries(self):
        gwchromstarts = self._chr_df.GwChromStart.tolist()
        gwchromends = self._chr_df.GwChromEnd.tolist()
        boundaries = np.unique(gwchromstarts + gwchromends)
        return boundaries

    def get_chromosome_gw_center(self):
        return self._chr_df.GwChromCenter.values

    def get_drg_gw_center(self, types=["drg", "sex"]):
        drg = self._drg_df.loc[lambda df: df.Comment.isin(types)]
        chromosome = drg.Chromosome.values
        centers = (drg.End + drg.Start) / 2.0
        return self.get_genome_wide_coords(chromosome, centers)

    def get_drg_labels(self, types=["drg", "sex"]):
        return self._drg_df[lambda df: df.Comment.isin(types)]["Name"].values

    @staticmethod
    def _get_pf3d7_chrlen_lst():
        s = """
            640851 947102 1067971 1200490 1343557 1418242 1445207 1472805
            1541735 1687656 2038340 2271494 2925236 3291936
            """
        return [int(i) for i in s.split()]

    @staticmethod
    def _get_annotation(label="drg"):
        if label == "drg":
            str_table = """
ID               Name           Chromosome  Start    End      Comment
pf3d7_0206800    msp2           2           273689   274507   not_drg
pf3d7_0304600    csp            3           221323   222516   not_drg
PF3D7_0417200    dhfr           4           748088   749914   drg
PF3D7_0523000    mdr1           5           957890   962149   drg
PF3D7_0629500    aat1           6           1213102  1217313  drg
PF3D7_0709000    crt            7           403222   406317   drg
PF3D7_0720700    pib7           7           891683   899051   drg
PF3D7_0731500    eba175         7           1358055  1362929  not_drg
PF3D7_0810800    dhps           8           548200   550616   drg
PF3D7_0930300    msp1           9           1201812  1206974  not_drg
PF3D7_1012700    pph            10          487252   491828   drg
PF3D7_1036300    mspdbl2        10          1432498  1434786  drg
PF3D7_1224000    gch1           12          974372   975541   drg
PF3D7_1222600    ap2-g*         12          907203   914501   sex
PF3D7_1318100    fd             13          748387   748971   drg
PF3D7_1322700    PF3D7_1322700  13          958219   959175   drg
PF3D7_1343700    k13            13          1724817  1726997  drg
PF3D7_1335900    trap           13          2005962  2007950  not_drg
PF3D7_1408200    ap2-g2*        14          300725   305833   sex
PF3D7_1451200    PF3D7_1451200  14          2094340  2099736  drg
PF3D7_1460900.1  arps10         14          2480440  2481916  drg"""

        else:
            assert False, f"Not implemented for {label}"
        s = re.sub(r"[ \t]+", ",", str_table.strip())

        return pd.read_csv(io.StringIO(s))

    @staticmethod
    def get_multigene_familiy_genes_df(gff_fn=None):
        """currently it only makes sence if genome is 'Pf3D7'"""
        if gff_fn is None:
            gff_fn = "/autofs/burnsfs/projects-t3/toconnor_grp/bing.guo/ref/PlasmoDB-44_Pfalciparum3D7.gff"

        columns = """
        Sequence_ID
        Source
        Feature_Type
        Start
        End
        Score
        Strand
        Phase
        Attributes
        """.strip().split()

        df = pd.read_csv(gff_fn, sep="\t", comment="#", names=columns)
        df["Description"] = (
            df["Attributes"]
            .str.extract("description=(?P<Description>[^;]*)", expand=True)
            .fillna("")
        )["Description"]

        df["GeneId"] = (
            df["Attributes"].str.extract("ID=(?P<ID>[^;]*)", expand=True).fillna("")
        )["ID"]

        gene_map = {
            "erythrocyte membrane protein 1%2C PfEMP1": "var",
            "rifin": "rifin",
            "stevor": "stevor",
        }
        df["MultiGeneFam"] = df.Description.map(gene_map).fillna("")
        df["Chromosome"] = (
            df["Sequence_ID"]
            .str.extract("Pf3D7_(?P<Chromosome>[0-9]+)+_v3", expand=True)
            .fillna(-1)
            .Chromosome.astype(int)
        )
        df["Center"] = (df.Start + df.End) // 2

        sel_rows = (df.MultiGeneFam != "") & (df.Feature_Type == "gene")
        sel_cols = [
            "Chromosome",
            "Start",
            "End",
            "Center",
            "GeneId",
            "MultiGeneFam",
            "Description",
        ]
        sel = (
            df.loc[sel_rows, sel_cols]
            .sort_values(["Chromosome", "Start"])
            .reset_index(drop=True)
        )

        return sel

    @staticmethod
    def get_genome(genome_model: str) -> "Genome":
        """[static function]:
        construct genome_model from predefined information.
        @genome_model: can be one of "simu_14chr_100cm" and "Pf3D7" """

        assert genome_model in ["simu_14chr_100cm", "Pf3D7"]
        if genome_model == "simu_14chr_100cm":
            bp_per_cm = 15_000
            chrlen = 100 * bp_per_cm
            nchroms = 14
            chrnos = list(range(1, nchroms + 1))

            chr_df = pd.DataFrame(
                {"Chromosome": chrnos, "ChromLength": [chrlen] * nchroms}
            )
            chr_df["GwChromEnd"] = chr_df.ChromLength.cumsum()
            chr_df["GwChromStart"] = chr_df.GwChromEnd - chr_df.ChromLength
            chr_df["GwChromCenter"] = (chr_df.GwChromStart + chr_df.GwChromEnd) / 2

            gmap = GeneticMap.from_const_rate(
                bp_per_cm=bp_per_cm, chrlen_cm_lst=[100] * 14
            )
            genome = Genome(
                chr_df, drg_df=None, bp_per_cm=bp_per_cm, gmap=gmap, label=genome_model
            )

        elif genome_model == "Pf3D7":
            bp_per_cm = 15_000
            chrlen_lst = Genome._get_pf3d7_chrlen_lst()
            nchroms = len(chrlen_lst)
            chrnos = list(range(1, 1 + nchroms))

            chr_df = pd.DataFrame({"Chromosome": chrnos, "ChromLength": chrlen_lst})
            chr_df["GwChromEnd"] = chr_df.ChromLength.cumsum()
            chr_df["GwChromStart"] = chr_df.GwChromEnd - chr_df.ChromLength
            chr_df["GwChromCenter"] = (chr_df.GwChromStart + chr_df.GwChromEnd) / 2

            drg_df = Genome._get_annotation()

            gmap = GeneticMap.from_const_rate(
                bp_per_cm=bp_per_cm, chrlen_cm_lst=chrlen_lst
            )
            genome = Genome(
                chr_df, drg_df, bp_per_cm=bp_per_cm, gmap=gmap, label=genome_model
            )

        else:
            genome = None
            raise Exception("Genome not implemented!")

        return genome

    def get_genome_size_cm(self):
        gmap = self._gmap
        chr_df = self._chr_df

        chrom = chr_df.Chromosome.values
        starts = np.zeros(shape=chrom.shape)
        ends = chr_df.ChromLength.values

        cms = gmap.get_length_in_cm(chrom, starts, ends)
        return cms.sum()

    def get_genome_size_bp(self):
        chr_df = self._chr_df
        return chr_df.ChromLength.sum()


class VCF:
    def __init__(self, vcf_fn: str, samples: pd.Series = None):
        self.vcf_fn = vcf_fn
        self.samples = samples
        self.calldata = None

    def duplicate(self) -> "VCF":
        v = VCF(self.vcf_fn, self.samples)
        calldata = dict()
        for k in self.calldata.keys():
            calldata[k] = self.calldata[k].copy()

        v.calldata = calldata
        return v

    def read_calldata(self, alt_number=1):
        allsamples = pd.DataFrame({"Orig": allel.read_vcf_headers(self.vcf_fn).samples})
        allsamples["Sample"] = allsamples.Orig.str.replace("~.*$", "", regex=True)
        if self.samples is None:
            self.samples = allsamples.Sample

        sel_samples = (
            pd.DataFrame({"Sample": self.samples})
            .merge(allsamples, how="left", on="Sample")
            .Orig
        )

        self.calldata = allel.read_vcf(
            self.vcf_fn, alt_number=alt_number, samples=sel_samples
        )
        # convert chromosome names to integer
        self.calldata["variants/CHROM"] = (
            pd.Series(self.calldata["variants/CHROM"])
            .str.replace("Pf3D7_|_v3", "", regex=True)
            .astype(int)
            .to_numpy()
        )
        # remove suffix from the sample names
        self.calldata["samples"] = (
            pd.Series(self.calldata["samples"])
            .str.replace("~.*$", "", regex=True)
            .to_numpy()
        )

    def calc_gw_pos(self, genome: Genome):
        self.calldata["variants/GWPOS"] = genome.get_genome_wide_coords(
            self.calldata["variants/CHROM"], self.calldata["variants/POS"]
        )

    def remove_sites_within_regions(self, regions: pd.DataFrame, genome: Genome):
        if "variants/GWPOS" not in self.calldata.keys():
            self.calc_gw_pos(genome)

        for col in ["Chromosome", "Start", "End"]:
            assert col in regions.columns

        starts = genome.get_genome_wide_coords(
            regions.Chromosome.values, regions.Start.values
        )
        ends = genome.get_genome_wide_coords(
            regions.Chromosome.values, regions.End.values
        )
        pos = self.calldata["variants/GWPOS"]

        in_region = np.repeat(False, pos.size)

        for s, e in zip(starts, ends):
            in_region = in_region | ((pos >= s) & (pos < e))

        for key in self.calldata.keys():
            if key != "samples":
                self.calldata[key] = self.calldata[key][~in_region]

    def calc_fst(
        self, assignment: pd.Series, per_variant=False
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        subpop: a pd.Series of Rankv(values) of seleced samples (in index)
        """
        samples = self.calldata["samples"]
        nsam = samples.size

        sample2id = pd.Series(np.arange(nsam), index=samples)
        sample_ids = sample2id[assignment.index.values]
        ga = self.calldata["calldata/GT"][:, sample_ids, :]

        subpop_lst_lst = (
            assignment.reset_index(
                drop=True
            )  # drop original index and  add new 0-n index
            .rename_axis(index="Idx")  # rename index name
            .rename("Assignment")  #  rename value name
            .reset_index()  # make a dataframe
            .groupby("Assignment")["Idx"]  # make a series of list
            .apply(list)
            .to_list()  # make list of list
        )

        # components of variance
        a, b, c = allel.weir_cockerham_fst(ga, subpops=subpop_lst_lst)

        if per_variant:
            # average over allele per variants
            fst = np.sum(a, axis=1) / (
                np.sum(a, axis=1) + np.sum(b, axis=1) + np.sum(c, axis=1)
            )
        else:
            # average over all alleles and all variants
            fst = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))

        return fst

    def pca(self):
        # n_variants x n_samples
        gn = self.calldata["calldata/GT"].sum(axis=2)

        # filter rare variants
        af = gn.sum(axis=1) / gn.shape[1]
        rare_variants = (af < 0.01) | (af > 0.09)
        gn_common = gn[~rare_variants, :]

        # locate unlinked
        is_unlinked = allel.locate_unlinked(gn_common)
        gn_unlinked = gn_common[is_unlinked]

        # pca
        PCs, model = allel.pca(gn_unlinked)
        npc = PCs.shape[1]
        colnames = [f"PC{i+1}" for i in range(npc)]
        df = pd.DataFrame(PCs, index=self.samples, columns=colnames)

        return df, model


class IBD:
    def __init__(self, genome: Genome = None, label=""):
        """ """
        self._df: pd.DataFrame = None
        self._ibd_format: str = None
        self._samples: pd.Series = None
        self._genome: Genome = genome
        self._supported_ibd_formats = ["treeibd"]
        self._label = label
        self._cov_df = None
        self._peaks_df = None
        self._flag_peaks_already_removed = False

    def set_genome(self, genome: Genome):
        self._genome = genome

    def read_ibd(
        self,
        ibd_fn_lst: List[str],
        chromosome_lst: List[Union[str, int]] = None,
        format: str = "treeibd",
        column_name_map: dict = None,
        rm_sample_name_suffix: bool = False,
        samples: List[Union[str, int]] = None,
    ):
        """read ibd from different formats
        - ibd_fn_lst: a list of IBD files. If all IBD segments are in a single
          file, just specify a single-element list to this argument
        - format: can be one of the following: 1. 'treeibd', 2. 'parquet'
        - samples: allow subsetting IBD by samples. If not None, only keep IBD
          shared between pairs within specified subset. If None, keep all IBD
          segments
        - column_name_map: used to rename column name after read into a dataframe
        """
        ibd = None

        # chromosomes
        if chromosome_lst is None:
            nchroms = len(ibd_fn_lst)
            chromosome_lst = list(range(1, 1 + nchroms))

        # format
        if format == "treeibd":
            self._ibd_format = "trueibd"
            ibd_lst = []
            for chromosome, ibd_fn in zip(chromosome_lst, ibd_fn_lst):
                ibd = pd.read_csv(ibd_fn, sep="\t")
                col_expected = [
                    "Id1",
                    "Id2",
                    "Start",
                    "End",
                    "HasMutation",
                    "Ancestor",
                    "Tmrca",
                ]
                assert pd.Series(col_expected).isin(ibd.columns).all()
                ibd["Chromosome"] = chromosome
                ibd_lst.append(ibd)
            ibd = pd.concat(ibd_lst, axis=0)
            if column_name_map:
                ibd.rename(columns=column_name_map, inplace=True)
        elif format == "parquet":
            self._ibd_format = "parquet"
            assert len(ibd_fn_lst) == 1
            ibd = pd.read_parquet(ibd_fn_lst[0])
            col_expected = "Id1	Id2	Chromosome	Start	End".split()
            assert pd.Series(col_expected).isin(ibd.columns).all()
            if column_name_map:
                ibd.rename(columns=column_name_map, inplace=True)
        else:
            raise Exception(f"{format} format is not implemented!")

        # remove sample name suffix:
        if rm_sample_name_suffix:
            assert pd.api.types.is_string_dtype(ibd.Id1)
            assert pd.api.types.is_string_dtype(ibd.Id2)
            ibd["Id1"] = ibd.Id1.str.replace("~.*$", "", regex=True)
            ibd["Id2"] = ibd.Id2.str.replace("~.*$", "", regex=True)

        # filter by samples
        if samples is not None:
            ibd: pd.DataFrame = ibd[ibd.Id1.isin(samples) & ibd.Id2.isin(samples)]

        # swap Id1 and Id2 if Id1 > Id2
        to_swap = ibd.Id1 > ibd.Id2

        tmp_id1 = np.where(to_swap, ibd.Id2.values, ibd.Id1.values)
        tmp_id2 = np.where(to_swap, ibd.Id1.values, ibd.Id2.values)

        ibd["Id1"] = tmp_id1
        ibd["Id2"] = tmp_id2

        # get unique sample series
        self._samples = pd.Series(
            np.unique(np.hstack([ibd.Id1.unique(), ibd.Id2.unique()]))
        )

        self._df = ibd

    def convert_to_heterozygous_diploids(self, remove_hbd=True, sep="@"):
        """
        Note: need to use a different separator that was used for remove_peaks
        and other steps that concatenate fields using the separator ':'
        """
        uniq_samples = self.get_samples_shared_ibd()

        # make number of haplotypes even
        if uniq_samples.size % 2 == 1:
            # omit one sample to make it an even number of haplotypes
            last = uniq_samples[-1]
            self._df = self._df[lambda df: ~((df.Id1 == last) | (df.Id2 == last))]
            uniq_samples = uniq_samples[:-1]

        # conversion: After sorting the unique sample names alphabetically,
        # take nearby two haplotypes to make a pseduo heterzyous diploid
        x = pd.Categorical(self._df.Id1, categories=uniq_samples).codes
        y = pd.Categorical(self._df.Id2, categories=uniq_samples).codes

        # make fake ids for pesudo heterzygous diploids
        uniq_samples = uniq_samples.astype(str)
        fake_samples = np.char.add(uniq_samples[::2], sep)
        fake_samples = np.char.add(fake_samples, uniq_samples[1::2])

        # reassign Id/Hap values
        self._df["Id1"] = fake_samples[x // 2]
        self._df["Hap1"] = x % 2
        self._df["Id2"] = fake_samples[y // 2]
        self._df["Hap2"] = y % 2

        # make sure (Sample1, Hap1) not larger than (Sample2, Hap2). This is
        # important for merge/flattening step
        condition1 = self._df.Id1 > self._df.Id2
        condition2 = (self._df.Id1 == self._df.Id2) & (self._df.Hap1 > self._df.Hap2)
        to_swap = condition1 | condition2

        tmp_id1 = np.where(to_swap, self._df.Id2.values, self._df.Id1.values)
        tmp_id2 = np.where(to_swap, self._df.Id1.values, self._df.Id2.values)
        tmp_Hap1 = np.where(to_swap, self._df.Hap2.values, self._df.Hap1.values)
        tmp_Hap2 = np.where(to_swap, self._df.Hap1.values, self._df.Hap2.values)

        self._df["Id1"] = tmp_id1
        self._df["Id2"] = tmp_id2
        self._df["Hap1"] = tmp_Hap1
        self._df["Hap2"] = tmp_Hap2

        # remove homozygosity by descent
        if remove_hbd:
            self._df = self._df[tmp_id1 != tmp_id2]

    def flatten_diploid_ibd(self, method="merge"):
        """method can be ["keep_hap_1_only",  "merge"]"""
        assert method in ["merge", "keep_hap_1_only"]
        if method == "keep_hap_1_only":
            self._df = self._df[lambda df: (df.Hap1 == 1) & (df.Hap2 == 1)]
        elif method == "merge":
            # save datatypes for restoring after merging
            dtype_id1 = self._df["Id1"].dtype
            dtype_id2 = self._df["Id2"].dtype
            dtype_chr = self._df["Chromosome"].dtype

            # fake Chromosome
            fake_chromosome = (
                self._df["Id1"]
                .astype(str)
                .str.cat(self._df[["Id2", "Chromosome"]].astype(str), sep=":")
            )
            # remove unnecessary columns
            self._df["Chromosome"] = fake_chromosome

            # merge ibdsegments and revert FakeChromosome to 'Id1', 'Id2',
            # 'Chromosome' columns

            # make bed file: 3 column, sorted bed file

            bed_file = pb.BedTool.from_dataframe(
                self._df[["Chromosome", "Start", "End"]].sort_values(
                    ["Chromosome", "Start"]
                )
            )

            # run pybedtools merge
            merged = bed_file.merge()

            # read bed file and convert dataframe
            df = merged.to_dataframe()
            df.columns = ["Chromosome", "Start", "End"]

            # restore the id1/id2/chromosome from the fake chromosome columns
            df[["Id1", "Id2", "Chromosome"]] = df["Chromosome"].str.split(
                ":", expand=True
            )
            df["Id1"] = df["Id1"].astype(dtype_id1)
            df["Id2"] = df["Id2"].astype(dtype_id2)
            df["Chromosome"] = df["Chromosome"].astype(dtype_chr)

            self._df = df[["Id1", "Id2", "Chromosome", "Start", "End"]]

        else:
            raise Exception(f"Method {method} not implemented")

    def filter_ibd_by_time(self, max_tmrca: float = None, min_tmrca: float = None):
        assert "Tmrca" in self._df.columns
        if max_tmrca:
            self._df = self._df[lambda df: df.Tmrca < max_tmrca]
        if min_tmrca:
            self._df = self._df[lambda df: df.Tmrca >= min_tmrca]

    def filter_ibd_by_mutation(
        self, exclude_ibd_carrying_mutation=True, exclude_ibd_without_mutation=False
    ):
        if exclude_ibd_without_mutation + exclude_ibd_carrying_mutation == 2:
            raise Exception("Can not exclude both types")
        assert "HasMutation" in self._df.columns

        sel = np.repeat(True, self._df.shape[0])
        if exclude_ibd_carrying_mutation:
            sel = self._df.HasMutation == 0
        if exclude_ibd_without_mutation:
            sel = self._df.HasMutation == 1

        self._df = self._df[sel]

    def filter_ibd_by_length(self, min_seg_cm: float = None, max_seg_cm: float = None):
        ibd = self._df
        cm = self._genome._gmap.get_length_in_cm(
            ibd.Chromosome.values, ibd.Start.values, ibd.End.values
        )

        # filter ibd segment by length
        sel = np.repeat(True, ibd.shape[0])
        if min_seg_cm:
            sel1 = cm >= min_seg_cm
            sel = sel & sel1
        if max_seg_cm:
            sel2 = cm < max_seg_cm
            sel = sel & sel2

        self._df = ibd[sel]

    def calc_ibd_cov(self, step_in_cm=0.1):
        assert self._genome is not None

        # get sampling bed file
        bp_per_cm = self._genome._bp_per_cm
        chr_df = self._genome._chr_df[["Chromosome", "ChromLength"]]

        step = int(step_in_cm * bp_per_cm)
        chr_arr_lst = []
        start_arr_lst = []
        for chr, chrlen in chr_df.itertuples(index=False):
            start_arr = np.arange(1, chrlen, step=step)
            chr_arr = np.repeat(chr, start_arr.size)

            start_arr_lst.append(start_arr)
            chr_arr_lst.append(chr_arr)

        starts = np.hstack(start_arr_lst)
        chroms = np.hstack(chr_arr_lst)

        sp_df = pd.DataFrame({"Chromosome": chroms, "Start": starts, "End": starts + 1})
        sp_df["Chromosome"] = sp_df.Chromosome.astype(str)
        sp_df.sort_values(["Chromosome", "Start"], inplace=True)
        sp_bed = pb.BedTool.from_dataframe(sp_df)

        # get ibd bed file (ignoring sample Ids)

        # NOTE: need to sort chromosomes according to the string values (it can
        # be integer type from simulated data) as bedtools sorts chromosomes as
        # strings
        ibd_df = (
            self._df[["Chromosome", "Start", "End"]]
            .assign(Chromosome=lambda df: df.Chromosome.astype(str))
            .sort_values(["Chromosome", "Start"])
        )
        ibd_bed = pb.BedTool.from_dataframe(ibd_df)

        # get coverage via bedtools intersect -c --sorted
        res_bed = sp_bed.intersect(ibd_bed, c=True, sorted=True)
        cov_df = res_bed.to_dataframe()
        cov_df.columns = ["Chromosome", "Start", "End", "Coverage"]

        # add Gw bp positions
        gwstarts = self._genome.get_genome_wide_coords(cov_df.Chromosome, cov_df.Start)
        cov_df["GwStart"] = gwstarts

        self._cov_df = cov_df.sort_values(["GwStart"])

    def get_coverage_within_and_outside_peaks(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert self._cov_df is not None
        assert self._peaks_df is not None

        # bed files
        cov = pb.BedTool.from_dataframe(
            self._cov_df[["Chromosome", "Start", "End", "Coverage"]]
        )
        peaks = pb.BedTool.from_dataframe(
            (self._peaks_df[["Chromosome", "Start", "End"]])
        )

        # call bed subtract
        outside = cov.subtract(peaks).to_dataframe()
        outside.columns = ["Chromosome", "Start", "End", "Coverage"]

        # call bed intersect
        within = cov.intersect(peaks).to_dataframe()
        within.columns = ["Chromosome", "Start", "End", "Coverage"]

        return within, outside

    def plot_coverage(self, ax=None, label="", plot_proportions=True):
        assert self._cov_df is not None

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3))

        # shortcuts
        chr_boundaries = self._genome.get_chromosome_boundaries()
        chr_centers = self._genome._chr_df["GwChromCenter"].to_numpy()
        chr_names = self._genome._chr_df["Chromosome"].to_numpy()
        n_samples = self.get_samples_shared_ibd().size
        n_pairs = n_samples * (n_samples - 1) / 2

        # plot coverage
        cov_df = self._cov_df[lambda df: df.Coverage > 0]  # omit zeros
        ax.plot("GwStart", "Coverage", data=cov_df, label=label)
        ax.set_xticks(chr_centers)
        ax.set_xticklabels(chr_names)
        ax.set_ylabel("IBD coverage")

        # chromosomes boundaries
        for x in chr_boundaries:
            ax.axvline(x, linestyle="--", color="grey")

        # mark IBD proportions
        if plot_proportions:
            ax_twin = ax.twinx()
            ylim = ax.get_ylim()
            ax_twin.set_ylim(ylim[0] / n_pairs, ylim[1] / n_pairs)
            ax_twin.set_ylabel("IBD proportions")

        # allow show label in legend
        if label != "":
            ax.legend()

        return ax

    def calc_ibd_length_in_cm(self) -> np.ndarray:
        return self._genome._gmap.get_length_in_cm(
            self._df.Chromosome.values, self._df.Start.values, self._df.End.values
        )

    @staticmethod
    def _find_peaks(cov_df: pd.DataFrame, chr_df):
        """find peaks in IBD coverage profile using bedtools/pybedtools where peaks
        are defined by regions > chromosome median + 1.5 IQR with extension to
        nearest point crossing the median line"""
        peaks_lst = []
        stats = {"Chromosome": [], "Median": [], "Thres": []}
        for chromosome, chrom_cov_df in cov_df.groupby("Chromosome"):
            # infer the step
            s = chrom_cov_df.Start.values
            assert np.all((s[1:-1] - s[:-2]) == (s[1:-1] - s[:-2]).mean())
            step = s[1] - s[0]
            # threshold
            q5, q25, q50, q75, q95 = np.quantile(
                chrom_cov_df.Coverage, q=[0.05, 0.25, 0.5, 0.75, 0.95]
            )
            iqr = q75 - q25
            trim_mean = chrom_cov_df.Coverage[lambda s: (s >= q5) & (s < q95)].mean()
            trim_std = chrom_cov_df.Coverage[lambda s: (s >= q5) & (s < q95)].std()
            # core regions
            core_df = chrom_cov_df.loc[
                lambda x: (
                    # x.Coverage > q75 + 1.5 * iqr
                    x.Coverage
                    > trim_mean + 2 * trim_std
                )
            ]
            core_bed = pb.BedTool.from_dataframe(core_df.iloc[:, :3]).merge(d=step)
            # extension region
            ext_df = chrom_cov_df.loc[lambda x: x.Coverage > q50]
            ext_bed = pb.BedTool.from_dataframe(ext_df.iloc[:, :3]).merge(d=step)
            # extension region intersect with core regions
            peaks = ext_bed.intersect(core_bed, wa=True).merge(d=step).to_dataframe()
            if peaks.shape[0] > 0:
                peaks["Median"] = q50
                peaks["Thres"] = q50 + iqr
            peaks_lst.append(peaks)
            stats["Chromosome"].append(chromosome)
            stats["Median"].append(q50)
            stats["Thres"].append(q50 + 1.5 * iqr)
        peaks_df = pd.concat(peaks_lst)
        if peaks_df.shape[0] > 0:
            peaks_df.columns = ["Chromosome", "Start", "End", "Median", "Thres"]
            # make genome-wide coordinates
            chr_df_tmp = chr_df[["Chromosome", "GwChromStart"]]
            peaks_df = peaks_df.merge(chr_df_tmp, how="left", on="Chromosome")
            peaks_df["GwStart"] = peaks_df.Start + peaks_df.GwChromStart
            peaks_df["GwEnd"] = peaks_df.End + peaks_df.GwChromStart
        # stats
        chr_df_tmp = chr_df[["Chromosome", "GwChromStart", "GwChromEnd"]]
        stats_df = pd.DataFrame(stats).merge(chr_df_tmp, how="left", on="Chromosome")
        return peaks_df, stats_df

    def find_peaks(self):
        assert self._genome is not None
        assert self._cov_df is not None
        chr_df = self._genome._chr_df
        cov_df_4col = self._cov_df[["Chromosome", "Start", "End", "Coverage"]]
        self._peaks_df, _ = IBD._find_peaks(cov_df_4col, chr_df)

    @staticmethod
    def _remove_peaks(ibd_in: pd.DataFrame, peaks_df):
        ibd = ibd_in.copy()

        # backup datatype
        id1_dtype = ibd["Id1"].dtype
        id2_dytpe = ibd["Id2"].dtype
        chr_dtype = ibd["Chromosome"].dtype

        ibd["Info"] = ibd.Id1.astype(str).str.cat(
            ibd[["Id2", "Chromosome"]].astype(str), sep=":"
        )
        ibd["Score"] = 1
        ibd["Strand"] = "*"
        ibd["ChromStr"] = ibd.Chromosome.astype(str)
        ibd_bed = pb.BedTool.from_dataframe(
            ibd[["ChromStr", "Start", "End", "Info", "Score", "Strand"]].sort_values(
                by=["ChromStr", "Start"], axis=0
            )
        )
        # only the first 3 columns are needed
        peaks_bed = pb.BedTool.from_dataframe(peaks_df.iloc[:, range(3)])

        # TODO: parallelize per chromosome
        ibd_rm_peaks = ibd_bed.subtract(peaks_bed).to_dataframe()
        ibd_rm_peaks.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
        ibd_rm_peaks[["Id1", "Id2"]] = (
            ibd_rm_peaks["Name"].str.split(":", expand=True).iloc[:, :2]
        )
        ibd_rm_peaks = ibd_rm_peaks[["Id1", "Id2", "Chromosome", "Start", "End"]]

        # resume datatypes
        ibd_rm_peaks["Id1"] = ibd_rm_peaks["Id1"].astype(id1_dtype)
        ibd_rm_peaks["Id2"] = ibd_rm_peaks["Id2"].astype(id2_dytpe)
        ibd_rm_peaks["Chromosome"] = ibd_rm_peaks["Chromosome"].astype(chr_dtype)

        # filter short segment after removing
        # too_short = ibd_rm_peaks.End - ibd_rm_peaks.Start < 2 * bp_per_cm
        # ibd_rm_peaks = ibd_rm_peaks[~too_short].copy()
        return ibd_rm_peaks

    def remove_peaks(self, min_seg_cm: float = 2.0, rm_short_seg=True):
        assert self._peaks_df is not None
        assert self._genome is not None

        ibd = self._remove_peaks(self._df, self._peaks_df)
        cm = self._genome._gmap.get_length_in_cm(ibd.Chromosome, ibd.Start, ibd.End)
        if rm_short_seg:
            ibd = ibd[cm >= min_seg_cm]

        self._df = ibd.copy()
        self._flag_peaks_already_removed = True

    # TODO: test the copied code
    @staticmethod
    def _split_chromosomes_at_peaks(ibd_in, peaks_df, chrlen_df):
        ibd = ibd_in.copy()

        # add genome wide chromsome start and end coordinates
        chrlen_df["GwChromStart"] = [0] + chrlen_df.ChromLength.cumsum().iloc[
            :-1
        ].tolist()
        chrlen_df["GwChromStart"] += 1
        chrlen_df["GwChromEnd"] = chrlen_df.ChromLength.cumsum()

        # np array of chromsomes' start coordinate (gw)
        gw_chrom_start = chrlen_df.GwChromStart.values

        # get all contig edges: real chromsome boundaries and peaks boundaries
        contig_edges = chrlen_df.GwChromStart.tolist()
        contig_edges.append(chrlen_df.GwChromEnd.tolist()[-1])
        contig_edges.extend(peaks_df.GwStart.tolist())
        contig_edges.extend(peaks_df.GwEnd.tolist())
        contig_edges = np.unique(np.array(contig_edges))  # also sorted by this function

        # shift IBD segment start and end to genome-wide coordinates
        ibd["Start"] = gw_chrom_start[ibd.Chromosome - 1] + ibd.Start
        ibd["End"] = gw_chrom_start[ibd.Chromosome - 1] + ibd.End

        # sort IBD segment accoriding gw_start
        ibd.sort_values("Start", inplace=True)

        # modifiy Chrom to contig id
        ibd["Chromosome"] = np.searchsorted(contig_edges, ibd.Start, "right")

        # shift all IBD segments so that each contig starts at 1
        ibd["Start"] = ibd.Start - contig_edges[ibd.Chromosome.values - 1]
        ibd["End"] = ibd.End - contig_edges[ibd.Chromosome.values - 1]

        return ibd

    def cut_and_split_ibd(self):
        """
        assume ibd peaks have been removed
        will not modify self.ibd

        1. split chromosome
        2. shift coordinates for ibd/cov/dataframe

        @return a new IBD dataframe
        """
        assert self._flag_peaks_already_removed
        assert self._peaks_df is not None
        return self._split_chromosomes_at_peaks(
            self._df, self._peaks_df, self._genome._chr_df
        )

    def make_ibd_matrix(
        self,
        assignment: Union[pd.Series, None] = None,
        min_seg_cm: float = None,
        max_seg_cm: float = None,
        min_gw_ibd_cm: float = None,
        max_gw_ibd_cm: float = None,
    ):
        """Make IBD matrix at a) sample level (if `assignment` is None), or
                              b) subpopulation level (if `assignment is a pd.Series)

        @assignment: 1. None --> caculate sample level total IBD matrix;
                     2. pd.Series, of which index are sample Ids, values are population assignment --> calculate
                       subpopulation level total IBD matrix
        @min_seg_cm: if not None, remove any segment less than this value
        @max_seg_cm: if not None remove any segment longer than or equal to this value
        @min_gw_ibd_cm: if not None, mask any element smaller than this in the raw total IBD matrix
        @max_gw_ibd_cm: if not None, mask any element larger than or equal to this value
        """
        cols = ["Id1", "Id2", "Chromosome", "Start", "End"]
        ibd = self._df[cols].copy()
        cm = self._genome._gmap.get_length_in_cm(
            ibd.Chromosome.values, ibd.Start.values, ibd.End.values
        )

        ibd["Cm"] = cm

        # filter ibd segment by length
        sel = np.repeat(True, ibd.shape[0])
        if min_seg_cm:
            sel1 = ibd.Cm >= min_seg_cm
            sel = sel & sel1
        if max_seg_cm:
            sel2 = ibd.Cm < max_seg_cm
            sel = sel & sel2

        ibd = ibd[sel]

        # aggregate over sample pairs
        names = self._samples
        ibd["Id1"] = pd.Categorical(ibd.Id1, categories=names)
        ibd["Id2"] = pd.Categorical(ibd.Id2, categories=names)
        ibd = ibd.groupby((["Id1", "Id2"]))["Cm"].sum().reset_index()

        # gw total ibd filter
        sel = np.repeat(True, ibd.shape[0])
        if min_gw_ibd_cm:
            sel3 = ibd.Cm >= min_gw_ibd_cm
            sel = sel & sel3
        if max_gw_ibd_cm:
            sel4 = ibd.Cm < max_gw_ibd_cm
            sel = sel & sel4
        ibd = ibd[sel]

        # make ibd matrix
        if assignment is None:
            # make sample total ibd matrix
            names = self._samples
            M = np.zeros(shape=(names.size, names.size), dtype=np.float64)
            M[ibd.Id1.cat.codes.values, ibd.Id2.cat.codes.values] = ibd.Cm.values
            M = M + M.T
        else:
            # aggregate over subpopulation
            names, counts = np.unique(assignment, return_counts=True)
            names = pd.Series(names)
            ibd["Assignment1"] = ibd.Id1.map(assignment)
            ibd["Assignment2"] = ibd.Id2.map(assignment)
            ibd["Assignment1"] = pd.Categorical(ibd["Assignment1"], categories=names)
            ibd["Assignment2"] = pd.Categorical(ibd["Assignment2"], categories=names)
            ibd = (
                ibd.groupby((["Assignment1", "Assignment2"]))["Cm"].sum().reset_index()
            )

            # make subpopulation level total ibd
            M = np.zeros(shape=(names.size, names.size), dtype=np.float64)
            M[
                ibd.Assignment1.cat.codes.values, ibd.Assignment2.cat.codes.values
            ] = ibd.Cm.values
            M = M + M.T

            # average and deal with the case where there is only sample in a cluster
            N = counts.reshape(1, -1) * counts.reshape(-1, 1)
            np.fill_diagonal(
                N, counts * (counts - 1)
            )  # not divided by 2 to account for M + M.T

            M[N != 0] = M[N != 0] / N[N != 0]
            M[N == 0] = M[N != 0].max()

        # make dataframe
        df = pd.DataFrame(M, columns=names, index=names)
        return df

    @staticmethod
    def clust_ibd_matrix(ibd_mat_df: pd.DataFrame):

        names = ibd_mat_df.columns.sort_values().values.copy()

        M = ibd_mat_df.loc[names, names].values.copy()

        # transform data
        mininum = M.min()
        maximum = M.max()
        np.add(M, -mininum + 1e-3, out=M)
        np.divide(maximum, M, out=M)

        # clustering
        model = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
        )
        model = model.fit(M)

        # make linkage matrix
        #  create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        #  build the linkage matrix
        lnk_matrix = np.column_stack(
            [model.children_, model.distances_ / 2, counts]
        ).astype(float)

        # leaves orders
        leaves_list = sch.leaves_list(lnk_matrix)
        sample_reordered = pd.Series(names[leaves_list])

        return sample_reordered, lnk_matrix

    # using igraph to remove highly one of the highly related sample pairs
    # get highly related samples ( not used )
    @staticmethod
    def _get_highly_related_samples_to_remove(M, ids, threshold=625):
        """
        Generate a list of sample ids to remove in order to get rid of high relatedness
        @params ids: a list of names in the same order as the relatedness matrix
        @params threshold: any number above the theshold will be treated to closedly related
        @params M: orginal relatedness matrix
        @return: a list of sample ids
        """

        # make a graph to represent high reletedness
        relate_mat = np.zeros(M.shape, dtype=np.uint8)
        relate_mat[M >= threshold] = 1
        rg = igraph.Graph.Adjacency(relate_mat)
        rg.vs["name"] = ids

        # remove non-connected from the graph
        deg = rg.vs.degree()
        zero_degree_nodes = [i for i, degree in enumerate(deg) if degree == 0]
        rg.delete_vertices(zero_degree_nodes)

        num_involved_samples = len(rg.vs)

        if num_involved_samples == 0:
            return []

        # list of samples names to be removed
        deleted_samples = []

        # Remove the samples that have the most connections
        while True:
            deg = rg.vs.degree()
            ix = np.argmax(deg)
            max_deg = deg[ix]
            if max_deg > 0:
                sample_name = rg.vs["name"][ix]
                rg.delete_vertices(ix)
                deleted_samples.append(sample_name)
            else:
                break

        return deleted_samples

    def get_unrelated_samples(self, ibdmat):
        assert (ibdmat.columns == ibdmat.index).all

        genome_size_gw = self._genome.get_genome_size_cm()
        m = ibdmat.values
        ids = ibdmat.columns.tolist()
        thres = genome_size_gw / 2
        to_del = self._get_highly_related_samples_to_remove(m, ids, thres)

        samples = self.get_samples_shared_ibd()
        to_keep = pd.Series(samples)[lambda s: ~s.isin(to_del)]

        return to_keep

    def subset_ibd_by_samples(self, subset_samples: pd.Series):
        self._df = self._df.loc[
            lambda df: df.Id1.isin(subset_samples) & df.Id2.isin(subset_samples)
        ]
        self._samples = pd.Series(self.get_samples_shared_ibd())

    @staticmethod
    def call_infomap_get_member_df(
        M: pd.DataFrame, meta, trials=500, seed=12345, transform=None
    ):
        """
        @transform: in ["square", "cube", None]
        """
        names = M.index.to_list()
        assert (M.index == M.columns).all()

        M1 = M.copy()
        # transform matrix
        assert transform in ["square", "cube", None]
        if transform == "square":
            M1 = M1 * M1
        elif transform == "cube":
            M1 = M1 * M1 * M1
        else:
            pass

        # make graph
        g = igraph.Graph.Weighted_Adjacency(M1.values)
        g.vs["name"] = names

        # call infomap
        random.seed(seed)
        vc = igraph.Graph.community_infomap(g, edge_weights="weight", trials=trials)

        # get vc.membership
        member_df = pd.DataFrame(
            {
                "SampleLong": g.vs["name"],
                "Membership": vc.membership,
                "Degree": g.vs.degree(),
            }
        )
        # add community size and rank
        comm_size = (
            member_df.Membership.value_counts()
            .reset_index()
            .rename(columns={"index": "Membership", "Membership": "Size"})
        )
        comm_size["Rank"] = np.arange(comm_size.shape[0])
        member_df = member_df.merge(comm_size, on="Membership", how="left")
        # add meta infommation
        if pd.api.types.is_string_dtype(member_df.SampleLong):
            member_df["Sample"] = member_df.SampleLong.str.replace(
                "~.*$", "", regex=True
            )
        else:
            member_df["Sample"] = member_df.SampleLong

        member_df = member_df.merge(meta, on="Sample", how="left")

        return member_df

    def duplicate(self, new_label="") -> "IBD":
        """Duplicate the whole IBD object"""
        new_ibd = IBD()
        new_ibd._df = self._df.copy()
        new_ibd._ibd_format = self._ibd_format
        new_ibd._genome = self._genome
        new_ibd._supported_ibd_formats = self._supported_ibd_formats
        new_ibd._label = new_label
        new_ibd._cov_df = self._cov_df
        new_ibd._samples = self._samples
        new_ibd._peaks_df = self._peaks_df

        return new_ibd

    def is_valid() -> bool:
        raise NotImplemented

    def pickle(self):
        raise NotImplemented

    def get_samples_shared_ibd(self) -> np.ndarray:
        s1 = self._df.Id1.unique()
        s2 = self._df.Id2.unique()
        return np.unique(np.hstack([s1, s2]))

    def __str__(self):
        # print df head
        return f"""\
        Number of IBD segments:                 {self._df.shape[0]}
        Number of Samples (shared ibd):         {self.get_samples_shared_ibd().size}
        Number of Chromosomes:                  {self._df.Chromosome.unique().size}
        Head of the IBD dataframe: \n{self._df.head(3)}
        """


class SnpEffResHandler:
    def __init__(self):
        self.sample_mutation_table_fn = "/autofs/chib/toconnor_grp/bing/20220314_analyze_joint_call_pf/snpeff/ESEA_annot/sample_mutation_table.tsv"
        mut = pd.read_csv(self.sample_mutation_table_fn, sep="\t")
        mut["Sample"] = mut.Sample.str.replace("~.*$", "", regex=True)
        self.mut = mut

    @staticmethod
    def get_common_mutations():
        # TODO: need to update this list
        return """dhfr:I164L
        mdr1:Y184F
        aat1:Q454E aat1:S258L
        crt:M74I crt:N75D crt:N326S crt:I356T
        PF3D7_0720700:C1484F
        dhps:G437A dhps:A581G
        fd:D193Y
        k13:C580Y k13:R539T k13:Y493H
        arps10:V127M
        """.split()

    def calc_group_allele_frequency(
        self,
        sample2group: pd.Series,
        sel_mutations: List[str] = None,
        sel_groups: List[str] = None,
    ):
        self.mut["Group"] = self.mut.Sample.map(sample2group).values.astype(int)
        commu_afreq = self.mut.groupby("Group").mean().T
        if sel_mutations is not None:
            commu_afreq = commu_afreq.loc[sel_mutations, :]
        if sel_groups is not None:
            commu_afreq = commu_afreq.loc[:, sel_groups]
        return commu_afreq


if __name__ == "__main__":
    # test_gmap()
    # test_ibd()
    pass
