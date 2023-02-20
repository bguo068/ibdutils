from ..utils.ibdutils import IBD
import numpy as np
import pandas as pd
from pathlib import Path
import hapne
from configparser import ConfigParser


class HapNeIBDRunner:
    """
    Wrapper to run HapNe-IBD

    adapted from script:
    /home/bing.guo/phd_milestones/sel_on_demo_paper/analyses/hapne/run_hapne_ibd_with_simudata.py
    """

    def __init__(
        self,
        ibd: IBD,
        id_is_pseudo_diploid: bool,
        outdir_root: str,
        min_contig_len_cm=20,
        pop_name="pop1",
    ) -> None:
        # make sure IBD are in the correct state:
        assert ibd._df is not None
        if ibd._flag_peaks_already_removed:
            assert ibd._peaks_df is not None

        self.ibd = ibd
        self.outdir = outdir_root
        self.min_contig_len_cm = min_contig_len_cm
        self.pop_name = pop_name
        self.id_is_pseudo_diploid = id_is_pseudo_diploid
        self.regions = None
        self.ibd_hist_dict = None
        self.proc_ibd = None  # will be filled during the calculation of regions

    def _calc_regions_from_ibd(self):
        """Construct the region dataframe and remove chromosomes that are too short"""

        if self.ibd._flag_peaks_already_removed:
            # convert chromsomes into contig
            ibd_df = self.ibd.cut_and_split_ibd()
        else:
            ibd_df = self.ibd._df.copy()

        colnames = "CHR FROM_BP TO_BP NAME LENGTH".split()
        agg_funs = {"Start": min, "End": max}
        col_map = {"Chromosome": "CHR", "Start": "FROM_BP", "End": "TO_BP"}

        regions = (
            # get chromosome smallest start and largest end
            ibd_df.groupby("Chromosome")
            .agg(agg_funs)
            .reset_index()
            # rename colums
            .rename(columns=col_map)
            # add name and length columns
            .assign(
                NAME=lambda df: "chr"
                + df.CHR.astype(str).str.cat(
                    df[["FROM_BP", "TO_BP"]].astype(str), sep="."
                ),
                # TODO: need to use the gnome map to do this
                LENGTH=lambda df: (df.TO_BP - df.FROM_BP) / 15000,
            )
            # remove chromosome that is too short
            [lambda df: df.LENGTH >= self.min_contig_len_cm]
            # reorder columns
            [colnames]
        )

        self.regions = regions
        self.proc_ibd = ibd_df

    def _calc_hist_from_ibd(self):
        assert (
            self.regions is not None
        ), "this function relies on `self.region`. Call`self._calc_regions_from_ibd()` first"

        ibd_df = self.proc_ibd.copy()

        if "Cm" not in ibd_df.columns:
            ibd_df["Cm"] = self.ibd.calc_ibd_length_in_cm()

        ibd_df["Morgan"] = ibd_df.Cm / 100

        max_len = ibd_df.Morgan.max()

        bins = np.arange(0, max_len, step=0.005)

        ibd_hist = {}
        for chr, name in self.regions[["CHR", "NAME"]].itertuples(index=False):
            counts = pd.cut(
                ibd_df[ibd_df.Chromosome == chr].Morgan, bins=bins, right=False
            ).value_counts()

            index = counts.index
            left = index.categories[index.codes].left.values
            right = index.categories[index.codes].right.values

            hist_df = pd.DataFrame(
                {
                    "Left": left,
                    "Right": right,
                    "Count": counts.values,
                }
            ).sort_values("Left")
            ibd_hist[name] = hist_df

        self.ibd_hist_dict = ibd_hist

    def _get_hapne_ibd_config(self, min_cm=2.0):

        outdir = self.outdir
        Path(outdir).mkdir(parents=True, exist_ok=True)
        config_dict = {
            "CONFIG": {
                "pseudo_diploid": self.id_is_pseudo_diploid,
                "population_name": self.pop_name,
                "u_min": min_cm / 100,  # to morgan
                "nb_samples":
                # TODO, # internal treated as no. of diploid
                self.ibd.get_samples_shared_ibd().size // 2,
                "output_folder": outdir,
            }
        }
        config = ConfigParser()
        config.read_dict(config_dict)
        return config

    def _get_path_for_regions(self):
        region_fn = self.outdir + "/regions.txt"
        Path(region_fn).parent.mkdir(parents=True, exist_ok=True)
        return region_fn

    def _get_path_for_ibd_hist(self, contig):
        hist_fn = self.outdir + f"/HIST/{contig}.ibd.hist"
        Path(hist_fn).parent.mkdir(parents=True, exist_ok=True)
        return hist_fn

    def _prepare_input(self):

        region_fn = self._get_path_for_regions()
        self.regions.to_csv(region_fn, sep="\t", index=None)

        for contig, hist in self.ibd_hist_dict.items():
            hist_fn = self._get_path_for_ibd_hist(contig)
            hist.to_csv(hist_fn, sep="\t", header=None, index=None)

    def run(self, dry_run=False, min_cm=2.5):
        # prepare input
        self._calc_regions_from_ibd()
        self._calc_hist_from_ibd()
        self._prepare_input()
        region_fn = self._get_path_for_regions()
        config = self._get_hapne_ibd_config(min_cm=min_cm)

        if dry_run:
            return

        # run
        hapne.utils.set_regions(region_fn)
        hapne.hapne_ibd(config)

        print(f"out_dir: {self.outdir}")

    def is_res_ne_fn_existing(self):
        path = self.outdir + "/HapNe/hapne.csv"
        return Path(path).exists()

    def get_res_ne_fn(self):
        path = self.outdir + "/HapNe/hapne.csv"
        assert Path(path).exists()
        return path
