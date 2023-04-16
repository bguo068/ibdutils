from ..utils.ibdutils import IBD

from subprocess import run
from pathlib import Path
import requests


class IbdNeRunner:
    def __init__(
        self, ibd: IBD, input_folder, output_folder, minregion=10, mincm=2
    ) -> None:
        self.ibd = ibd.duplicate(ibd._label)
        self.input_dir = input_folder
        self.output_dir = output_folder

        self.input_ibd_fn = f"{input_folder}/{ibd._label}.ibd.gz"
        self.input_map_fn = f"{input_folder}/{ibd._label}.map"
        self.ouptut_ne_fn = f"{output_folder}/{ibd._label}.ne"

        self.ibdne_jar_fn = self.output_dir + "/ibdne.jar"
        self.minregion = minregion
        self.mincm = mincm

        Path(input_folder).mkdir(parents=True, exist_ok=True)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    def prepare_ibdne_input(self, bp_per_cm=15_000):
        """
        for ibdne called on peak-removed ibd,
        cut_and_split_ibd should be called before this method
        """

        # add Hap1, Hap2 columns
        if "Hap1" not in self.ibd._df.columns:
            self.ibd._df["Hap1"] = 0
        if "Hap2" not in self.ibd._df.columns:
            self.ibd._df["Hap2"] = 0
        if "Cm" not in self.ibd._df.columns:
            # NOTE: for rmpeaks groups, this has to be called before cut_and_split_ibd
            # code has been added to ensure this see `IBD.cut_and_split_ibd` method
            self.ibd._df["Cm"] = self.ibd.calc_ibd_length_in_cm()

        # write xx.ibd.gz file
        col_order = ["Id1", "Hap1", "Id2", "Hap2", "Chromosome", "Start", "End", "Cm"]
        self.ibd._df[col_order].to_csv(
            self.input_ibd_fn, sep="\t", index=None, header=None, compression="gzip"
        )

        """
        Chromosome code. PLINK 1.9 also permits contig names here, but most older programs do not.
        Variant identifier
        Position in morgans or centimorgans (optional; also safe to use dummy value of '0')
        Base-pair coordinate
        """

        lines = []
        for chr, len in self.ibd._df.groupby(["Chromosome"])["End"].max().items():
            lines.append(f"{chr} . {1} 0\n")
            lines.append(f"{chr} . {len/bp_per_cm} {len}\n")

        with open(self.input_map_fn, "w") as f:
            f.writelines(lines)

    def download_ibdne_jar_file(self):
        url = "https://faculty.washington.edu/browning/ibdne/ibdne.23Apr20.ae9.jar"

        response = requests.get(url)
        with open(self.ibdne_jar_fn, "wb") as f:
            f.write(response.content)

    def run(self, nthreads=24, mem_gb: int = 20, dry_run=False):
        # make sure files are present
        if (not Path(self.input_map_fn).exists()) or (
            not Path(self.input_ibd_fn).exists()
        ):
            self.prepare_ibdne_input()

        output_prefix = self.ouptut_ne_fn.replace(".ne", "")
        output_script = self.ouptut_ne_fn.replace(".ne", ".sh")

        # make sure jar file is present
        if not Path(self.ibdne_jar_fn).exists():
            self.download_ibdne_jar_file()

        cmd = f"""
            #! /usr/bin/env bash
            zcat {self.input_ibd_fn} \\
                    | java -Xmx{mem_gb}G -jar {self.ibdne_jar_fn} \\
                    map={self.input_map_fn} \\
                    out={output_prefix} \\
                    nthreads={nthreads} \\
                    minregion={self.minregion} \\
                    mincm={self.mincm}
        """
        with open(output_script, "w") as f:
            f.write(cmd)

        if not dry_run:
            res = run(cmd, shell=True, check=True, capture_output=True, text=True)

            # ibdne does not always return nonzero when error is detected.
            # Mannually detect error here
            assert "ERROR" not in res.stderr
            assert "ERROR" not in res.stdout

    def get_res_ne_fn(self):
        assert Path(self.output_dir).exists()
        return self.ouptut_ne_fn
