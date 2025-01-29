import argparse
import textwrap

import configargparse

from falcon import __version__


class NewlineTextHelpFormatter(argparse.HelpFormatter):

    def _fill_text(self, text, width, indent):
        return "\n".join(
            textwrap.fill(
                line,
                width,
                initial_indent=indent,
                subsequent_indent=indent,
                replace_whitespace=False,
            ).strip()
            for line in text.splitlines(keepends=True)
        )


class Config:
    """
    Commandline and file-based configuration.

    Configuration settings can be specified in a config.ini file (by default in
    the working directory), or as command-line arguments.
    """

    def __init__(self) -> None:
        """
        Initialize the configuration settings and provide sensible default
        values if possible.
        """

        self._parser = configargparse.ArgParser(
            description=f"falcon: Fast spectrum clustering using nearest "
            f"neighbor searching\n"
            f"==============================================="
            f"==================\n\n"
            f"falcon version {__version__}\n\n"
            f"Official code website: "
            f"https://github.com/bittremieux/falcon\n\n",
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=NewlineTextHelpFormatter,
        )

        # IO
        self._parser.add_argument(
            "input_filenames",
            nargs="+",
            help="Input peak files (supported formats: .mzML, .mzXML, .MGF).",
        )
        self._parser.add_argument("output_filename", help="Output file name.")
        self._parser.add_argument(
            "--work_dir",
            default=None,
            help="Working directory (default: temporary directory).",
        )
        self._parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing results (default: don't overwrite).",
        )
        self._parser.add_argument(
            "--export_representatives",
            action="store_true",
            help="Export cluster representatives to an MGF file "
            "(default: no export).",
        )

        # CLUSTERING
        self._parser.add_argument(
            "--precursor_tol",
            nargs=2,
            default=[20, "ppm"],
            help="Precursor tolerance mass and mode (default: 20 ppm). "
            'Mode should be either "ppm" or "Da".',
        )
        self._parser.add_argument(
            "--rt_tol",
            type=float,
            default=None,
            help="Retention time tolerance (default: no retention time "
            "filtering).",
        )
        self._parser.add_argument(
            "--fragment_tol",
            type=float,
            default=0.05,
            help="Fragment mass tolerance in m/z (default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--linkage",
            type=str,
            default="complete",
            help="Linkage criterion for hierarchical clustering "
            "(default: %(default)s). Should be one of "
            "'single', 'complete', 'average'.",
        )
        self._parser.add_argument(
            "--distance_threshold",
            type=float,
            default=0.1,
            help="The distance threshold parameter (cosine distance) for clustering "
            "(default: %(default)s). Relevant cosine distance thresholds "
            "are typically between 0.05 and 0.30.",
        )
        self._parser.add_argument(
            "--min_matched_peaks",
            type=int,
            default=0,
            help="Minimum number of matched peaks to consider the spectra similar "
            "(default: %(default)s). Typically 6 for metabolomics data.",
        )
        self._parser.add_argument(
            "--consensus_method",
            type=str,
            default="medoid",
            help="Method to use for consensus spectrum computation "
            "(default: %(default)s). Should be one of 'medoid', 'average'.",
        )
        self._parser.add_argument(
            "--outlier_cutoff_lower",
            type=float,
            default=1.5,
            help="Number of standard deviations below the median for outlier rejection "
            "(default: %(default)s). Only used when consensus_method='average'.",
        )
        self._parser.add_argument(
            "--outlier_cutoff_upper",
            type=float,
            default=1.5,
            help="Number of standard deviations above the median for outlier rejection "
            "(default: %(default)s). Only used when consensus_method='average'.",
        )
        self._parser.add_argument(
            "--batch_size",
            type=int,
            default=2**15,
            help="Batch size for clustering (default: %(default)s).",
        )

        # PREPROCESSING
        self._parser.add_argument(
            "--min_peaks",
            default=5,
            type=int,
            help="Discard spectra with fewer than this number of peaks "
            "(default: %(default)s).",
        )
        self._parser.add_argument(
            "--min_mz_range",
            default=250.0,
            type=float,
            help="Discard spectra with a smaller mass range "
            "(default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--min_mz",
            default=101.0,
            type=float,
            help="Minimum peak m/z value (inclusive, "
            "default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--max_mz",
            default=1500.0,
            type=float,
            help="Maximum peak m/z value (inclusive, "
            "default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--remove_precursor_tol",
            default=1.5,
            type=float,
            help="Window around the precursor mass to remove peaks "
            "(default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--min_intensity",
            default=0.01,
            type=float,
            help="Remove peaks with a lower intensity relative to the base "
            "intensity (default: %(default)s).",
        )
        self._parser.add_argument(
            "--max_peaks_used",
            default=50,
            type=int,
            help="Only use the specified most intense peaks in the spectra "
            "(default: %(default)s).",
        )
        self._parser.add_argument(
            "--scaling",
            default="off",
            type=str,
            choices=["off", "root", "log", "rank"],
            help="Peak scaling method used to reduce the influence of very "
            "intense peaks (default: %(default)s).",
        )
        # Filled in 'parse', contains the specified settings.
        self._namespace = None

    def parse(self, args_str: str = None) -> None:
        """
        Parse the configuration settings.

        Parameters
        ----------
        args_str : str
            If None, the arguments are taken from sys.argv. Arguments that are
            not explicitly specified are taken from the configuration file.
        """
        self._namespace = vars(self._parser.parse_args(args_str))

        self._namespace["precursor_tol"][0] = float(
            self._namespace["precursor_tol"][0]
        )

    def __getattr__(self, option):
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized")
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)


config = Config()
