import collections
import gc
import logging
import math
import multiprocessing
import tempfile
from functools import partial
from typing import Iterator, List, Tuple

import fastcluster
import joblib
import lance
import numba as nb
import numpy as np
import scipy.cluster.hierarchy as sch
import spectrum_utils.utils as suu
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm

from . import similarity
from .. import utils

logger = logging.getLogger("falcon")

ConsensusTuple = collections.namedtuple(
    "ConsensusTuple",
    [
        "precursor_mz",
        "precursor_charge",
        "mz",
        "intensity",
        "retention_time",
        "cluster_id",
    ],
)


def generate_clusters(
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_tol: float,
    batch_size: int,
    consensus_method: str,
    consensus_params: dict,
) -> np.ndarray:
    """
    Hierarchical clustering of the given pairwise distance matrix.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    linkage: str
        The linkage method to use for hierarchical clustering.
    distance_threshold : float
        The linkage distance threshold at or above which clusters will not be merged.
    min_matches: int
        The minimum number of matched peaks to consider the spectra similar.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    fragment_tol: float
        The fragment m/z tolerance.
    batch_size : int
        Maximum interval size.
    consensus_method : str
        The method to use for consensus spectrum computation. Should be either
        'medoid' or 'average'.
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    np.ndarray
        Cluster labels. Noisy samples are given the label -1.
    """
    # Hierarchical clustering using the precomputed pairwise distance matrix.
    min_samples = 2
    logger.debug(
        "Hierarchical clustering (distance_threshold=%.4f, min_samples=%d)",
        distance_threshold,
        min_samples,
    )
    # Sort the metadata by increasing precursor m/z for easy subsetting.
    data = dataset.to_table(columns=["precursor_mz"]).to_pandas()
    data["row_id"] = data.index
    data.sort_values("precursor_mz", inplace=True)
    # Cluster per contiguous block of precursor m/z's (relative to the
    # precursor m/z threshold).
    logger.info(
        "Cluster %d spectra with charge %s",
        len(data),
        dataset.uri.split("_")[-1].split(".")[0],
    )
    with tempfile.NamedTemporaryFile(suffix=".npy") as cluster_file:
        cluster_filename = cluster_file.name
        cluster_labels = np.lib.format.open_memmap(
            cluster_filename, mode="w+", dtype=np.int32, shape=(data.shape[0],)
        )
        cluster_labels.fill(-1)
        max_label, rep_spectra = 0, []
        with tqdm(
            total=len(data), desc="Clustering", unit="spectra", smoothing=0
        ) as pbar:
            idx = data.index.values
            mz = data["precursor_mz"].values
            splits = _get_precursor_mz_splits(
                mz, precursor_tol_mass, precursor_tol_mode, batch_size
            )
            # Per m/z split clustering.
            chunks, single_spectra_tasks, two_spectra_tasks = (
                cost_based_chunking(splits, multiprocessing.cpu_count() - 1)
            )
            # Process single-spectrum m/z splits
            for rep_spectrum in joblib.Parallel(
                n_jobs=multiprocessing.cpu_count(), backend="threading"
            )(
                joblib.delayed(cluster_1_spectrum)(
                    dataset,
                    data["row_id"].values[spec_idx],
                )
                for _, spec_idx in single_spectra_tasks
            ):
                rep_spectra.append(rep_spectrum)
                pbar.update(1)
            # Process two-spectrum m/z splits
            for i, rep_spectra_chunk in enumerate(
                joblib.Parallel(
                    n_jobs=multiprocessing.cpu_count(), backend="threading"
                )(
                    joblib.delayed(cluster_2_spectra)(
                        dataset,
                        data.row_id[spec_idx],
                        data.row_id[spec_idx + 1],
                        fragment_tol,
                        distance_threshold,
                        min_matches,
                        consensus_method,
                        consensus_params,
                    )
                    for _, spec_idx in two_spectra_tasks
                )
            ):
                rep_spectra.extend(rep_spectra_chunk)
                # Clustered two spectra into one cluster.
                if len(rep_spectra_chunk) == 1:
                    spec_idx = two_spectra_tasks[i][1]
                    cluster_labels[spec_idx : spec_idx + 2] = 0
                pbar.update(2)
            # Process multi-spectrum m/z splits
            if len(chunks) > 0:
                # Process chunks
                process_chunk = partial(
                    cluster_chunk,
                    dataset=dataset,
                    linkage=linkage,
                    distance_threshold=distance_threshold,
                    min_matches=min_matches,
                    precursor_tol_mass=precursor_tol_mass,
                    precursor_tol_mode=precursor_tol_mode,
                    rt_tol=rt_tol,
                    fragment_tol=fragment_tol,
                    consensus_method=consensus_method,
                    consensus_params=consensus_params,
                )
                # TODO: move to chunking method
                data_chunks = []
                for chunk in chunks:
                    data_chunk = []
                    for task_id, (interval_start, interval_stop) in chunk:
                        row_ids = data.row_id[interval_start:interval_stop]
                        idx_interval = idx[interval_start:interval_stop]
                        mz_interval = mz[interval_start:interval_stop]
                        data_chunk.append(
                            (
                                task_id,
                                row_ids,
                                idx_interval,
                                mz_interval,
                            )
                        )
                    data_chunks.append(data_chunk)

                with multiprocessing.Pool(
                    multiprocessing.cpu_count() - 1
                ) as pool:
                    results = pool.map(process_chunk, data_chunks)
                flattened_results = [
                    split_result
                    for chunk_results in results
                    for split_result in chunk_results
                ]
                flattened_results.sort(key=lambda x: x[0])
                for task_id, (
                    interval_rep_spectra,
                    labels,
                ) in flattened_results:
                    if interval_rep_spectra is not None:
                        rep_spectra.extend(interval_rep_spectra)
                        cluster_labels[
                            splits[task_id] : splits[task_id + 1]
                        ] = labels
            max_label = _assign_global_cluster_labels(
                cluster_labels, splits, max_label
            )
        cluster_labels.flush()
        noise_mask = cluster_labels == -1
        n_clusters = np.unique(cluster_labels[~noise_mask]).size
        n_noise = noise_mask.sum()
        logger.info(
            "%d spectra grouped in %d clusters, %d spectra remain as singletons",
            (cluster_labels != -1).sum(),
            n_clusters,
            n_noise,
        )
        # Reassign noise points to singleton clusters.
        cluster_labels[noise_mask] = np.arange(
            n_clusters, n_clusters + n_noise
        )
        return cluster_labels, rep_spectra


@nb.njit
def _get_precursor_mz_splits(
    precursor_mzs: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
) -> nb.typed.List:
    """
    Find contiguous blocks of precursor m/z's, relative to the precursor m/z
    tolerance.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        The sorted precursor m/z's.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    batch_size : int
        Maximum interval size.

    Returns
    -------
    nb.typed.List[int]
        A list of start and end indices of blocks of precursor m/z's that do
        not exceed the precursor m/z tolerance and are separated by at least
        the precursor m/z tolerance.
    """
    splits, i = nb.typed.List([0]), 1
    for i in range(1, len(precursor_mzs)):
        if (
            suu.mass_diff(
                precursor_mzs[i],
                precursor_mzs[i - 1],
                precursor_tol_mode == "Da",
            )
            > precursor_tol_mass
        ):
            block_size = i - splits[-1]
            if block_size < batch_size:
                splits.append(i)
            else:
                n_chunks = math.ceil(block_size / batch_size)
                chunk_size = block_size // n_chunks
                for _ in range(block_size % n_chunks):
                    splits.append(splits[-1] + chunk_size + 1)
                for _ in range(n_chunks - (block_size % n_chunks)):
                    splits.append(splits[-1] + chunk_size)
    splits.append(len(precursor_mzs))
    return splits


def cost_based_chunking(tasks: List[int], num_chunks: int) -> Tuple[
    List[List[Tuple[int, Tuple[int, int]]]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """
    Groups tasks into chunks based on estimated computational costs.

    Parameters
    ----------
    tasks: List[int]
        List of tasks.
    num_chunks: int
        Number of chunks (e.g., number of workers).

    Returns
    -------
    Tuple[List[List[Tuple[int, Tuple[int, int]]], List[Tuple[int, int]], List[Tuple[int, int]]]
        A tuple containing the chunks, single-spectra tasks, and two-spectra tasks:
            Each chunk is a list of tuples containing the index of the task and the
            m/z split bounds.
            Each single-spectra task is a tuple containing the index of the task and the
            spectrum index.
            Each two-spectra task is a tuple containing the index of the task and the
            lower m/z split bound.
    """
    split_tuples = [(tasks[i], tasks[i + 1]) for i in range(len(tasks) - 1)]
    indexed_tasks = list(enumerate(split_tuples))
    indexed_tasks.sort(key=lambda x: (x[1][1] - x[1][0]) ** 2, reverse=True)

    # Initialize chunks and their cumulative costs
    chunks = [[] for _ in range(num_chunks)]
    costs = [0] * num_chunks
    single_spectra_tasks = []
    two_spectra_tasks = []

    # Assign tasks to the chunk with the least cumulative cost
    for index, task in indexed_tasks:
        if task[1] - task[0] == 1:
            single_spectra_tasks.append((index, task[0]))
        elif task[1] - task[0] == 2:
            two_spectra_tasks.append((index, task[0]))
        else:
            idx = costs.index(min(costs))  # Find the chunk with the least cost
            chunks[idx].append((index, task))
            costs[idx] += (task[1] - task[0]) ** 2

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]

    return chunks, single_spectra_tasks, two_spectra_tasks


def cluster_chunk(
    chunk: List[Tuple[int, Tuple[int, int]]],
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_tol: float,
    consensus_method: str,
    consensus_params: dict,
):
    """
    Cluster the vectors in the given interval.

    Parameters
    ----------
    chunk : List[Tuple[int, Tuple[int, int]]]
        The cluster tasks as a list of tuples containing the index of the task and the
        m/z split bounds.
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    linkage : str
        Linkage method to calculate the cluster distances.
    distance_threshold : float
        The maximum linkage distance threshold during clustering. Either
        'complete', 'average' or 'single'.
    min_matches: int
        The minimum number of matched peaks to consider the spectra similar.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    fragment_tol: float
        The fragment m/z tolerance.
    consensus_method : str
        The method to use for consensus spectrum computation.
    min_mz : float
        The minimum peak m/z value.
    max_mz : float
        The maximum peak m/z value.
    bin_size : float
        The width of each bin in m/z units.
    n_min : float
        The number of standard deviations for the lower bound for outlier rejection.
    n_max : float
        The number of standard deviations for the upper bound for outlier rejection.

    Returns
    -------
    List[Tuple[int, List[ConsensusTuple]]]
        The task index and representative spectra for each cluster.
    """

    return [
        (
            i,
            _cluster_mz_interval(
                dataset,
                row_ids,
                idx,
                mzs,
                linkage,
                distance_threshold,
                min_matches,
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                fragment_tol,
                consensus_method,
                consensus_params,
            ),
        )
        for i, row_ids, idx, mzs in chunk
    ]


def _cluster_mz_interval(
    dataset: lance.LanceDataset,
    row_ids: List[int],
    idx: np.ndarray,
    mzs: np.ndarray,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_mz_tol: float,
    consensus_method: str,
    consensus_params: dict,
) -> np.ndarray:
    """
    Cluster the vectors in the given interval.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    row_ids : List[int]
        The row ids of the spectra in the current interval.
    idx : np.ndarray
        The indexes of the spectra in the current interval.
    mzs : np.ndarray
        The precursor m/z's corresponding to the current interval indexes.
    linkage : str
        Linkage method to calculate the cluster distances. See
        `scipy.cluster.hierarchy.linkage` for possible options.
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    min_matches: int
        The minimum number of matched peaks to consider the spectra similar.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    fragment_mz_tol : float
        The fragment m/z tolerance.
    consensus_method : str
        The method to use for consensus spectrum computation. Should be either
        'medoid' or 'average'.
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    np.ndarray
        List of representative spectra for each cluster.
    """
    spectra = dataset.take(
        indices=row_ids,
        columns=[
            "precursor_mz",
            "precursor_charge",
            "retention_time",
            "mz",
            "intensity",
        ],
    ).to_pandas()
    rts = spectra["retention_time"].values
    spectra = spectra.apply(
        similarity.df_row_to_spectrum_tuple, axis=1
    ).tolist()
    n_spectra = len(spectra)
    cluster_labels = np.empty(n_spectra, np.int32)
    if n_spectra > 1:
        # Hierarchical clustering of the vectors.
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        pdist = compute_condensed_distance_matrix(
            spectra,
            fragment_mz_tol,
            min_matches,
        )
        labels = (
            sch.fcluster(
                fastcluster.linkage(pdist, linkage),
                distance_threshold,
                "distance",
            )
            - 1
        )
        # Refine initial clusters to make sure spectra within a cluster don't
        # have an excessive precursor m/z difference.
        order = np.argsort(labels)
        rev_order = np.argsort(order)
        idx, mzs, rts = (
            idx[order],
            mzs[order],
            rts[order],
        )
        labels, current_label = labels[order], 0
        for start_i, stop_i in _get_cluster_group_idx(labels):
            n_clusters = _postprocess_cluster(
                labels[start_i:stop_i],
                mzs[start_i:stop_i],
                rts[start_i:stop_i],
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                2,
                current_label,
            )
            current_label += n_clusters
        # Assign cluster labels.
        # Returned cluster labels should be sorted by precursor mass.
        cluster_labels = labels[rev_order]
        if current_label > 0:
            # Compute cluster medoids.
            order_ = np.argsort(labels)
            idx = idx[order_]
            labels = labels[order_]
            rts = rts[order_]
            order_map = order[order_]
            if consensus_method == "medoid":
                consensus_params["pdist"] = pdist
            rep_spectra = _get_representative_spectra(
                spectra,
                labels,
                rts,
                order_map,
                consensus_method,
                consensus_params,
            )
        else:
            rep_spectra = spectra
            rep_spectra = [
                ConsensusTuple(
                    *spec,
                    retention_time=rts[i],
                    cluster_id=-1,
                )
                for i, spec in enumerate(rep_spectra)
            ]
        # Force memory clearing.
        del pdist
        if n_spectra > 2**11:
            gc.collect()
    else:
        rep_spectra = [
            ConsensusTuple(
                *spectra[0],
                retention_time=rts[0],
                cluster_id=-1,
            )
        ]
    # Log clustering progress.
    # Only for large splits (> 1% of spectra)
    if n_spectra > 0.01 * len(spectra) and n_spectra > 1000:
        logger = utils.configure_logger()
        logger.debug(
            "Clustered %d spectra in %d clusters.", n_spectra, len(rep_spectra)
        )
    return rep_spectra, cluster_labels


def cluster_1_spectrum(
    dataset: lance.LanceDataset,
    row_id: int,
) -> ConsensusTuple:
    """
    "Cluster" a single spectrum.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    row_id : int
        The row id of the spectrum to be clustered.

    Returns
    -------
    ConsensusTuple
        The representative spectrum.
    """
    spectrum = dataset.take(
        indices=[row_id],
        columns=[
            "precursor_mz",
            "precursor_charge",
            "retention_time",
            "mz",
            "intensity",
        ],
    ).to_pandas()
    rt = spectrum["retention_time"].values[0]
    spectrum = spectrum.apply(
        similarity.df_row_to_spectrum_tuple, axis=1
    ).tolist()[0]
    return ConsensusTuple(
        *spectrum,
        retention_time=rt,
        cluster_id=-1,
    )


def cluster_2_spectra(
    dataset: lance.LanceDataset,
    row_id1: int,
    row_id2: int,
    fragment_mz_tol: float,
    distance_threshold: float,
    min_matches: int,
    consensus_method: str,
    consensus_params: dict,
):
    """
    Cluster two spectra.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    row_id1 : int
        The row id of the first spectrum to be clustered.
    row_id2 : int
        The row id of the second spectrum to be clustered.
    fragment_mz_tol : float
        The fragment m/z tolerance.
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    min_matches: int
        The minimum number of matched peaks to consider the spectra similar.
    consensus_method : str
        The method to use for consensus spectrum computation.
    min_mz : float
        The minimum m/z value to consider for binning.
    max_mz : float
        The maximum m/z value to consider for binning.
    bin_size : float
        The width of each bin in m/z units.
    n_min : float
        The number of standard deviations for the lower bound for outlier rejection.
    n_max : float
        The number of standard deviations for the upper bound for outlier rejection.

    Returns
    -------
    List[ConsensusTuple]
        The representative spectra for each cluster.
    """
    spectra = dataset.take(
        indices=[row_id1, row_id2],
        columns=[
            "precursor_mz",
            "precursor_charge",
            "retention_time",
            "mz",
            "intensity",
        ],
    ).to_pandas()
    spec1, spec2 = spectra.apply(
        similarity.df_row_to_spectrum_tuple, axis=1
    ).tolist()
    rt1, rt2 = spectra["retention_time"].values
    sim, n_match = similarity.cosine_fast(spec1, spec2, fragment_mz_tol)
    if (1 - sim) <= distance_threshold and n_match >= min_matches:
        # consensus spectrum
        if consensus_method == "medoid":
            consensus_params["pdist"] = [sim]
        return _get_representative_spectra(
            spectra=[spec1, spec2],
            labels=np.array([0, 0]),
            rts=np.array([rt1, rt2]),
            order_map=np.array([0, 1]),
            consensus_method=consensus_method,
            consensus_params=consensus_params,
        )

    else:
        return [
            ConsensusTuple(
                *spec1,
                retention_time=rt1,
                cluster_id=-1,
            ),
            ConsensusTuple(
                *spec2,
                retention_time=rt2,
                cluster_id=-1,
            ),
        ]


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.

    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).

    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while clusters[start_i] == -1 and start_i < clusters.shape[0]:
        yield start_i, start_i + 1
        start_i += 1
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


@nb.njit(boundscheck=False)
def _postprocess_cluster(
    cluster_labels: np.ndarray,
    cluster_mzs: np.ndarray,
    cluster_rts: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    min_samples: int,
    start_label: int,
) -> int:
    """
    Partitioning based on the precursor m/z's within each initial cluster to
    avoid that spectra within a cluster have an excessive precursor m/z
    difference.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    cluster_rts: np.ndarray
        Retention times of the samples in a single intial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol: float
        Maximum retention time tolerance for points to be clustered together.
    min_samples : int
        The minimum number of samples in a cluster.
    start_label : int
        The first cluster label.

    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    # No splitting needed if there are too few items in cluster.
    if cluster_labels.shape[0] < min_samples:
        cluster_labels.fill(-1)
        return 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        linkage = _linkage(cluster_mzs, precursor_tol_mode)
        with nb.objmode(cluster_assignments="int32[:]"):
            cluster_assignments = (
                sch.fcluster(linkage, precursor_tol_mass, "distance") - 1
            )
        # Optionally restrict clusters by their retention time as well.
        if rt_tol is not None:
            with nb.objmode(cluster_assignments="int32[:]"):
                cluster_assignments_rt = (
                    fcluster(_linkage(cluster_rts), rt_tol, "distance") - 1
                )
                # Merge cluster assignments based on precursor m/z and RT.
                # First prime factorization is used to get unique combined cluster
                # labels, after which consecutive labels are obtained.
                cluster_assignments = np.unique(
                    cluster_assignments * 2 + cluster_assignments_rt * 3,
                    return_inverse=True,
                )[1]

        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels.fill(start_label)
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            cluster_labels.fill(-1)
            n_clusters = 0
        else:
            labels = nb.typed.Dict.empty(
                key_type=nb.int64, value_type=nb.int64
            )
            # Count cluster sizes
            for i, label in enumerate(cluster_assignments):
                labels[label] = labels.get(label, 0) + 1
            n_clusters = 0
            # Assign unique cluster labels
            for label, count in labels.items():
                if count < min_samples:
                    labels[label] = -1
                else:
                    labels[label] = start_label + n_clusters
                    n_clusters += 1
            for i, label in enumerate(cluster_assignments):
                cluster_labels[i] = labels[label]
        return n_clusters


@nb.njit(cache=True, fastmath=True)
def _linkage(values: np.ndarray, tol_mode: str = None) -> np.ndarray:
    """
    Perform hierarchical clustering of a one-dimensional m/z or RT array.

    Because the data is one-dimensional, no pairwise distance matrix needs to
    be computed, but rather sorting can be used.

    For information on the linkage output format, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    Parameters
    ----------
    values : np.ndarray
        The precursor m/z's or RTs for which pairwise distances are computed.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z, 'rt' for
        retention time).

    Returns
    -------
    np.ndarray
        The hierarchical clustering encoded as a linkage matrix.
    """
    linkage = np.zeros((values.shape[0] - 1, 4), np.double)
    # min, max, cluster index, number of cluster elements
    # noinspection PyUnresolvedReferences
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(values.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]  # Always positive.
            if tol_mode == "ppm":
                dist = dist / clusters[i][0] * 10**6
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [
            clusters[min_i][2],
            clusters[min_i + 1][2],
            min_dist,
            n_points,
        ]
        clusters[min_i] = (
            clusters[min_i][0],
            clusters[min_i + 1][1],
            values.shape[0] + it,
            n_points,
        )
        del clusters[min_i + 1]

    return linkage


def _get_representative_spectra(
    spectra: List[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    consensus_method: str,
    consensus_params: dict,
) -> List[ConsensusTuple]:
    """
    Get the representative spectra for each cluster.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra.
    labels : np.ndarray
        Cluster labels.
    rts : np.ndarray
        The retention times corresponding to the current interval indexes.
        order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.
    consensus_method : str
        The method to use for consensus spectrum computation.
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    List[ConsensusTuple]
        The representative spectra for each cluster.
    """
    if consensus_method == "medoid":
        return _get_cluster_medoids(
            spectra, labels, rts, order_map, **consensus_params
        )
    elif consensus_method == "average":
        return _get_cluster_average(
            spectra,
            labels,
            rts,
            order_map,
            **consensus_params,
        )
    else:
        raise ValueError(
            f"Unknown consensus spectrum method: {consensus_method}"
        )


@nb.njit(fastmath=True, boundscheck=False)
def _get_cluster_medoids(
    spectra: List[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
) -> List[ConsensusTuple]:
    """
    Get the indexes of the cluster medoids.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra.
    labels : np.ndarray
        Cluster labels.
    rts: np.ndarray
        The retention times corresponding to the current interval indexes.
    pdist : np.ndarray
        Condensed pairwise distance matrix.
    order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.

    Returns
    -------
    List[ConsensusTuple]
        The medoids for each cluster.
    """
    medoids, m = [], len(spectra)
    for start_i, stop_i in _get_cluster_group_idx(labels):
        if stop_i - start_i > 2:
            row_sum = np.zeros(stop_i - start_i, np.float32)
            for row in range(stop_i - start_i):
                for col in range(row + 1, stop_i - start_i):
                    i, j = order_map[start_i + row], order_map[start_i + col]
                    if i > j:
                        i, j = j, i
                    pdist_ij = pdist[m * i + j - ((i + 2) * (i + 1)) // 2]
                    row_sum[row] += pdist_ij
                    row_sum[col] += pdist_ij
            medoid_spec = spectra[order_map[start_i + np.argmin(row_sum)]]
            medoids.append(
                ConsensusTuple(
                    precursor_mz=medoid_spec.precursor_mz,
                    precursor_charge=medoid_spec.precursor_charge,
                    mz=medoid_spec.mz,
                    intensity=medoid_spec.intensity,
                    retention_time=rts[start_i + np.argmin(row_sum)],
                    cluster_id=labels[start_i + np.argmin(row_sum)],
                )
            )
        else:
            medoid_spec = spectra[start_i]
            medoids.append(
                ConsensusTuple(
                    precursor_mz=medoid_spec.precursor_mz,
                    precursor_charge=medoid_spec.precursor_charge,
                    mz=medoid_spec.mz,
                    intensity=medoid_spec.intensity,
                    retention_time=rts[start_i],
                    cluster_id=labels[start_i],
                )
            )
    return medoids


def _get_cluster_average(
    spectra: List[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    min_mz: float,
    max_mz: float,
    bin_size: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> List[ConsensusTuple]:
    """
    Get the average spectra for each cluster. The average spectrum is computed
    by binning the spectra, removing (intensity) outliers in each bin, and averaging the remaining peaks.
    Adapted from Carr, A. V. et al. Proteomics (2024).
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.202300234.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra.
    labels : np.ndarray
        Cluster labels.
    rts : np.ndarray
        The retention times corresponding to the current interval indexes.
    order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.
    min_mz : float
        The minimum m/z value to consider for binning.
    max_mz : float
        The maximum m/z value to consider for binning.
    bin_size : float
        The width of each bin in m/z units.
    outlier_cutoff_lower : float
        The number of standard deviations for the lower bound for outlier rejection.
    outlier_cutoff_upper : float
        The number of standard deviations for the upper bound for outlier rejection.

    Returns
    -------
    List[ConsensusTuple]
        The average spectra for each cluster.
    """
    average_spectra = []
    for start_i, stop_i in _get_cluster_group_idx(labels):
        if stop_i - start_i > 1:
            spectra_to_average = [
                spectra[order_map[i]] for i in range(start_i, stop_i)
            ]
            # average precursor mz
            avg_mz = np.mean(
                [spec.precursor_mz for spec in spectra_to_average], axis=0
            )
            charge = spectra_to_average[0].precursor_charge
            avg_rt = np.mean(rts[start_i:stop_i])

            # Bin the spectra
            bins_idx, bins_peaks, bins_mz = _spectrum_binning(
                spectra_to_average, min_mz, max_mz, bin_size
            )
            del spectra_to_average
            # Outlier rejection
            bins_idx, bins_peaks, bins_mz = _outlier_rejection(
                bins_idx,
                bins_peaks,
                bins_mz,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            # Construct average spectrum
            avg_spectrum = _construct_average_spectrum(
                bins_idx,
                bins_peaks,
                bins_mz,
                avg_mz,
                charge,
                avg_rt,
                labels[start_i],
            )
            average_spectra.append(avg_spectrum)
        else:
            # Single spectrum cluster
            avg_spectrum = spectra[order_map[start_i]]
            average_spectra.append(
                ConsensusTuple(
                    *avg_spectrum,
                    retention_time=rts[start_i],
                    cluster_id=labels[start_i],
                )
            )
    return average_spectra


@nb.njit(cache=True)
def _spectrum_binning(
    spectra: List[similarity.SpectrumTuple],
    min_mz: float,
    max_mz: float,
    bin_size: float,
) -> Tuple[List[int], nb.typed.List]:
    """
    Jointly bin multiple spectra into fixed-size bins based on m/z values.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        A list of spectra to be binned.
    min_mz : float
        The minimum m/z value to consider for binning.
    max_mz : float
        The maximum m/z value to consider for binning.
    bin_size : float
        The width of each bin in m/z units.

    Returns
    -------
    Tuple[np.ndarray, nb.typed.List]
        A tuple containing:
        - An array of integers representing the indices of the non-empty bins.
        - A Numba typed list of arrays containing the intensities for each bin.
    """
    start_dim = min_mz - (min_mz % bin_size)
    end_dim = max_mz + bin_size - (max_mz % bin_size)
    n_bins = math.ceil((end_dim - start_dim) / bin_size)

    n_spectra = len(spectra)

    bins_indices = -np.ones(n_bins, dtype=np.int32)
    bins_peaks = nb.typed.List()
    bins_mz = nb.typed.List()
    for _ in range(n_bins):
        nested_list_p = nb.typed.List.empty_list(nb.types.float32)
        nested_list_m = nb.typed.List.empty_list(nb.types.float32)
        bins_peaks.append(nested_list_p)
        bins_mz.append(nested_list_m)
    bins_spectra_count = np.zeros(n_bins, dtype=np.int32)

    for spec in spectra:
        bins_peak_presence = np.zeros(n_bins, dtype=np.int32)
        for mz, intensity in zip(spec.mz, spec.intensity):
            bin_idx = math.floor((mz - min_mz) / bin_size)
            if 0 <= bin_idx < n_bins:
                if bins_indices[bin_idx] == -1:
                    bins_indices[bin_idx] = bin_idx
                bins_peaks[bin_idx].append(intensity)
                bins_mz[bin_idx].append(mz)
                bins_peak_presence[bin_idx] = 1
        bins_spectra_count += bins_peak_presence
    # Mark peaks that appear in less than 70% of the spectra as empty for removal
    for i in range(n_bins):
        if bins_indices[i] != -1:
            if bins_spectra_count[i] < 0.7 * n_spectra:
                bins_indices[i] = -1
    # Remove empty bins
    mask = bins_indices != -1
    bins_indices = bins_indices[mask]
    bins_peaks_nb = nb.typed.List()
    bins_mz_nb = nb.typed.List()
    for i in bins_indices:
        bins_peaks_nb.append(typed_list_to_numpy(bins_peaks[i]))
        bins_mz_nb.append(typed_list_to_numpy(bins_mz[i]))

    return bins_indices, bins_peaks_nb, bins_mz_nb


@nb.njit(cache=True)
def _outlier_rejection(
    bins_indices: List[int],
    bins_peaks: nb.typed.List,
    bins_mz: nb.typed.List,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> Tuple[nb.typed.List]:
    """
    Remove outliers from binned spectra using the sigma clipping algorithm and
    return the averaged intensities.

    Parameters
    ----------
    bins_indices : List[int]
        The indices of the non-empty bins.
    bins_peaks : nb.typed.List
        The intensities for each bin.
    bins_mz : nb.typed.List
        The m/z values for each bin.
    outlier_cutoff_lower : float
        The number of standard deviations for the lower bound.
    outlier_cutoff_upper : float
        The number of standard deviations for the upper bound.

    Returns
    -------
    nb.typed.List
        The cleaned and averaged intensities for each bin.
    """
    n_peaks = len(bins_indices)
    cleaned_bins_peaks = np.zeros(n_peaks, dtype=np.float32)
    cleaned_bins_mz = np.zeros(n_peaks, dtype=np.float32)

    zero_peaks = np.zeros(len(bins_indices), dtype=np.bool_)
    for i in range(len(bins_indices)):
        intensities = bins_peaks[i]
        mzs = bins_mz[i]
        if len(intensities) > 2:
            clipped_p, clipped_m = _sigma_clipping(
                intensities,
                mzs,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            if len(clipped_p) < 1:
                zero_peaks[i] = True
            else:
                cleaned_bins_peaks[i] = np.mean(clipped_p)
                cleaned_bins_mz[i] = np.mean(clipped_m)
        else:
            cleaned_bins_peaks[i] = np.mean(intensities)
            cleaned_bins_mz[i] = np.mean(mzs)

    # Return non-empty bins
    return (
        bins_indices[~zero_peaks],
        cleaned_bins_peaks[~zero_peaks],
        cleaned_bins_mz[~zero_peaks],
    )


@nb.njit(cache=True)
def _sigma_clipping(
    intensities: np.ndarray,
    mzs: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> np.ndarray:
    """
    Apply sigma clipping to remove outliers from the array.

    Parameters
    ----------
    intensities : np.ndarray
        The array of intensities.
    mzs : np.ndarray
        The array of m/z values.
    outlier_cutoff_lower : float
        The number of standard deviations for the lower bound.
    outlier_cutoff_upper : float
        The number of standard deviations for the upper bound.

    Returns
    -------
    np.ndarray
        The array of intensities with outliers removed.
    """
    while len(intensities) > 2:
        med = np.median(intensities)
        std = np.std(intensities)
        if std == 0.0:
            break
        # Mask outliers
        mask = _sigma_clip(
            intensities, med, std, outlier_cutoff_lower, outlier_cutoff_upper
        )
        # Break if no outliers were found
        if np.sum(mask) == len(intensities):
            break
        intensities = intensities[mask]
        mzs = mzs[mask]
    return intensities, mzs


@nb.njit(cache=True)
def _sigma_clip(
    values: np.ndarray,
    median: float,
    std: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> np.ndarray:
    """
    Label outlier intensities.

    Parameters
    ----------
    values : np.ndarray
        The values to check.
    median : float
        The median of the values.
    std : float
        The standard deviation of the values.
    outlier_cutoff_lower : float
        The number of standard deviations below the median.
    outlier_cutoff_upper : float
        The number of standard deviations above the median.

    Returns
    -------
    np.ndarray
        mask:
            True if the value is not an outlier (keep)
            False otherwise (remove)
    """
    lower_bound = median - outlier_cutoff_lower * std
    upper_bound = median + outlier_cutoff_upper * std
    return (values >= lower_bound) & (values <= upper_bound)


@nb.njit(cache=True)
def _construct_average_spectrum(
    bins_indices: List[int],
    bins_peaks: nb.typed.List,
    bins_mz: nb.typed.List,
    avg_precursor_mz: float,
    charge: int,
    avg_rt: float,
    cluster: int,
) -> similarity.SpectrumTuple:
    """
    Construct the average spectrum from the binned spectra.

    Parameters
    ----------
    bins_indices : List[int]
        The indices of the non-empty bins.
    bins_peaks : nb.typed.List
        The intensities for each bin.
    bins_mz : nb.typed.List
        The m/z values for each bin
    avg_precursor_mz : float
        The average precursor m/z.
    charge : int
        The precursor charge.
    avg_rt : float
        The average retention time.
    cluster : int
        The cluster label.

    Returns
    -------
    similarity.SpectrumTuple
        The average spectrum.
    """
    mz = np.empty(len(bins_indices), np.float32)
    intensity = np.empty(len(bins_indices), np.float32)

    idx = 0
    for avg_intensity, avg_mz in zip(bins_peaks, bins_mz):
        # use the middle of the bin as the m/z value
        mz[idx] = avg_mz
        intensity[idx] = avg_intensity
        idx += 1

    return ConsensusTuple(
        precursor_mz=avg_precursor_mz,
        precursor_charge=charge,
        mz=mz,
        intensity=intensity,
        retention_time=avg_rt,
        cluster_id=cluster,
    )


@nb.njit(cache=True)
def typed_list_to_numpy(lst: nb.typed.List) -> np.ndarray:
    """
    Convert a Numba typed list of floats to a Numpy array.

    Parameters
    ----------
    lst : nb.typed.List
        The list of floats to convert.

    Returns
    -------
    np.ndarray
        The converted array.
    """
    n = len(lst)
    arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        arr[i] = lst[i]
    return arr


@nb.njit(boundscheck=False)
def _assign_global_cluster_labels(
    cluster_labels: np.ndarray,
    splits: nb.typed.List,
    current_label: int,
) -> int:
    """
    Convert cluster labels per split to globally unique labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        The cluster labels.
    splits : nb.typed.List
        A list of start and end indices of cluster chunks.
    current_label : int
        First cluster label.

    Returns
    -------
    int
        Last cluster label.
    """
    max_label = current_label
    update_current_label = False
    for i in range(len(splits) - 1):
        for j in range(splits[i], splits[i + 1]):
            if cluster_labels[j] != -1:
                cluster_labels[j] += current_label
                update_current_label = True
                if cluster_labels[j] > max_label:
                    max_label = cluster_labels[j]
        # only update after non-noise clusters
        if update_current_label:
            current_label = max_label + 1
            update_current_label = False
    return max_label


def compute_condensed_distance_matrix(
    spec_tuples: List[similarity.SpectrumTuple],
    fragment_mz_tol: float,
    min_matches: int,
) -> np.ndarray:
    """
    Compute the condensed pairwise distance matrix for the given spectra.

    Parameters
    ----------
    spec_tuples : List[similarity.SpectrumTuple]
        The spectra to compute the pairwise distance matrix for.
    fragment_mz_tolerance : float
        The fragment m/z tolerance.
    min_matches : int
        The minimum number of matched peaks to consider the spectra similar.

    Returns
    -------
    np.ndarray
        The condensed pairwise distance matrix.
    """
    n = len(spec_tuples)
    with tempfile.NamedTemporaryFile(suffix=".npy") as pdist_file:
        pdist_filename = pdist_file.name
        condensed_dist_matrix = np.lib.format.open_memmap(
            pdist_filename,
            mode="w+",
            dtype=np.float32,
            shape=(n * (n - 1) // 2,),
        )

        def cosine_worker(i, j):
            spec_tup1 = spec_tuples[i]
            spec_tup2 = spec_tuples[j]
            sim, n_match = similarity.cosine_fast(
                spec_tup1, spec_tup2, fragment_mz_tol
            )
            if n_match < min_matches:
                sim = 0.0
            distance = 1.0 - sim
            idx = condensed_index(i, j, n)
            condensed_dist_matrix[idx] = distance

        for i in range(n - 1):
            for j in range(i + 1, n):
                cosine_worker(i, j)

        return condensed_dist_matrix


@nb.njit
def condensed_index(i: int, j: int, n: int) -> int:
    """
    Get the index of the condensed distance matrix.

    Parameters
    ----------
    i : int
        The row index.
    j : int
        The column index.
    n : int
        The number of spectra.

    Returns
    -------
    int
        The index of the condensed distance matrix.
    """
    if i == j:
        raise ValueError("No diagonal elements in condensed matrix")
    if i > j:
        i, j = j, i
    return int(n * i + j - ((i + 2) * (i + 1)) // 2)
