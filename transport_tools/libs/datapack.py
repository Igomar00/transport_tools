# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021 The TransportTools Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Export of supercluster results into a compact, pickle-free datapack for the TTviz Lite PyMOL plugin.

The datapack is a single uncompressed zip archive holding .npy arrays and a JSON manifest. Members are aligned so
that readers can memory-map every array directly from the archive. It keeps everything needed to recompute
supercluster statistics for each result set (initial, filtered01, filtered02) and comparative label, to merge
superclusters exactly as TransportTools would, and to visualize them: per-snapshot tunnels of the initial
supercluster profiles (one tunnel per snapshot), layered pathsets of superclusters and of their tunnel clusters,
supercluster definitions, active filters, and transport event counts (optionally also event records and geometry).

NOTE: this module is vendored byte-identically into TTviz Lite (ttviz_lite/vendor/tt_datapack.py), therefore it
imports from transport_tools only what TransportTools 0.9.8 already provides.
"""

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import os
import json
import glob
import shutil
import pickle
import struct
import zipfile
import tempfile
import argparse
import numpy as np
from datetime import datetime, timezone
from logging import getLogger
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = getLogger(__name__)

DATAPACK_FORMAT = "ttviz-datapack"
DATAPACK_VERSION = 1
DEFAULT_DATAPACK_FILENAME = "ttviz.datapack"
EVENT_TYPES = ("entry", "release")

XYZ_SCALE = 0.01  # Angstrom per int16 step of quantized sphere coordinates
RADIUS_SCALE = 0.001  # Angstrom per uint16 step of quantized sphere radii

PATHSET_KIND_SUPERCLUSTER = 0
PATHSET_KIND_CLUSTER = 1
PATHSET_KIND_EVENT = 2

_ALIGNMENT = 64  # member data start alignment, matches numpy's own .npy header padding
_PARALLEL_MIN_BYTES = 256 * 1024 ** 2  # convert dumps in parallel only above this total size
_RECYCLE_MIN_BYTES = 512 * 1024 ** 2  # recycle worker processes after each profile above this size
_ZIP_ALIGN_EXTRA_ID = 0xD935  # extra field id used for alignment padding (same as Android zipalign)
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)  # fixed member timestamps for reproducible datapacks

_PARAMETERS2EXPORT = ["snapshots_per_simulation", "layer_thickness", "sp_radius", "random_seed",
                      "perform_comparative_analysis", "process_bottleneck_residues", "relevant_tunnel_min_length",
                      "relevant_tunnel_min_radius", "relevant_tunnel_max_curvature", "snapshot_id_position",
                      "snapshot_delimiter", "caver_traj_offset", "clustering_cutoff", "clustering_method",
                      "clustering_linkage", "tunnel_properties_quantile", "max_events_per_cluster4visualization",
                      "output_path"]

ProgressCallback = Callable[[float, str], None]


class _ResultsPaths:
    def __init__(self, results_dir: str, parameters: dict):
        """
        Resolves locations inside a results folder; paths stored in job parameters are relative to the working folder
        of the original run, so they are re-based on the actual location of the results folder
        :param results_dir: path to the results folder (output_path of the run)
        :param parameters: job configuration parameters stored in a checkpoint
        """

        self.results_dir = os.path.abspath(results_dir)
        self.parameters = parameters

    def get(self, parameter_name: str) -> str:
        """
        Location of a folder or file stored in job parameters, re-based to the results folder
        :param parameter_name: name of the parameter holding the path
        :return: path inside the results folder
        """

        value = self.parameters[parameter_name]
        output_path = self.parameters["output_path"]
        rel_path = os.path.relpath(value, output_path)
        if rel_path.startswith(os.pardir):  # not inside the original output folder, keep as is
            return value
        return os.path.normpath(os.path.join(self.results_dir, rel_path))


def find_checkpoints(results_dir: str) -> Dict[int, str]:
    """
    Enumerate checkpoint files available in a results folder
    :param results_dir: path to the results folder
    :return: mapping of stage number to checkpoint file
    """

    checkpoints = dict()
    for filename in glob.glob(os.path.join(results_dir, "_internal", "checkpoints", "stage[0-9][0-9][0-9].dump")):
        stage = int(os.path.basename(filename)[5:8])
        checkpoints[stage] = filename

    return checkpoints


def _load_checkpoint(filename: str):
    """
    Unpickle TransportProcesses object from a checkpoint file
    :param filename: checkpoint file
    :return: TransportProcesses object
    """

    from transport_tools.libs import tools  # noqa: F401 - registers classes needed for unpickling
    with open(filename, "rb") as in_stream:
        return pickle.load(in_stream)


def _jsonable(value: Any) -> Any:
    """
    Converts value to JSON-serializable form
    :param value: value to convert
    :return: converted value
    """

    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(val) for val in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _encode_pathset(path_set) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts LayeredPathSet into arrays
    :param path_set: LayeredPathSet to convert
    :return: nodes data (x, y, z, layer, terminal, radius), concatenated node indices of paths, and lengths of paths
    """

    if path_set.nodes_data is None or not path_set.node_paths:
        return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)

    nodes = np.asarray(path_set.nodes_data, dtype=np.float64)[:, [0, 1, 2, 3, 4, 5]].astype(np.float32)
    label2index = {label: i for i, label in enumerate(path_set.node_labels)}
    paths = list()
    lengths = list()
    for path in path_set.node_paths:
        paths.extend(label2index[str(label)] for label in path)
        lengths.append(len(path))

    return nodes, np.asarray(paths, dtype=np.int32), np.asarray(lengths, dtype=np.int64)


class _PathsetChunk:
    def __init__(self):
        """
        Accumulator of encoded pathsets produced by a worker
        """

        self.keys: List[Tuple[int, int, int]] = list()  # (kind, a, b)
        self.nodes: List[np.ndarray] = list()
        self.paths: List[np.ndarray] = list()
        self.path_lengths: List[np.ndarray] = list()

    def add(self, kind: int, key_a: int, key_b: int, path_set):
        nodes, paths, lengths = _encode_pathset(path_set)
        self.keys.append((kind, key_a, key_b))
        self.nodes.append(nodes)
        self.paths.append(paths)
        self.path_lengths.append(lengths)

    def save(self, prefix: str) -> Dict[str, Any]:
        """
        Saves accumulated pathsets to .npy files
        :param prefix: path prefix of the files
        :return: description of saved data
        """

        num_nodes = np.array([nodes.shape[0] for nodes in self.nodes], dtype=np.int64)
        num_paths = np.array([lengths.shape[0] for lengths in self.path_lengths], dtype=np.int64)
        arrays = {
            "keys": np.asarray(self.keys, dtype=np.int32).reshape(-1, 3),
            "num_nodes": num_nodes,
            "num_paths": num_paths,
            "nodes": np.concatenate(self.nodes) if self.nodes else np.zeros((0, 6), dtype=np.float32),
            "paths": np.concatenate(self.paths) if self.paths else np.zeros(0, dtype=np.int32),
            "path_lengths": (np.concatenate(self.path_lengths) if self.path_lengths
                             else np.zeros(0, dtype=np.int64)),
        }
        for name, array in arrays.items():
            np.save("{}_{}.npy".format(prefix, name), array)

        return {"prefix": prefix, "num_pathsets": len(self.keys), "num_nodes": int(num_nodes.sum()),
                "num_paths": int(num_paths.sum()), "num_path_nodes": int(arrays["paths"].shape[0])}


def _process_super_cluster_worker(task: Tuple[int, str, str, Dict[str, int], List[str], str]) -> Dict[str, Any]:
    """
    Converts initial profile and pathsets of a single supercluster into arrays saved in the temporary folder
    :param task: (sc_id, profile dump file, pathset dump file, md_label -> index, label table, output prefix)
    :return: description of saved data
    """

    from transport_tools.libs import networks  # noqa: F401 - registers classes needed for unpickling

    sc_id, profile_file, path_set_file, md_indices, labels, prefix = task

    with open(profile_file, "rb") as in_stream:
        profile = pickle.load(in_stream)

    ids = list()
    scalars = list()
    bottleneck_xyz = list()
    num_spheres = list()
    spheres = list()
    num_residues = list()
    residues = list()
    for md_label, tunnel_profile4md in profile.tunnel_profiles4md.items():
        md_index = md_indices[md_label]
        for snap_id, tunnel in tunnel_profile4md.records.items():
            ids.append((md_index, int(snap_id), int(tunnel.caver_cluster_id), int(tunnel.tunnel_id)))
            scalars.append((tunnel.throughput, tunnel.cost, tunnel.bottleneck_radius, tunnel.length,
                            tunnel.curvature))
            if tunnel.bottleneck_xyz is None:
                bottleneck_xyz.append((np.nan, np.nan, np.nan))
            else:
                bottleneck_xyz.append(tuple(tunnel.bottleneck_xyz[:3]))
            num_spheres.append(tunnel.spheres_data.shape[0])
            spheres.append(np.asarray(tunnel.spheres_data[:, [0, 1, 2, 4]], dtype=np.float32))
            num_residues.append(len(tunnel.bottleneck_residues))
            residues.extend(tunnel.bottleneck_residues)
    del profile

    residue_table, residue_indices = np.unique(np.asarray(residues, dtype=str), return_inverse=True) \
        if residues else (np.zeros(0, dtype=str), np.zeros(0, dtype=np.int64))
    arrays = {
        "ids": np.asarray(ids, dtype=np.int32).reshape(-1, 4),
        "scalars": np.asarray(scalars, dtype=np.float64).reshape(-1, 5),
        "bottleneck_xyz": np.asarray(bottleneck_xyz, dtype=np.float32).reshape(-1, 3),
        "num_spheres": np.asarray(num_spheres, dtype=np.int64),
        "spheres": np.concatenate(spheres) if spheres else np.zeros((0, 4), dtype=np.float32),
        "num_residues": np.asarray(num_residues, dtype=np.int64),
        "residues": np.asarray(residue_indices, dtype=np.int32),
    }
    for name, array in arrays.items():
        np.save("{}_{}.npy".format(prefix, name), array)

    sphere_data = arrays["spheres"]
    description = {
        "sc_id": sc_id,
        "prefix": prefix,
        "num_tunnels": arrays["ids"].shape[0],
        "num_spheres": sphere_data.shape[0],
        "num_residues": arrays["residues"].shape[0],
        "residue_table": residue_table.tolist(),
        "max_abs_xyz": float(np.abs(sphere_data[:, :3]).max()) if sphere_data.shape[0] else 0.,
        "max_radius": float(sphere_data[:, 3].max()) if sphere_data.shape[0] else 0.,
        "pathset_labels": list(),
    }

    # stored supercluster pathsets (computed once at stage 5)
    chunk = _PathsetChunk()
    if os.path.exists(path_set_file):
        with open(path_set_file, "rb") as in_stream:
            path_sets = pickle.load(in_stream)
        for label, path_set in path_sets.items():
            if label not in labels:
                continue
            chunk.add(PATHSET_KIND_SUPERCLUSTER, sc_id, labels.index(label), path_set)
            description["pathset_labels"].append(label)
    description["pathsets"] = chunk.save(prefix + "_ps")

    return description


def _process_md_worker(task: Tuple[int, str, List[int], str, List[Tuple[int, str]], str]) -> Dict[str, Any]:
    """
    Converts layered pathsets of tunnel clusters (and optionally of transport events) from a single MD simulation
    :param task: (md index, layered caver dump file, cluster IDs to export, layered aquaduct dump file,
                  list of (event record index, entity key) to export, output prefix)
    :return: description of saved data
    """

    from transport_tools.libs import networks  # noqa: F401 - registers classes needed for unpickling

    md_index, caver_file, cluster_ids, aquaduct_file, event_keys, prefix = task
    chunk = _PathsetChunk()
    missing_events = list()

    if cluster_ids:
        with open(caver_file, "rb") as in_stream:
            layered_entities = pickle.load(in_stream)
        for cls_id in cluster_ids:
            if cls_id in layered_entities:
                chunk.add(PATHSET_KIND_CLUSTER, md_index, cls_id, layered_entities[cls_id])
        del layered_entities

    if event_keys:
        if os.path.exists(aquaduct_file):
            with open(aquaduct_file, "rb") as in_stream:
                layered_entities = pickle.load(in_stream)
        else:
            layered_entities = dict()
        for event_index, entity_key in event_keys:
            if entity_key in layered_entities:
                chunk.add(PATHSET_KIND_EVENT, event_index, 0, layered_entities[entity_key])
            else:
                missing_events.append(entity_key)
        del layered_entities

    description = chunk.save(prefix)
    description["missing_events"] = missing_events
    return description


class _DatapackZipWriter:
    def __init__(self, filename: str):
        """
        Writer of uncompressed zip archive with members aligned for direct memory-mapping
        :param filename: output file
        """

        self._zip = zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_STORED, allowZip64=True)

    def _open_member(self, name: str):
        zinfo = zipfile.ZipInfo(name, date_time=_ZIP_DATE_TIME)
        zinfo.compress_type = zipfile.ZIP_STORED
        # local header = 30 bytes + name + our padding extra field + zip64 extra field (20 bytes, forced below)
        header_offset = self._zip.start_dir
        unpadded_end = header_offset + 30 + len(name.encode("utf-8")) + 4 + 20
        padding = (-unpadded_end) % _ALIGNMENT
        zinfo.extra = struct.pack("<HH", _ZIP_ALIGN_EXTRA_ID, padding) + b"\0" * padding
        return self._zip.open(zinfo, "w", force_zip64=True)

    def write_bytes(self, name: str, data: bytes):
        with self._open_member(name) as out_stream:
            out_stream.write(data)

    def write_array(self, name: str, array: np.ndarray):
        array = np.ascontiguousarray(array)
        with self._open_member(name) as out_stream:
            np.lib.format.write_array_header_1_0(out_stream, np.lib.format.header_data_from_array_1_0(array))
            out_stream.write(array.tobytes())

    def write_chunked_array(self, name: str, dtype: np.dtype, shape: Tuple[int, ...],
                            chunks: List[np.ndarray]):
        """
        Writes array assembled from consecutive chunks without materializing it in memory
        :param name: member name
        :param dtype: data type of the array
        :param shape: shape of the complete array
        :param chunks: iterable of chunks (concatenated along the first axis)
        """

        dtype = np.dtype(dtype)
        header = {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": tuple(shape)}
        written_rows = 0
        with self._open_member(name) as out_stream:
            np.lib.format.write_array_header_1_0(out_stream, header)
            for chunk in chunks:
                chunk = np.ascontiguousarray(chunk, dtype=dtype)
                written_rows += chunk.shape[0] if chunk.ndim else 0
                out_stream.write(chunk.tobytes())

        if shape and written_rows != shape[0]:
            raise RuntimeError("Datapack member '{}' expected {} rows, got {}".format(name, shape[0], written_rows))

    def close(self):
        self._zip.close()


class _ProgressReporter:
    def __init__(self, callback: Optional[ProgressCallback], weights: Dict[str, float]):
        """
        Maps progress of individual steps onto overall progress
        :param callback: function receiving (fraction done, message)
        :param weights: relative weights of steps
        """

        self.callback = callback
        total = sum(weights.values())
        self.starts = dict()
        self.weights = dict()
        position = 0.
        for step, weight in weights.items():
            self.starts[step] = position / total
            self.weights[step] = weight / total
            position += weight

    def __call__(self, step: str, done: float, total: float, message: str):
        fraction = self.starts[step] + self.weights[step] * (done / total if total else 1.)
        if self.callback is not None:
            self.callback(min(fraction, 1.), message)
        logger.debug("datapack {:5.1f}%: {}".format(100 * fraction, message))


def _run_tasks(worker: Callable, tasks: List[Any], num_processes: int, report: Callable[[int, int], None],
               max_tasks_per_child: Optional[int] = None) -> List[Any]:
    """
    Executes tasks either serially or in a pool of spawned processes, preserving order of results
    :param worker: worker function
    :param tasks: list of task arguments
    :param num_processes: number of processes to use
    :param report: function called with (finished tasks, total tasks)
    :param max_tasks_per_child: recycle worker processes after this many tasks (frees memory of large dumps)
    :return: list of results in order of tasks
    """

    results: List[Any] = [None] * len(tasks)
    report(0, len(tasks))
    if num_processes <= 1 or len(tasks) <= 1:
        for i, task in enumerate(tasks):
            results[i] = worker(task)
            report(i + 1, len(tasks))
        return results

    with get_context("spawn").Pool(processes=min(num_processes, len(tasks)),
                                   maxtasksperchild=max_tasks_per_child) as pool:
        indexed_worker_results = pool.imap_unordered(_IndexedWorker(worker), list(enumerate(tasks)))
        for done, (i, result) in enumerate(indexed_worker_results):
            results[i] = result
            report(done + 1, len(tasks))

    return results


class _IndexedWorker:
    def __init__(self, worker: Callable):
        self.worker = worker

    def __call__(self, indexed_task: Tuple[int, Any]) -> Tuple[int, Any]:
        index, task = indexed_task
        return index, self.worker(task)


def _filters_from_checkpoint(transport_processes) -> Dict[str, Any]:
    return _jsonable(transport_processes._active_filters)


def _collect_events(super_clusters: Dict[int, Any], outliers, md_indices: Dict[str, int]) \
        -> Tuple[List[Tuple[int, int, int, str, int, int, int, str]], List[str]]:
    """
    Collects transport events assigned to superclusters and global outliers
    :param super_clusters: superclusters from checkpoint
    :param outliers: OutlierTransportEvents from checkpoint
    :param md_indices: md_label -> index
    :return: list of (sc_id, md index, event type index, resname, resid, start frame, end frame, path_id) and
             sorted table of residue names
    """

    records = list()
    sources = [(sc_id, super_clusters[sc_id].transport_events) for sc_id in sorted(super_clusters.keys())]
    if outliers is not None:
        sources.append((-1, outliers.transport_events_global))

    for sc_id, transport_events in sources:
        for event_type in EVENT_TYPES:
            for md_label, events in transport_events.get(event_type, dict()).items():
                for path_id, traced_event in events:
                    resname, _, resid = traced_event[0].partition(":")
                    try:
                        resid_value = int(resid)
                    except ValueError:
                        resid_value = -1
                    start_frame, end_frame = traced_event[1]
                    records.append((sc_id, md_indices[md_label], EVENT_TYPES.index(event_type), resname,
                                    resid_value, int(start_frame), int(end_frame), str(path_id)))

    resnames = sorted({record[3] for record in records})
    return records, resnames


def _event_assignments(event_records: list, resname_indices: Dict[str, int]) -> np.ndarray:
    """
    Lists assignments of transport events to superclusters with an identifier of each distinct event, so that events
    assigned to several superclusters can be counted once when superclusters are merged
    :param event_records: records from _collect_events
    :param resname_indices: residue name -> index in the table of residue names
    :return: int32 array (A, 5) of (sc_id, md index, event type index, resname index, event ID) of assigned events,
             event IDs number the distinct (md index, AQUA-DUCT path ID, event type) triples
    """

    event_ids: Dict[Tuple[int, str, int], int] = dict()
    for key in sorted({(record[1], record[7], record[2]) for record in event_records}):
        event_ids[key] = len(event_ids)
    rows = sorted((record[0], record[1], record[2], resname_indices[record[3]],
                   event_ids[(record[1], record[7], record[2])]) for record in event_records if record[0] >= 0)
    return np.asarray(rows, dtype=np.int32).reshape(-1, 5)


def write_datapack(results_dir: str, out_file: Optional[str] = None, include_events: bool = False,
                   num_processes: int = 1, progress: Optional[ProgressCallback] = None,
                   transport_processes=None, stage: Optional[int] = None) -> str:
    """
    Creates datapack from TransportTools results folder (stage 5 or later must be finished)
    :param results_dir: path to the results folder (output_path of the run)
    :param out_file: output datapack file; by default 'ttviz.datapack' in the results folder
    :param include_events: if transport event records and their layered geometry should be included
    :param num_processes: number of parallel processes used to convert dump files
    :param progress: optional function receiving (fraction done, message)
    :param transport_processes: TransportProcesses object of the latest finished stage, if already loaded
    :param stage: stage of the provided transport_processes object
    :return: path to the created datapack
    """

    results_dir = os.path.abspath(results_dir)
    if out_file is None:
        out_file = os.path.join(results_dir, DEFAULT_DATAPACK_FILENAME)
    out_file = os.path.abspath(out_file)

    report = _ProgressReporter(progress, {"checkpoints": 1, "superclusters": 6, "clusters": 2, "assembly": 2})
    report("checkpoints", 0, 1, "Loading checkpoints")

    checkpoints = find_checkpoints(results_dir)
    if stage is not None:  # ignore checkpoints left by previous runs of later stages
        checkpoints = {stage_id: filename for stage_id, filename in checkpoints.items() if stage_id <= stage}
    finished_stages = sorted(stage_id for stage_id in checkpoints if stage_id >= 5)
    if stage is not None and transport_processes is not None and stage not in finished_stages:
        finished_stages = sorted(finished_stages + [stage])
    if not finished_stages:
        raise RuntimeError("No checkpoint of stage 5 or later found in '{}', datapack cannot be created".format(
            os.path.join(results_dir, "_internal", "checkpoints")))

    latest_stage = finished_stages[-1]
    if transport_processes is not None and stage == latest_stage:
        latest = transport_processes
    else:
        latest = _load_checkpoint(checkpoints[latest_stage])

    parameters = latest.parameters
    paths = _ResultsPaths(results_dir, parameters)
    md_labels = list(latest.caver_input_folders)
    md_indices = {md_label: i for i, md_label in enumerate(md_labels)}

    # result sets: initial from stage 5, filtered01 from stage 6, filtered02 from stage 10; filters of the stage-6 set
    # are kept by checkpoints of stages 6-9
    result_sets = [{"name": "initial", "stage": 5, "filters": None, "consider_events": False}]
    if latest_stage >= 6:
        if latest_stage <= 9:
            stage6_filters = _filters_from_checkpoint(latest)
        else:
            stage6_checkpoint = [stage_id for stage_id in range(6, 10) if stage_id in checkpoints]
            if not stage6_checkpoint:
                raise RuntimeError("Checkpoint from stages 6-9 is required to recover filters of stage 6")
            stage6_filters = _filters_from_checkpoint(_load_checkpoint(checkpoints[stage6_checkpoint[0]]))
        result_sets.append({"name": "filtered01", "stage": 6, "filters": stage6_filters, "consider_events": False})
    if latest_stage >= 10:
        result_sets.append({"name": "filtered02", "stage": 10, "filters": _filters_from_checkpoint(latest),
                            "consider_events": True})
    report("checkpoints", 1, 1, "Checkpoints loaded")

    events_assigned = bool(getattr(latest, "_events_assigned", False))
    groups = None
    if parameters.get("perform_comparative_analysis") and parameters.get("comparative_groups_definition"):
        groups = {group: sorted(members) for group, members in parameters["comparative_groups_definition"].items()}

    # labels of supercluster pathsets and statistics (same convention as save_super_clusters_visualization)
    labels = ["overall"]
    if parameters.get("perform_comparative_analysis"):
        grouped = set()
        if groups:
            for group in sorted(groups.keys()):
                labels.append(group)
                grouped.update(groups[group])
        labels.extend(md_label for md_label in md_labels if md_label not in grouped)

    super_clusters = latest._super_clusters
    sc_ids = sorted(super_clusters.keys())

    tmp_dir = tempfile.mkdtemp(prefix=".ttviz_datapack_", dir=os.path.dirname(out_file))
    try:
        # --- per-supercluster data
        profiles_folder = os.path.join(paths.get("super_cluster_profiles_folder"), "initial")
        path_sets_folder = paths.get("super_cluster_path_set_folder")
        sc_tasks = list()
        for sc_id in sc_ids:
            dump_filename = "super_cluster_{:02d}.dump".format(sc_id)
            profile_file = os.path.join(profiles_folder, dump_filename)
            if not os.path.exists(profile_file):
                raise RuntimeError("Supercluster profile '{}' is missing".format(profile_file))
            sc_tasks.append((sc_id, profile_file, os.path.join(path_sets_folder, dump_filename), md_indices,
                             labels, os.path.join(tmp_dir, "sc{:05d}".format(sc_id))))

        # small inputs are converted serially (spawning workers costs more than the work); workers converting
        # large profiles are recycled after each task to release memory
        profile_sizes = [os.path.getsize(task[1]) for task in sc_tasks]
        sc_processes = num_processes if sum(profile_sizes) > _PARALLEL_MIN_BYTES else 1
        sc_results = _run_tasks(_process_super_cluster_worker, sc_tasks, sc_processes,
                                lambda done, total: report("superclusters", done, total,
                                                           "Converted {}/{} supercluster profiles".format(done,
                                                                                                          total)),
                                max_tasks_per_child=1 if max(profile_sizes) > _RECYCLE_MIN_BYTES else None)

        # --- events
        event_records: List[Tuple[int, int, int, str, int, int, int, str]] = list()
        event_resnames: List[str] = list()
        if events_assigned:
            event_records, event_resnames = _collect_events(super_clusters, latest._outlier_transport_events,
                                                            md_indices)

        # --- per-MD layered pathsets of tunnel clusters (and of events)
        clusters4md: Dict[str, set] = {md_label: set() for md_label in md_labels}
        for sc_id in sc_ids:
            for md_label, validity in super_clusters[sc_id].tunnel_clusters_valid.items():
                clusters4md[md_label].update(validity.keys())

        events4md: Dict[str, List[Tuple[int, str]]] = {md_label: list() for md_label in md_labels}
        if include_events:
            for event_index, record in enumerate(event_records):
                entity_key = "{}_{}_{}".format(record[3].lower(), record[7], EVENT_TYPES[record[2]])
                events4md[md_labels[record[1]]].append((event_index, entity_key))

        caver_layered_folder = paths.get("layered_caver_network_data_path")
        aquaduct_layered_folder = paths.get("layered_aquaduct_network_data_path") \
            if "layered_aquaduct_network_data_path" in parameters else ""
        md_tasks = list()
        for md_index, md_label in enumerate(md_labels):
            if not clusters4md[md_label] and not events4md[md_label]:
                continue
            md_tasks.append((md_index, os.path.join(caver_layered_folder, md_label + "_layered_paths.dump"),
                             sorted(clusters4md[md_label]),
                             os.path.join(aquaduct_layered_folder, md_label + "_layered_paths.dump"),
                             events4md[md_label], os.path.join(tmp_dir, "md{:06d}".format(md_index))))

        layered_size = sum(os.path.getsize(task[1]) for task in md_tasks if os.path.exists(task[1]))
        md_processes = num_processes if layered_size > _PARALLEL_MIN_BYTES else 1
        md_results = _run_tasks(_process_md_worker, md_tasks, md_processes,
                                lambda done, total: report("clusters", done, total,
                                                           "Converted {}/{} layered networks".format(done, total)))
        missing_events = [key for result in md_results for key in result["missing_events"]]
        if missing_events:
            logger.warning("Geometry of {} transport events was not found in layered AQUA-DUCT data, e.g. "
                           "{}".format(len(missing_events), ", ".join(missing_events[:5])))

        # --- assembly
        report("assembly", 0, 1, "Assembling datapack")
        tmp_out_file = os.path.join(tmp_dir, "datapack.tmp")
        _assemble(tmp_out_file, results_dir, paths, latest, latest_stage, finished_stages, checkpoints, parameters,
                  md_labels, labels, groups, result_sets, sc_ids, sc_results, md_results, events_assigned,
                  event_records, event_resnames, include_events,
                  lambda done, total: report("assembly", done, total, "Assembling datapack"))
        os.replace(tmp_out_file, out_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    report("assembly", 1, 1, "Datapack saved to {}".format(out_file))
    logger.info("Datapack saved to: {}".format(out_file))
    return out_file


def _assemble(tmp_out_file: str, results_dir: str, paths: _ResultsPaths, latest, latest_stage: int,
              finished_stages: List[int], checkpoints: Dict[int, str], parameters: dict, md_labels: List[str],
              labels: List[str], groups: Optional[Dict[str, List[str]]], result_sets: List[dict], sc_ids: List[int],
              sc_results: List[Dict[str, Any]], md_results: List[Dict[str, Any]], events_assigned: bool,
              event_records: list, event_resnames: List[str], include_events: bool,
              report: Callable[[int, int], None]):
    """
    Writes datapack archive from converted data
    """

    def _load(prefix: str, name: str) -> np.ndarray:
        return np.load("{}_{}.npy".format(prefix, name), mmap_mode="r")

    num_members_done = 0
    writer = _DatapackZipWriter(tmp_out_file)
    try:
        # bottleneck residue table shared by all superclusters
        residue_table = sorted({residue for result in sc_results for residue in result["residue_table"]})
        residue_indices = {residue: i for i, residue in enumerate(residue_table)}
        residue_maps = [np.asarray([residue_indices[residue] for residue in result["residue_table"]],
                                   dtype=np.int32) for result in sc_results]

        num_tunnels = sum(result["num_tunnels"] for result in sc_results)
        num_spheres = sum(result["num_spheres"] for result in sc_results)
        num_residues = sum(result["num_residues"] for result in sc_results)
        max_abs_xyz = max([result["max_abs_xyz"] for result in sc_results] + [0.])
        max_radius = max([result["max_radius"] for result in sc_results] + [0.])

        quantize_xyz = max_abs_xyz / XYZ_SCALE < np.iinfo(np.int16).max - 1
        quantize_radius = max_radius / RADIUS_SCALE < np.iinfo(np.uint16).max - 1
        encoding = {
            "xyz": {"dtype": "int16", "scale": XYZ_SCALE} if quantize_xyz else {"dtype": "float32", "scale": 1.},
            "radius": ({"dtype": "uint16", "scale": RADIUS_SCALE} if quantize_radius
                       else {"dtype": "float32", "scale": 1.}),
        }

        def _xyz_chunks():
            for result in sc_results:
                data = _load(result["prefix"], "spheres")[:, :3]
                if quantize_xyz:
                    yield np.rint(data / XYZ_SCALE).astype(np.int16)
                else:
                    yield np.asarray(data, dtype=np.float32)

        def _radius_chunks():
            for result in sc_results:
                data = _load(result["prefix"], "spheres")[:, 3]
                if quantize_radius:
                    yield np.rint(data / RADIUS_SCALE).astype(np.uint16)
                else:
                    yield np.asarray(data, dtype=np.float32)

        def _offsets(counts_name: str):
            yield np.zeros(1, dtype=np.int64)
            position = 0
            for result in sc_results:
                counts = np.asarray(_load(result["prefix"], counts_name), dtype=np.int64)
                yield position + np.cumsum(counts)
                if counts.shape[0]:
                    position += int(counts.sum())

        def _residue_chunks():
            for result, residue_map in zip(sc_results, residue_maps):
                local = np.asarray(_load(result["prefix"], "residues"))
                yield residue_map[local] if local.shape[0] else np.zeros(0, dtype=np.int32)

        def _column_chunks(name: str, columns):
            for result in sc_results:
                yield np.asarray(_load(result["prefix"], name))[:, columns]

        tunnel_members = [
            ("tunnels/md.npy", np.int32, (num_tunnels,), _column_chunks("ids", 0)),
            ("tunnels/snapshot.npy", np.int32, (num_tunnels,), _column_chunks("ids", 1)),
            ("tunnels/cluster.npy", np.int32, (num_tunnels,), _column_chunks("ids", 2)),
            ("tunnels/tunnel_id.npy", np.int32, (num_tunnels,), _column_chunks("ids", 3)),
            ("tunnels/scalars.npy", np.float64, (num_tunnels, 5), _column_chunks("scalars", slice(None))),
            ("tunnels/bottleneck_xyz.npy", np.float32, (num_tunnels, 3),
             _column_chunks("bottleneck_xyz", slice(None))),
            ("tunnels/sphere_offsets.npy", np.int64, (num_tunnels + 1,), _offsets("num_spheres")),
            ("tunnels/xyz.npy", np.dtype(encoding["xyz"]["dtype"]), (num_spheres, 3), _xyz_chunks()),
            ("tunnels/radius.npy", np.dtype(encoding["radius"]["dtype"]), (num_spheres,), _radius_chunks()),
            ("tunnels/residue_offsets.npy", np.int64, (num_tunnels + 1,), _offsets("num_residues")),
            ("tunnels/residues.npy", np.int32, (num_residues,), _residue_chunks()),
        ]
        total_members = len(tunnel_members) + 8
        for name, dtype, shape, chunks in tunnel_members:
            writer.write_chunked_array(name, dtype, shape, chunks)
            num_members_done += 1
            report(num_members_done, total_members)

        sc_record_offsets = np.zeros(len(sc_results) + 1, dtype=np.int64)
        sc_record_offsets[1:] = np.cumsum([result["num_tunnels"] for result in sc_results])
        writer.write_array("superclusters/record_offsets.npy", sc_record_offsets)

        # pathsets: stored superclusters' first, then clusters and events
        pathset_chunks = [result["pathsets"] for result in sc_results] + md_results
        num_pathsets = sum(chunk["num_pathsets"] for chunk in pathset_chunks)
        num_nodes = sum(chunk["num_nodes"] for chunk in pathset_chunks)
        num_paths = sum(chunk["num_paths"] for chunk in pathset_chunks)
        num_path_nodes = sum(chunk["num_path_nodes"] for chunk in pathset_chunks)

        def _pathset_chunks(name: str):
            for chunk in pathset_chunks:
                yield _load(chunk["prefix"], name)

        def _pathset_offsets(name: str):
            yield np.zeros(1, dtype=np.int64)
            position = 0
            for chunk in pathset_chunks:
                counts = np.asarray(_load(chunk["prefix"], name), dtype=np.int64)
                yield position + np.cumsum(counts)
                if counts.shape[0]:
                    position += int(counts.sum())

        writer.write_chunked_array("pathsets/keys.npy", np.int32, (num_pathsets, 3), _pathset_chunks("keys"))
        writer.write_chunked_array("pathsets/node_offsets.npy", np.int64, (num_pathsets + 1,),
                                   _pathset_offsets("num_nodes"))
        writer.write_chunked_array("pathsets/path_offsets.npy", np.int64, (num_pathsets + 1,),
                                   _pathset_offsets("num_paths"))
        writer.write_chunked_array("pathsets/nodes.npy", np.float32, (num_nodes, 6), _pathset_chunks("nodes"))
        writer.write_chunked_array("pathsets/path_node_offsets.npy", np.int64, (num_paths + 1,),
                                   _pathset_offsets("path_lengths"))
        writer.write_chunked_array("pathsets/path_nodes.npy", np.int32, (num_path_nodes,),
                                   _pathset_chunks("paths"))
        num_members_done += 6
        report(num_members_done, total_members)

        # events
        sc_entries = list()
        event_counts = list()
        resname_indices = {resname: i for i, resname in enumerate(event_resnames)}
        for sc_id, result in zip(sc_ids, sc_results):
            super_cluster = latest._super_clusters[sc_id]
            entry = {
                "sc_id": int(sc_id),
                "clusters": {md_label: sorted(int(cls_id) for cls_id in validity.keys())
                             for md_label, validity in super_cluster.tunnel_clusters_valid.items()},
                "avg_direction": _jsonable(super_cluster.avg_direction),
                "pathset_labels": result["pathset_labels"],
            }
            if events_assigned:
                entry["num_events"] = _jsonable(super_cluster.num_events)
                entry["num_events_by_residue"] = _jsonable(super_cluster.num_events_by_residue)
            sc_entries.append(entry)

        if events_assigned:
            counts: Dict[Tuple[int, int, int, int], int] = dict()
            for record in event_records:
                key = (record[0], record[1], resname_indices[record[3]], record[2])
                counts[key] = counts.get(key, 0) + 1
            event_counts = [(*key, count) for key, count in sorted(counts.items())]
            writer.write_array("events/counts.npy", np.asarray(event_counts, dtype=np.int32).reshape(-1, 5))
            writer.write_array("events/assignments.npy", _event_assignments(event_records, resname_indices))
            if include_events:
                columns = [(record[0], record[1], record[2], resname_indices[record[3]], record[4], record[5],
                            record[6]) for record in event_records]
                writer.write_array("events/records.npy", np.asarray(columns, dtype=np.int32).reshape(-1, 7))
                path_ids = np.asarray([record[7] for record in event_records], dtype=str)
                writer.write_array("events/path_ids.npy", path_ids if path_ids.shape[0]
                                   else np.zeros(0, dtype="<U1"))

        outliers = None
        if events_assigned and latest._outlier_transport_events is not None:
            outliers = {"num_events": _jsonable(latest._outlier_transport_events.num_events),
                        "num_events_by_residue": _jsonable(latest._outlier_transport_events.num_events_by_residue)}

        # reference structure
        protein_file = os.path.join(paths.get("transformation_folder"), "ref_transformed.pdb")
        has_protein = os.path.exists(protein_file)
        if has_protein:
            with open(protein_file, "rb") as in_stream:
                writer.write_bytes("protein.pdb", in_stream.read())

        exported_parameters = {name: _jsonable(parameters.get(name)) for name in _PARAMETERS2EXPORT}
        manifest = {
            "format": DATAPACK_FORMAT,
            "version": DATAPACK_VERSION,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "writer": "transport_tools.libs.datapack {}".format(__version__),
            "source": {
                "results_dir": results_dir,
                "latest_stage": latest_stage,
                "finished_stages": finished_stages,
                "checkpoint_mtimes": {str(stage_id): os.path.getmtime(checkpoints[stage_id])
                                      for stage_id in finished_stages if stage_id in checkpoints},
            },
            "parameters": exported_parameters,
            "md_labels": md_labels,
            "total_num_md_sims": len(md_labels),
            "comparative_groups": groups,
            "labels": labels,
            "result_sets": result_sets,
            "encoding": encoding,
            "bottleneck_residues": residue_table,
            "superclusters": sc_entries,
            "events": {
                "assigned": events_assigned,
                "records_included": bool(events_assigned and include_events),
                "assignments_included": events_assigned,
                "types": list(EVENT_TYPES),
                "resnames": event_resnames,
                "outliers": outliers,
            },
            "has_protein": has_protein,
        }
        writer.write_bytes("manifest.json", json.dumps(manifest, indent=1).encode("utf-8"))
        report(total_members, total_members)
    finally:
        writer.close()


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create TTviz datapack from TransportTools results folder")
    parser.add_argument("results_dir", help="TransportTools results folder (output_path of the run)")
    parser.add_argument("-o", "--output", default=None,
                        help="output datapack file (default: {} in the results folder)".format(
                            DEFAULT_DATAPACK_FILENAME))
    parser.add_argument("-e", "--events", action="store_true",
                        help="include transport event records and their layered geometry")
    parser.add_argument("-n", "--num_cpus", type=int, default=1, help="number of parallel processes")
    parser.add_argument("--progress", action="store_true",
                        help="print machine-readable progress lines ('PROGRESS <fraction> <message>') to stdout")
    return parser


def main(argv: Optional[List[str]] = None):
    import logging

    args = init_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _print_progress(fraction: float, message: str):
        print("PROGRESS {:.4f} {}".format(fraction, message), flush=True)

    out_file = write_datapack(args.results_dir, args.output, include_events=args.events,
                              num_processes=args.num_cpus, progress=_print_progress if args.progress else None)
    print("DONE {}".format(out_file), flush=True)


if __name__ == "__main__":
    main()
