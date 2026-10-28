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

import contextlib
import io
import os
import pickle
import shutil
import struct
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

from transport_tools.libs import datapack


def _member_data_offset(filename: str, name: str) -> int:
    with zipfile.ZipFile(filename) as archive:
        info = archive.getinfo(name)
    with open(filename, "rb") as stream:
        stream.seek(info.header_offset)
        header = struct.unpack("<IHHHHHIIIHH", stream.read(30))
    return info.header_offset + 30 + header[9] + header[10]


class TestDatapackZipWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_datapack_")
        self.filename = os.path.join(self.temp_dir, "test.datapack")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_members_are_aligned_and_readable(self):
        writer = datapack._DatapackZipWriter(self.filename)
        arrays = {"a.npy": np.arange(7, dtype=np.int16), "dir/b.npy": np.ones((3, 5), dtype=np.float64),
                  "c.npy": np.zeros((0, 3), dtype=np.float32)}
        for name, array in arrays.items():
            writer.write_array(name, array)
        writer.write_chunked_array("d.npy", np.int64, (5, 2), [np.zeros((2, 2)), np.ones((3, 2))])
        writer.write_bytes("manifest.json", b"{}")
        writer.close()

        with zipfile.ZipFile(self.filename) as archive:
            for info in archive.infolist():
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(_member_data_offset(self.filename, info.filename) % 64, 0, info.filename)
            for name, array in arrays.items():
                np.testing.assert_array_equal(np.load(io.BytesIO(archive.read(name))), array)
                self.assertEqual(np.load(io.BytesIO(archive.read(name))).dtype, array.dtype)
            chunked = np.load(io.BytesIO(archive.read("d.npy")))
        np.testing.assert_array_equal(chunked, np.vstack([np.zeros((2, 2)), np.ones((3, 2))]).astype(np.int64))

    def test_chunked_array_checks_size(self):
        writer = datapack._DatapackZipWriter(self.filename)
        with self.assertRaises(RuntimeError):
            writer.write_chunked_array("x.npy", np.int32, (4,), [np.zeros(3)])
        writer.close()


class TestDatapackHelpers(unittest.TestCase):
    def test_encode_pathset(self):
        path_set = SimpleNamespace(
            nodes_data=np.array([[0, 0, 0, -1, 0, 0.5, 0.5], [1, 2, 3, 0, 0, 1.0, 0.1], [2, 3, 4, 1, 1, 1.5, 0.2]]),
            node_labels=["SP", "0_1", "1_1"],
            node_paths=[np.array(["SP", "0_1", "1_1"]), np.array(["SP", "1_1"])])
        nodes, paths, lengths = datapack._encode_pathset(path_set)
        self.assertEqual(nodes.shape, (3, 6))
        self.assertEqual(nodes.dtype, np.float32)
        np.testing.assert_array_equal(nodes[2], np.array([2, 3, 4, 1, 1, 1.5], dtype=np.float32))
        self.assertEqual(paths.tolist(), [0, 1, 2, 0, 2])
        self.assertEqual(lengths.tolist(), [3, 2])

    def test_encode_empty_pathset(self):
        nodes, paths, lengths = datapack._encode_pathset(SimpleNamespace(nodes_data=None, node_labels=[],
                                                                         node_paths=[]))
        self.assertEqual(nodes.shape, (0, 6))
        self.assertEqual(paths.shape, (0,))
        self.assertEqual(lengths.shape, (0,))

    def test_results_paths_rebase(self):
        parameters = {"output_path": "run_out", "internal_folder": "run_out/_internal",
                      "caver_results_path": "/data/caver", "other": "elsewhere/file"}
        paths = datapack._ResultsPaths("/moved/results", parameters)
        self.assertEqual(paths.get("internal_folder"), "/moved/results/_internal")
        self.assertEqual(paths.get("caver_results_path"), "/data/caver")
        self.assertEqual(paths.get("other"), "elsewhere/file")
        absolute = datapack._ResultsPaths("/moved/results", {"output_path": "/orig/out",
                                                              "x": "/orig/out/_internal/y"})
        self.assertEqual(absolute.get("x"), "/moved/results/_internal/y")

    def test_snapshot_mapping_parameters_are_exported(self):
        # readers map event frames to CAVER snapshots as utils.SnapshotFrameMap does (the writer names its keys
        # literally, being importable with TransportTools 0.9.8)
        from transport_tools.libs.utils import SnapshotFrameMap
        for name in ("caver_traj_offset", "caver_snapshot_stride", "interpolate_missing_snapshots4matching",
                     "snapshots_per_simulation", SnapshotFrameMap.MODE_KEY, SnapshotFrameMap.ID_STRIDE_KEY):
            self.assertIn(name, datapack._PARAMETERS2EXPORT)

    def test_parse_sc_ids(self):
        self.assertEqual(datapack.parse_sc_ids("1-3,8, 5,2"), [1, 2, 3, 5, 8])
        self.assertEqual(datapack.parse_sc_ids("7"), [7])
        for text in ("", "a", "1-", "-3", "5-2", "1.5", ","):
            with self.assertRaises(datapack.SelectionError, msg=text):
                datapack.parse_sc_ids(text)

    def test_prioritized_order(self):
        self.assertEqual(datapack._prioritized_order(SimpleNamespace(_prioritized_clusters={2: 5, 1: 3})), [3, 5])
        self.assertEqual(datapack._prioritized_order(SimpleNamespace()), [])

    def test_select_super_clusters(self):
        orders = {"initial": [5, 1, 2, 3, 4], "filtered01": [4, 2]}
        select = datapack._select_super_clusters
        self.assertEqual(select([1, 2, 3, 4, 5], orders, [3, 2], None), [2, 3])
        self.assertEqual(select([1, 2, 3, 4, 5], orders, None, "filtered01"), [2, 4])
        self.assertEqual(select([1, 2, 3, 4, 5], orders, None, "initial"), [1, 2, 3, 4, 5])
        self.assertEqual(select([1, 2, 3, 4, 5], orders, [1, 2, 3], "filtered01"), [2])
        for superclusters, result_set in (([2, 9], None), (None, "filtered02"), ([3], "filtered01")):
            with self.assertRaises(datapack.SelectionError):
                select([1, 2, 3, 4, 5], orders, superclusters, result_set)

    def test_color_ranks(self):
        # TTviz Lite colours: the set's priority order, then superclusters failing it in the initial order, then IDs
        orders = {"initial": [3, 1, 2], "filtered01": [2]}
        ranks = datapack._color_ranks([1, 2, 3, 4], orders, [1, 4])
        self.assertEqual(ranks, {"initial": {"1": 1, "4": 3}, "filtered01": {"1": 2, "4": 3}})

    def test_single_threaded_libraries_keep_values_set_by_the_user(self):
        # NumExpr reads these when pandas loads it with TransportTools; it would otherwise use up to 16 threads
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "4"}, clear=True):
            datapack._single_threaded_libraries()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
            for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
                self.assertEqual(os.environ[variable], "1")

    def test_selection_arguments(self):
        args = datapack.init_parser().parse_args(["results", "-s", "1-3,7", "-r", "filtered01"])
        self.assertEqual((args.superclusters, args.result_set), ([1, 2, 3, 7], "filtered01"))
        args = datapack.init_parser().parse_args(["results"])
        self.assertEqual((args.superclusters, args.result_set), (None, None))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for argv in (["results", "-s", "x"], ["results", "-r", "filtered03"]):
                with self.assertRaises(SystemExit):
                    datapack.init_parser().parse_args(argv)

    def test_collect_events(self):
        super_clusters = {
            2: SimpleNamespace(transport_events={"entry": {"md1": [("5", ("WAT:12", (1, 9)))]},
                                                 "release": {"md2": [("7", ("U01:1", (3, 4)))]}}),
            1: SimpleNamespace(transport_events={}),
        }
        outliers = SimpleNamespace(transport_events_global={"release": {"md1": [("9", ("WAT:3", (5, 6)))]}})
        records, resnames = datapack._collect_events(super_clusters, outliers, {"md1": 0, "md2": 1})
        self.assertEqual(resnames, ["U01", "WAT"])
        self.assertEqual(records, [(2, 0, 0, "WAT", 12, 1, 9, "5"), (2, 1, 1, "U01", 1, 3, 4, "7"),
                                   (-1, 0, 1, "WAT", 3, 5, 6, "9")])

    def test_event_assignments_identify_shared_events(self):
        # event "5" (md 0, entry) is assigned to superclusters 1 and 2; "5" released in md 0 and "5" in md 1 differ
        records = [(1, 0, 0, "WAT", 12, 1, 9, "5"), (2, 0, 0, "WAT", 12, 1, 9, "5"), (2, 0, 1, "WAT", 12, 9, 11, "5"),
                   (2, 1, 0, "U01", 1, 3, 4, "5"), (-1, 0, 1, "WAT", 3, 5, 6, "9")]
        assignments = datapack._event_assignments(records, {"U01": 0, "WAT": 1})
        self.assertEqual(assignments.dtype, np.int32)
        self.assertEqual(assignments.tolist(), [[1, 0, 0, 1, 0], [2, 0, 0, 1, 0], [2, 0, 1, 1, 2], [2, 1, 0, 0, 3]])
        self.assertEqual(datapack._event_assignments([], {}).shape, (0, 5))

    def test_event_assignments_tell_water_and_ligand_events_apart(self):
        # AQUA-DUCT numbers paths per run, one run per traced ligand: water and ligand path "1" of md 0 are two events
        records = [(1, 0, 0, "WAT", 12, 1, 9, "1"), (1, 0, 0, "U01", 301, 40, 52, "1"),
                   (2, 0, 0, "U01", 301, 40, 52, "1")]
        assignments = datapack._event_assignments(records, {"U01": 0, "WAT": 1})
        self.assertEqual(assignments.tolist(), [[1, 0, 0, 0, 0], [1, 0, 0, 1, 1], [2, 0, 0, 0, 0]])
        self.assertEqual(len(datapack.EVENT_IDENTITY), len(datapack._event_identity(records[0])))

    def test_num_events_by_residue_of_checkpoints_before_tt097(self):
        # TransportTools before 0.9.7 kept no per-residue event counts: recounted, they equal those it keeps now
        from transport_tools.libs.networks import SuperCluster
        from transport_tools.libs.tools import OutlierTransportEvents
        parameters = {"perform_comparative_analysis": True, "comparative_groups_definition": {"g1": ["md1", "md2"]},
                      "transformation_folder": "transform", "caver_foldername": "caver",
                      "super_cluster_path_set_folder": "pathsets"}
        events = [("md1", "5", "entry", ("WAT:12", (1, 9))), ("md1", "6", "entry", ("U01:1", (2, 3))),
                  ("md2", "7", "release", ("WAT:3", (3, 4))), ("md3", "2", "entry", ("WAT:7", (4, 5))),
                  ("md3", "3", "release", ("U01:1", (6, 8)))]
        super_cluster = SuperCluster(sc_id=1, parameters=parameters, total_num_md_sims=3)
        super_cluster.set_properties({"g1": True})  # events of md3 cannot be assigned locally
        outliers = OutlierTransportEvents(parameters)
        for i, (md_label, path_id, event_type, traced_event) in enumerate(events):
            super_cluster.add_transport_event(md_label, path_id, event_type, traced_event)
            outliers.add_transport_event(md_label, path_id, event_type, traced_event, globally_unassigned=i % 2 == 0)
        self.assertNotIn("md3", super_cluster.num_events)
        membership = {"md1": "g1", "md2": "g1"}
        for holder, is_outliers in ((super_cluster, False), (outliers, True)):
            kept = holder.num_events_by_residue
            del holder.num_events_by_residue
            self.assertEqual(datapack._num_events_by_residue(holder, membership, outliers=is_outliers), kept)
            holder.num_events_by_residue = kept
            self.assertIs(datapack._num_events_by_residue(holder, membership, outliers=is_outliers), kept)

    def test_parameters_renamed_in_tt098_are_exported_from_older_checkpoints(self):
        from transport_tools.libs.config import AnalysisConfig
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertEqual({old: new for new, old in datapack._LEGACY_PARAMETER_NAMES.items()}, config.legacy_params)
        for new in datapack._LEGACY_PARAMETER_NAMES:
            self.assertIn(new, datapack._PARAMETERS2EXPORT)
        self.assertEqual(datapack._exported_parameter({"min_tunnel_radius4clustering": 0.75},
                                                      "relevant_tunnel_min_radius"), 0.75)
        self.assertEqual(datapack._exported_parameter({"min_tunnel_radius4clustering": None,
                                                       "relevant_tunnel_min_radius": 0.9},
                                                      "relevant_tunnel_min_radius"), 0.9)
        self.assertIsNone(datapack._exported_parameter({}, "caver_snapshot_stride"))

    def test_event_geometry_of_checkpoints_before_tt097(self):
        # layered AQUA-DUCT data of TransportTools before 0.9.7 name events without the traced residue
        path_set = SimpleNamespace(nodes_data=np.array([[0, 0, 0, -1, 0, 0.5], [1, 2, 3, 0, 1, 1.0]]),
                                   node_labels=["SP", "0_1"], node_paths=[np.array(["SP", "0_1"])])
        tmp_dir = tempfile.mkdtemp()
        try:
            aquaduct_file = os.path.join(tmp_dir, "md1_layered_paths.dump")
            with open(aquaduct_file, "wb") as out_stream:
                pickle.dump({"7_entry": path_set}, out_stream)
            result = datapack._process_md_worker((0, os.path.join(tmp_dir, "none.dump"), [], aquaduct_file,
                                                  [(0, "wat_7_entry"), (1, "wat_8_release")],
                                                  os.path.join(tmp_dir, "md0")))
            self.assertEqual((result["num_pathsets"], result["missing_events"]), (1, ["wat_8_release"]))
            self.assertEqual(np.load(os.path.join(tmp_dir, "md0_keys.npy")).tolist(),
                             [[datapack.PATHSET_KIND_EVENT, 0, 0]])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_stale_checkpoints_of_an_earlier_stage5_run_are_skipped(self):
        def run(clusters):
            return SimpleNamespace(_active_filters={}, _super_clusters={sc_id: SimpleNamespace(
                tunnel_clusters_valid={md: {cls_id: True for cls_id in ids} for md, ids in mds.items()},
                transport_events={}) for sc_id, mds in clusters.items()})

        old_run = run({1: {"md1": [1, 2]}, 2: {"md1": [3]}})
        new_run = run({1: {"md1": [1]}, 2: {"md1": [2, 3]}})
        filtered = run({1: {"md1": [1]}, 2: {"md1": [2, 3]}})
        filtered._super_clusters[2].tunnel_clusters_valid["md1"][3] = False  # filtering changes only validity
        self.assertEqual(datapack.supercluster_fingerprint(filtered), datapack.supercluster_fingerprint(new_run))
        checkpoints = {stage: "stage{:03d}".format(stage) for stage in (5, 6, 7, 10)}
        files = {"stage005": new_run, "stage006": filtered, "stage007": old_run, "stage010": old_run}
        stage, latest, stale = datapack._newest_current_checkpoint(checkpoints, [5, 6, 7, 10], files.__getitem__)
        self.assertEqual((stage, stale), (6, [7, 10]))
        self.assertIs(latest, filtered)
        files["stage006"] = old_run
        stage, latest, stale = datapack._newest_current_checkpoint(checkpoints, [5, 6, 7, 10], files.__getitem__)
        self.assertEqual((stage, stale), (5, [6, 7, 10]))
        self.assertIs(latest, new_run)
        files.update(stage006=filtered, stage007=filtered, stage010=filtered)
        self.assertEqual(datapack._newest_current_checkpoint(checkpoints, [5, 6, 7, 10], files.__getitem__)[2], [])

    def test_stale_checkpoints_of_later_stage_reruns_are_skipped(self):
        """Re-running stage 6 (or 9) alone keeps the superclusters, so only the content carried over by later stages
        tells their old checkpoints apart: stages 7-9 keep the filters and validity flags, stage 10 the events"""
        import copy

        def run(min_snapshots, valid, events=None):
            events = events or {}
            return SimpleNamespace(
                _active_filters={"min_snapshots_num": min_snapshots}, _outlier_transport_events=None,
                _super_clusters={1: SimpleNamespace(tunnel_clusters_valid={"md1": {1: True, 2: valid}},
                                                    transport_events=events)})

        assigned = {"entry": {"md1": [("1", ("WAT:7", (1, 3))), ("1", ("U01:301", (40, 52)))]}}
        old = {5: run(-1, True), 6: run(1, True), 7: run(1, True), 9: run(1, True, assigned),
               10: run(5, False, assigned)}
        checkpoints = {stage: stage for stage in old}
        stages = sorted(old)
        self.assertEqual(datapack._newest_current_checkpoint(checkpoints, stages, old.__getitem__)[2], [])

        rerun6 = {**old, 6: run(600, False)}  # stage 6 re-run alone with other filters
        stage, latest, stale = datapack._newest_current_checkpoint(checkpoints, stages, rerun6.__getitem__)
        self.assertEqual((stage, stale), (6, [7, 9, 10]))
        self.assertIs(latest, rerun6[6])

        moved = {"entry": {"md1": [("1", ("WAT:7", (1, 3)))]}}
        rerun9 = {**old, 9: run(1, True, moved)}  # stage 9 re-run alone, e.g. another assignment cutoff
        stage, latest, stale = datapack._newest_current_checkpoint(checkpoints, stages, rerun9.__getitem__)
        self.assertEqual((stage, stale), (9, [10]))

        reordered = copy.deepcopy(old)  # the order of events does not matter, only which events are held
        reordered[10]._super_clusters[1].transport_events["entry"]["md1"].reverse()
        self.assertEqual(datapack._newest_current_checkpoint(checkpoints, stages, reordered.__getitem__)[2], [])
        self.assertIsNone(datapack.stale_checkpoint_reason(10, datapack.checkpoint_signature(old[10]), 7,
                                                           datapack.checkpoint_signature(old[7])))

    def test_sphere_encoding(self):
        encoding = datapack._sphere_encoding(120.0, 3.5, 0.2, 80.0)
        self.assertEqual(encoding, {"xyz": {"dtype": "int16", "scale": datapack.XYZ_SCALE},
                                    "radius": {"dtype": "uint16", "scale": datapack.RADIUS_SCALE},
                                    "length": {"dtype": "uint16", "scale": datapack.LENGTH_SCALE}})
        encoding = datapack._sphere_encoding(400.0, 70.0, -0.5, 80.0)  # out of range or negative: float32
        for member in ("xyz", "radius", "length"):
            self.assertEqual(encoding[member], {"dtype": "float32", "scale": 1.})
        self.assertEqual(datapack._sphere_encoding(1.0, 1.0, 0.0, 700.0)["length"]["dtype"], "float32")

    def test_jsonable(self):
        value = datapack._jsonable({"a": {"x", "y"}, 1: np.int64(3), "f": np.float32(0.5), "arr": np.arange(2),
                                    "t": (np.bool_(True), None)})
        self.assertEqual(value, {"a": ["x", "y"], "1": 3, "f": 0.5, "arr": [0, 1], "t": [True, None]})

    def test_find_checkpoints_and_missing_stage5(self):
        temp_dir = tempfile.mkdtemp(prefix="TT_test_datapack_")
        try:
            folder = os.path.join(temp_dir, "_internal", "checkpoints")
            os.makedirs(folder)
            for stage in (1, 4):
                open(os.path.join(folder, "stage{:03d}.dump".format(stage)), "w").close()
            self.assertEqual(sorted(datapack.find_checkpoints(temp_dir)), [1, 4])
            with self.assertRaises(RuntimeError):
                datapack.write_datapack(temp_dir)
        finally:
            shutil.rmtree(temp_dir)

    def test_progress_reporter(self):
        reports = []
        reporter = datapack._ProgressReporter(lambda fraction, message: reports.append((fraction, message)),
                                              {"a": 1, "b": 3})
        reporter("a", 1, 1, "first")
        reporter("b", 1, 2, "second")
        self.assertAlmostEqual(reports[0][0], 0.25)
        self.assertAlmostEqual(reports[1][0], 0.625)


if __name__ == "__main__":
    unittest.main()
