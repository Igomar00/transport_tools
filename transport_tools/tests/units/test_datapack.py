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

import io
import os
import shutil
import struct
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

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
        self.assertEqual(assignments.tolist(), [[1, 0, 0, 1, 0], [2, 0, 0, 1, 0], [2, 0, 1, 1, 1], [2, 1, 0, 0, 3]])
        self.assertEqual(datapack._event_assignments([], {}).shape, (0, 5))

    def test_stale_checkpoints_of_an_earlier_stage5_run_are_skipped(self):
        def run(clusters):
            return SimpleNamespace(_super_clusters={sc_id: SimpleNamespace(tunnel_clusters_valid={
                md: {cls_id: True for cls_id in ids} for md, ids in mds.items()}) for sc_id, mds in clusters.items()})

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
