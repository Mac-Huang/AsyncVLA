"""Tests for GPU-only timing, independent of CUDA and ML dependencies."""

import ast
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from gpu_trace import PREFIX, analyze_trace, interval_union_ns


class TraceTests(unittest.TestCase):
    def fixture(self, directory):
        db = Path(directory) / "trace.sqlite"
        connection = sqlite3.connect(db)
        connection.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                text TEXT, textId INTEGER, eventType INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER,
                streamId INTEGER, deviceId INTEGER, shortName INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, correlationId INTEGER
            );
        """)
        pid = 123 << 24
        labels = [PREFIX + "isolated/base/0", PREFIX + "async/edge/0"]
        for i, label in enumerate(labels):
            connection.execute("INSERT INTO StringIds VALUES (?,?)", (i + 1, label))
        connection.execute("INSERT INTO StringIds VALUES (3,'kernel')")
        # Two host threads submit overlapping GPU work, with a long CPU gap.
        connection.executemany("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,59)", [
            (0, 1000, pid + 1, labels[0], None),
            (0, 1000, pid + 2, None, 2),
        ])
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)", [
            (10, 20, pid + 1, 1), (500, 510, pid + 1, 2),
            (10, 20, pid + 2, 3), (600, 610, pid + 2, 4),
            (30, 40, pid + 1, 5),
            (1100, 1110, pid + 1, 6),
            # Same correlation in another process must not contaminate the result.
            (10, 20, (124 << 24) + 1, 1),
        ])
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,3)", [
            (30, 80, pid, 1, 7, 0), (550, 600, pid, 2, 7, 0),
            (50, 150, pid, 3, 8, 0), (120, 200, pid, 4, 9, 0),
            (30, 999, 124 << 24, 1, 7, 0),
            (1200, 1800, pid, 6, 7, 0),
        ])
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (30,900,5)")
        connection.commit()
        connection.close()
        dispatch = Path(directory) / "dispatch.json"
        dispatch.write_text(json.dumps({"metadata": {}, "calls": [
            {"label": labels[0], "condition": "isolated", "component": "base", "call_id": 0},
            {"label": labels[1], "condition": "async", "component": "edge", "call_id": 0},
        ]}))
        return db, dispatch, Path(directory) / "result.json"

    def test_union_handles_overlap_and_separate_intervals(self):
        self.assertEqual(interval_union_ns([(10, 20), (15, 30), (50, 60)]), 30)
        self.assertEqual(interval_union_ns([]), 0)
        with self.assertRaises(ValueError):
            interval_union_ns([(20, 10)])

    def test_gpu_time_excludes_host_gap_transfers_and_other_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            db, dispatch, result = self.fixture(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                data = analyze_trace(db, dispatch, result)
            base, edge = data["calls"]
            self.assertEqual(base["gpu_kernel_busy_ms"], 100 / 1e6)
            self.assertEqual(base["gpu_kernel_span_ms"], 570 / 1e6)
            self.assertEqual(base["gpu_gap_ms"], 470 / 1e6)
            self.assertEqual(base["gpu_kernel_count"], 2)
            self.assertEqual(edge["gpu_kernel_busy_ms"], 150 / 1e6)
            self.assertEqual(edge["gpu_kernel_sum_ms"], 180 / 1e6)
            self.assertEqual(edge["cuda_stream_ids"], [8, 9])
            self.assertEqual(json.loads(result.read_text())["metric"]["primary"],
                             "gpu_kernel_busy_ms")

    def test_missing_call_or_kernel_fails_instead_of_reporting_zero(self):
        for mutation in (
            "DELETE FROM NVTX_EVENTS WHERE globalTid % 16777216 = 2",
            "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=7",
        ):
            with tempfile.TemporaryDirectory() as directory:
                db, dispatch, result = self.fixture(directory)
                with sqlite3.connect(db) as connection:
                    connection.execute(mutation)
                with self.assertRaises(ValueError):
                    analyze_trace(db, dispatch, result)
                self.assertFalse(result.exists())

    def test_duplicate_call_label_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            db, dispatch, result = self.fixture(directory)
            data = json.loads(dispatch.read_text())
            data["calls"].append(data["calls"][0])
            dispatch.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unique"):
                analyze_trace(db, dispatch, result)


if __name__ == "__main__":
    unittest.main()
