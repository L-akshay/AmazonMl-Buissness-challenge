import unittest
import os
import subprocess
import sys

from src.local_benchmark import GIB, WindowsJob, stop_reason


class BenchmarkGuardTests(unittest.TestCase):
    def test_monitor_stops_for_each_resource_limit(self):
        self.assertIsNone(stop_reason(10, 5*GIB, 6*GIB, 300))
        self.assertEqual(stop_reason(301, 5*GIB, 6*GIB, 300), "time_limit")
        self.assertEqual(stop_reason(10, 3*GIB, 6*GIB, 300), "system_available_memory_below_4_GiB")
        self.assertEqual(stop_reason(10, 5*GIB, 4*GIB, 300), "disk_free_below_5_GiB")

    @unittest.skipUnless(os.name == "nt", "Windows hard memory guard")
    def test_job_rejects_allocation_above_memory_cap(self):
        code = "import sys\nsys.stdin.readline()\ntry:\n bytearray(256*1024*1024)\nexcept MemoryError:\n print('bounded')\nelse:\n print('UNBOUNDED')\n"
        job = WindowsJob(128 * 1024 * 1024)
        process = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            job.assign(process)
            stdout, stderr = process.communicate("go\n", timeout=15)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(stdout.strip(), "bounded")
        finally:
            job.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
