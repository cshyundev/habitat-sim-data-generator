"""Import-order regression tests.

``src.utils.coords`` <-> ``src.datatypes`` used to form an import cycle
(coords -> datatypes.pose -> datatypes/__init__ -> datatypes.map -> coords)
that only trips when coords is the *first* of the two imported -- exactly the
``stream_data.py`` entry chain (runtime_config -> robot_config -> robot ->
coords). Test discovery imports datatypes first, so the whole suite can be
green while the entry point is broken; these tests import each module in a
fresh interpreter to pin the entry-point order.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_in_fresh_interpreter(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )


class TestFreshInterpreterImports(unittest.TestCase):
    def test_coords_imports_first(self):
        proc = _import_in_fresh_interpreter("src.utils.coords")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_runtime_config_imports_first(self):
        # The exact first project import stream_data.py performs.
        proc = _import_in_fresh_interpreter("src.runtime_config")
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
