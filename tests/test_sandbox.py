import asyncio

import pytest

from geotech_rl import sandbox


def run(code, root):
    return asyncio.run(sandbox.run(code, root))


@pytest.fixture
def root(tmp_path):
    (tmp_path / "records.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("code,expected", [
    ("print(open('records.csv').read().splitlines()[1])", "1,2"),
    ("import numpy as np; print(np.linalg.solve(np.eye(2), np.ones(2)).sum())", "2.0"),
    ("from scipy.optimize import brentq; print(round(brentq(lambda x: x - 3, 0, 5), 3))", "3.0"),
    ("from scipy.integrate import quad; print(round(quad(lambda x: x, 0, 2)[0], 3))", "2.0"),
    ("import csv, json, math, re, statistics; print(math.floor(2.5))", "2"),
    ("open('out.txt', 'w').write('x'); print(open('out.txt').read())", "x"),
    ("import os; print(sorted(os.listdir('.'))[0])", "records.csv"),
    ("import glob; print(glob.glob('*.csv')[0])", "records.csv"),
    ("import pandas as pd; print(pd.read_csv('records.csv')['b'].sum())", "2"),
    ("import pandas as pd; d = pd.read_csv('records.csv'); d.to_csv('o.csv'); print(pd.read_csv('o.csv').shape[0])", "1"),
    ("import pandas as pd; print(pd.read_csv('records.csv').to_json(orient='records'))", '"b":2'),
    ("import numpy as np; print(np.genfromtxt('records.csv', delimiter=',', skip_header=1).sum())", "3.0"),
    ("import json, datetime, collections, itertools, functools, decimal, fractions; print(json.dumps({'a': 1}))", '"a"'),
    ("import statistics, random, string, textwrap, pathlib; print(pathlib.Path('records.csv').exists())", "True"),
])
def test_engineering_work_runs(code, expected, root):
    assert expected in run(code, root)


@pytest.mark.parametrize("code,expected", [
    ("1 + 1", "2"),
    ("x = 2\nx * 3", "6"),
    ("import pandas as pd\npd.read_csv('records.csv').shape", "(1, 2)"),
])
def test_a_final_expression_is_shown_like_a_notebook(code, expected, root):
    """189 of 191 cells had no print(): the model saw nothing and invented its answer."""
    assert expected in run(code, root)


def test_a_silent_cell_says_how_to_see_results(root):
    assert run("x = 1", root) == sandbox.SILENT
    assert run("print('a')", root).strip() == "a"


def test_a_failed_cell_reports_its_own_line_and_no_library_frames(root):
    output = run("import pandas as pd\nprint('before')\npd.read_csv('records.csv')['missing']", root)
    assert "before" in output and "line 3" in output and "KeyError: 'missing'" in output
    assert "site-packages" not in output and "sandbox_boot" not in output


def test_a_syntax_error_reports_the_cell(root):
    output = run("def broken(:\n    pass", root)
    assert "SyntaxError" in output and "sandbox_boot" not in output


@pytest.mark.parametrize("code", [
    "import socket; socket.socket()",
    "import socket; socket.create_connection(('example.com', 80), 2)",
    "import urllib.request; urllib.request.urlopen('http://example.com')",
    "import subprocess; subprocess.run(['ls', '/'])",
    "import os; os.system('ls /')",
    "import os; os.fork()",
    "import os; os.execv('/bin/ls', ['ls'])",
    "open('/etc/passwd').read()",
    "open('/tmp/escape.txt', 'w').write('x')",
    "import os; os.open('/etc/hostname', os.O_RDONLY)",
    "import os; os.listdir('/')",
    "import os; os.chdir('/')",
    "import os; os.remove('/tmp/nothing')",
    "import ctypes; ctypes.CDLL(None)",
    "import ctypes; ctypes.pythonapi.Py_IsInitialized()",
    "import geotech_rl",
    "import geotech_rl.solver",
    "import site, os; p = site.getsitepackages()[0]; print(open(os.path.join(p, 'geotech_rl', 'solver.py')).read())",
    "import sys; sys.addaudithook(lambda *a: None); open('/etc/passwd').read()",
    "import sys; sys._getframe(1)",
    "import gc; gc.get_objects()",
    "import resource; resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))",
    "import pickle, os; pickle.loads(pickle.dumps(os.system))",
    "open('../outside.txt', 'w').write('x')",
])
def test_escapes_are_blocked(code, root):
    output = run(code + "\nprint('ESCAPED')", root)
    assert "ESCAPED" not in output, output
    assert "Error" in output, output


def test_runaway_code_is_stopped(root):
    assert "stopped" in asyncio.run(sandbox.run("while True: pass", root, wall_seconds=2.0))


def test_output_is_clipped(root):
    assert len(run("print('x' * 100000)", root)) < sandbox.OUTPUT_LIMIT + 100


def test_environment_carries_no_secrets(root, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    assert "secret-value" not in run("import os; print(dict(os.environ))", root)
