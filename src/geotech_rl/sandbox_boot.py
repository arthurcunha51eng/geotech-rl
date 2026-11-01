"""Runs one cell of agent code under an audit hook. Executed as a script, never imported.

The hook cannot be removed from inside the interpreter, and native libraries load before it starts. It is not
a boundary against a determined exploit: it removes the easy paths, and the container or disposable machine
around it is the outer layer.
"""
import ast
import os
import resource
import sys
import traceback

CELL = sys.argv[1]
ROOT = os.path.realpath(os.getcwd())
CPU_SECONDS = 12
MEMORY_BYTES = 3 * 1024 ** 3
FILE_BYTES = 20 * 1024 ** 2
OPEN_FILES = 256

source = open(CELL, encoding="utf-8").read()

import numpy  # noqa: E402,F401  (native libraries must load before the hook)
import scipy.integrate  # noqa: E402,F401
import scipy.optimize  # noqa: E402,F401
import scipy.stats  # noqa: E402,F401

try:
    import pandas  # noqa: F401
except ImportError:
    pass

# Libraries import each other lazily (pandas loads zstandard inside read_csv), so all of site-packages stays
# readable. What must stay closed is named below; editable installs live outside site-packages anyway.
SITE_PACKAGES = os.path.dirname(os.path.dirname(numpy.__file__))
READABLE = tuple(os.path.realpath(r) + os.sep for r in (os.path.dirname(os.__file__), SITE_PACKAGES))
PRIVATE = ("geotech_rl", "geotech_poles", "prime_rl", "verifiers")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

BLOCKED = {
    "socket.__new__", "socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo",
    "subprocess.Popen", "os.system", "os.exec", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn",
    "os.kill", "os.killpg", "os.chdir", "os.fchdir", "os.chroot", "os.startfile", "pty.spawn",
    "sys.addaudithook", "sys.settrace", "sys.setprofile", "sys._getframe", "resource.setrlimit", "resource.prlimit",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect", "webbrowser.open",
    "mmap.__new__", "pickle.find_class", "gc.get_objects", "gc.get_referrers", "gc.get_referents",
    "shutil.rmtree", "tempfile.mkdtemp", "tempfile.mkstemp",
}
PATH_EVENTS = {"os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link", "os.symlink",
               "os.truncate", "os.chmod", "os.chown", "os.utime", "os.walk", "glob.glob", "glob.glob/2"}


def inside(path) -> bool:
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    real = os.path.realpath(os.fspath(path))
    return real == ROOT or real.startswith(ROOT + os.sep)


def readable(path) -> bool:
    real = os.path.realpath(os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path))
    if inside(real):
        return True
    return real.startswith(READABLE) and not any(part in PRIVATE for part in real.split(os.sep))


def hook(event, args):
    if event in BLOCKED or event.startswith("ctypes.") or event.startswith("winreg."):
        raise PermissionError(f"blocked in this sandbox: {event}")
    if event == "open":
        path, mode, flags = args
        if isinstance(path, int) or path is None:
            return
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS)
        if not (inside(path) if writing else readable(path)):
            raise PermissionError(f"blocked in this sandbox: open {path}")
    elif event in PATH_EVENTS:
        for path in args[:2]:
            if isinstance(path, (str, bytes, os.PathLike)) and not inside(path):
                raise PermissionError(f"blocked in this sandbox: {event} {path}")


def split_cell(text: str):
    """Compiles a cell the way a notebook runs it: a final bare expression is evaluated and shown."""
    tree = ast.parse(text, "cell")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final = ast.Expression(tree.body.pop().value)
        return compile(tree, "cell", "exec"), compile(final, "cell", "eval")
    return compile(tree, "cell", "exec"), None


def explain(error: Exception, lines: list[str]) -> str:
    """Only the agent's own frames: library internals fill its context and say nothing it can act on."""
    report = ["Traceback (most recent call last):"]
    for frame in traceback.extract_tb(error.__traceback__):
        if frame.filename == "cell":
            report.append(f'  File "cell", line {frame.lineno}, in {frame.name}\n    {lines[frame.lineno - 1].strip()}')
    return "\n".join(report) + "\n" + "".join(traceback.format_exception_only(error))


resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
resource.setrlimit(resource.RLIMIT_NOFILE, (OPEN_FILES, OPEN_FILES))
cell_lines = source.splitlines()
try:
    body, final = split_cell(source)
    sys.argv = ["cell"]
    sys.addaudithook(hook)
    del source
    scope = {"__name__": "__main__"}
    exec(body, scope)
    if final is not None and (value := eval(final, scope)) is not None:
        print(repr(value))
except Exception as error:
    sys.stdout.flush()
    sys.stderr.write(explain(error, cell_lines))
    sys.exit(1)
