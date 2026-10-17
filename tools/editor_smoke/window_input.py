"""Real mouse input for the window smoke on Windows. Cursor moves and clicks play on a thread while
the editor draws frames, because a title-bar drag runs the system's move loop inside a frame."""
import ctypes
import os
import threading
import time
from ctypes import wintypes

_user = ctypes.windll.user32
_user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_uint]
_MOVE, _LEFTDOWN, _LEFTUP = 0x0001, 0x0002, 0x0004
_NOSIZE, _NOMOVE, _NOACTIVATE = 0x0001, 0x0002, 0x0010
_TOPMOST, _NOTOPMOST = -1, -2


def window():
    """The editor's visible top-level window in this process."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        pid = wintypes.DWORD()
        _user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == os.getpid() and _user.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    _user.EnumWindows(each, 0)
    return found[0]


def topmost(on):
    """Keeps the editor above other windows so the input lands on it."""
    _user.SetWindowPos(window(), _TOPMOST if on else _NOTOPMOST, 0, 0, 0, 0, _NOSIZE | _NOMOVE | _NOACTIVATE)


def work_area():
    """The primary monitor's work area as (x, y, width, height)."""
    r = wintypes.RECT()
    _user.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0)
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def drive(app, gesture, until=None):
    """Plays ("move", x, y), ("down",), ("up",) and ("wait", seconds) while frames pump; returns
    the window state after, once `until(state)` holds or 2 s have passed."""
    def play():
        for g in gesture:
            if g[0] == "move":
                # SetCursorPos places exactly; the nudge makes it a mouse move the window sees.
                _user.SetCursorPos(int(g[1]), int(g[2]))
                _user.mouse_event(_MOVE, 1, 0, 0, 0)
                _user.mouse_event(_MOVE, -1, 0, 0, 0)
                _user.SetCursorPos(int(g[1]), int(g[2]))
            elif g[0] == "down":
                _user.mouse_event(_LEFTDOWN, 0, 0, 0, 0)
            elif g[0] == "up":
                _user.mouse_event(_LEFTUP, 0, 0, 0, 0)
            time.sleep(g[1] if g[0] == "wait" else 0.03)

    t = threading.Thread(target=play)
    t.start()
    while t.is_alive():
        app.update(True)
    for _ in range(10):
        app.update(True)
    state = app.actions.run("window.state")
    # The system can finish a maximize or minimize some frames after the click.
    deadline = time.monotonic() + 2.0
    while until is not None and not until(state) and time.monotonic() < deadline:
        app.update(True)
        state = app.actions.run("window.state")
    return state


def drag(app, x, y, dx, dy, steps=10):
    """Presses at screen (x, y), moves by (dx, dy) and releases."""
    moves = [("move", x + dx * k / steps, y + dy * k / steps) for k in range(1, steps + 1)]
    return drive(app, [("move", x, y), ("wait", 0.15), ("down",), ("wait", 0.15), *moves, ("wait", 0.15), ("up",)])


def double_click(app, x, y, until=None):
    return drive(app, [("move", x, y), ("wait", 0.15), ("down",), ("up",), ("down",), ("up",), ("wait", 0.3)], until)


def grab_point(state):
    """The middle of the title bar's widest drag span, in screen pixels."""
    x0, x1 = max(state["title_bar"]["drag"], key=lambda d: d[1] - d[0])
    return state["x"] + (x0 + x1) / 2, state["y"] + state["title_bar"]["height"] / 2


def button_point(state, i):
    """The centre of window button i (minimize, maximize, close), in screen pixels."""
    x0 = state["title_bar"]["drag"][-1][1]
    w = (state["width"] - x0) / 3
    return state["x"] + x0 + (i + 0.5) * w, state["y"] + state["title_bar"]["height"] / 2
