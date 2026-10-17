"""Panels a script builds in the editor shell, through ``app.ui``.

    panel = app.ui.panel("tools", "Robot Tools", dock="right", icon="robot")
    gain = panel.float_field("Gain", value=1.0, min=0.0, max=4.0, slider=True,
                             on_change=lambda v: print("gain", v))
    row = panel.hstack()
    row.button("Reset", icon="reset", on_click=lambda: app.actions.run("sim.reset"))
    app.ui.toast("Tools ready", "ok")

Nodes live in the editor until removed. Events (click, change, commit, select, activate, run,
toggle, enter, browse) reach their callbacks at the top of the next frame, on the thread that
runs ``update()``; what a callback prints, and its traceback, goes to the editor console.
``icon`` takes a shell icon name ("play", "robot", "trash", ...) or a glyph string.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import traceback
from typing import Callable, Optional, Sequence

_LEVELS = {"info": 0, "ok": 1, "warn": 2, "error": 3}
_EVENTS = {"on_click": "click", "on_change": "change", "on_commit": "commit", "on_select": "select",
           "on_activate": "activate", "on_run": "run", "on_toggle": "toggle", "on_enter": "enter",
           "on_browse": "browse"}


def _adapt(fn: Callable) -> Callable:
    """fn(value), or fn() when it takes no positional argument."""
    params = inspect.signature(fn).parameters.values()
    if any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in params):
        return fn
    return lambda _value: fn()


def _capture(fn: Callable[[], object]):
    """Run fn with stdout and stderr captured; a traceback lands in the text."""
    sink = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            fn()
        except SystemExit:
            ok = False
            print("SystemExit ignored (a script cannot exit the editor)")
        except Exception:
            ok = False
            traceback.print_exc()
    return ok, sink.getvalue()


class Node:
    """One widget or container. Props are the keyword arguments of the builder that made it;
    every node also takes ``id``, ``width``, ``tooltip``, ``visible`` and ``enabled``."""

    __slots__ = ("_ui", "handle", "kind")

    def __init__(self, ui: "UI", handle: int, kind: str):
        self._ui = ui
        self.handle = handle
        self.kind = kind

    def add(self, kind: str, **props) -> "Node":
        """Append a child of any nui kind; ``on_<event>`` keywords become callbacks."""
        handlers = {name: props.pop(name) for name in list(props) if name in _EVENTS}
        node = Node(self._ui, self._ui._native.nui_add(self.handle, kind, _dump(props)), kind)
        for name, fn in handlers.items():
            if fn is not None:
                node.on(_EVENTS[name], fn)
        return node

    def set(self, **props) -> "Node":
        self._ui._native.nui_set(self.handle, _dump(props))
        return self

    @property
    def value(self):
        """The node's current value: a field's value, a toggle's state, a tree's selection."""
        return json.loads(self._ui._native.nui_get(self.handle))

    @value.setter
    def value(self, value) -> None:
        self.set(value=value)

    def on(self, event: str, fn: Callable) -> "Node":
        """Call fn(value) on each ``event`` of this node; fn may also take no argument."""
        self._ui._listen(self.handle, event, _adapt(fn))
        return self

    def clear(self) -> None:
        """Remove the children."""
        self._ui._native.nui_clear(self.handle)

    def remove(self) -> None:
        """Remove this node and its subtree; on a panel's root, the panel."""
        self._ui._native.nui_remove(self.handle)
        self._ui._handlers.pop(self.handle, None)

    # ---- containers

    def vstack(self, spacing: Optional[float] = None, **props) -> "Node":
        return self.add("vstack", spacing=spacing, **props)

    def hstack(self, spacing: Optional[float] = None, **props) -> "Node":
        return self.add("hstack", spacing=spacing, **props)

    def grid(self, columns: int = 2, **props) -> "Node":
        return self.add("grid", columns=columns, **props)

    def collapsible(self, title: str, open: bool = True, icon: Optional[str] = None, **props) -> "Node":
        return self.add("collapsible", title=title, open=open, icon=icon, **props)

    def tabs(self, **props) -> "Node":
        return self.add("tabs", **props)

    def tab(self, title: str, **props) -> "Node":
        return self.add("tab", title=title, **props)

    def scroll(self, height: Optional[float] = None, **props) -> "Node":
        return self.add("scroll", height=height, **props)

    def property_grid(self, schema: Optional[list] = None, **props) -> "Node":
        """Rows of labelled fields; add fields to it, or pass a schema of {kind, id, label, ...}."""
        return self.add("property_grid", schema=schema, **props)

    # ---- display

    def label(self, text: str = "", tone: Optional[str] = None, style: Optional[str] = None, **props) -> "Node":
        """tone: normal muted dim accent ok warn error info; style: body strong label mono hero."""
        return self.add("label", text=text, tone=tone, style=style, **props)

    def stat(self, label: str, value: str = "", tone: Optional[str] = None, **props) -> "Node":
        return self.add("stat", label=label, value=value, tone=tone, **props)

    def badge(self, text: str, tone: Optional[str] = None, icon: Optional[str] = None, **props) -> "Node":
        return self.add("badge", text=text, tone=tone, icon=icon, **props)

    def status(self, label: str, tone: Optional[str] = None, pulse: Optional[bool] = None, **props) -> "Node":
        return self.add("status", label=label, tone=tone, pulse=pulse, **props)

    def progress(self, value: float = 0.0, label: Optional[str] = None, **props) -> "Node":
        return self.add("progress", value=value, label=label, **props)

    def separator(self, label: Optional[str] = None, **props) -> "Node":
        return self.add("separator", label=label, **props)

    def spacer(self, size: Optional[float] = None, **props) -> "Node":
        return self.add("spacer", size=size, **props)

    def plot(self, series: Sequence[dict] = (), height: Optional[float] = None, sparkline: Optional[bool] = None,
             **props) -> "Node":
        """series: [{"name", "x": [...], "y": [...]}]; set(series=...) replaces them."""
        return self.add("plot", series=list(series), height=height, sparkline=sparkline, **props)

    def table(self, columns: Sequence[str], rows: Sequence[Sequence] = (), height: Optional[float] = None,
              **props) -> "Node":
        return self.add("table", columns=list(columns), rows=[list(r) for r in rows], height=height, **props)

    def tree(self, items: Sequence[dict] = (), on_select: Optional[Callable] = None, **props) -> "Node":
        """items: [{"id" (>0), "label", "parent" (0 = top level), "icon", "badge"}]; value is the selected id."""
        return self.add("tree", items=list(items), on_select=on_select, **props)

    def log(self, text: str = "", height: Optional[float] = None, **props) -> "Node":
        return self.add("log", text=text, height=height, **props)

    def code(self, text: str = "", height: Optional[float] = None, on_run: Optional[Callable] = None,
             **props) -> "Node":
        """A Python editor; Ctrl+Enter fires on_run with the text."""
        return self.add("code", text=text, height=height, on_run=on_run, **props)

    # ---- input

    def button(self, label: str, on_click: Optional[Callable] = None, icon: Optional[str] = None,
               action: Optional[str] = None, primary: Optional[bool] = None, **props) -> "Node":
        """A click runs on_click, or the editor action ``action`` with ``args``."""
        return self.add("button", label=label, on_click=on_click, icon=icon, action=action, primary=primary,
                        **props)

    def icon_button(self, icon: str, on_click: Optional[Callable] = None, action: Optional[str] = None,
                    **props) -> "Node":
        return self.add("icon_button", icon=icon, on_click=on_click, action=action, **props)

    def toggle(self, label: str, value: bool = False, on_change: Optional[Callable] = None, **props) -> "Node":
        return self.add("toggle", label=label, value=value, on_change=on_change, **props)

    def segmented(self, options: Sequence[str], value: int = 0, on_change: Optional[Callable] = None,
                  **props) -> "Node":
        return self.add("segmented", options=list(options), value=value, on_change=on_change, **props)

    def combo(self, label: str, options: Sequence[str], value: int = 0, on_change: Optional[Callable] = None,
              **props) -> "Node":
        return self.add("combo", label=label, options=list(options), value=value, on_change=on_change, **props)

    def float_field(self, label: str, value: float = 0.0, on_change: Optional[Callable] = None,
                    on_commit: Optional[Callable] = None, **props) -> "Node":
        """Also: speed, min, max, slider, format, unit. on_commit fires once per finished edit."""
        return self.add("float", label=label, value=value, on_change=on_change, on_commit=on_commit, **props)

    def int_field(self, label: str, value: int = 0, on_change: Optional[Callable] = None,
                  on_commit: Optional[Callable] = None, **props) -> "Node":
        return self.add("int", label=label, value=value, on_change=on_change, on_commit=on_commit, **props)

    def bool_field(self, label: str, value: bool = False, on_change: Optional[Callable] = None, **props) -> "Node":
        return self.add("bool", label=label, value=value, on_change=on_change, **props)

    def vec3_field(self, label: str, value: Sequence[float] = (0.0, 0.0, 0.0), on_change: Optional[Callable] = None,
                   on_commit: Optional[Callable] = None, **props) -> "Node":
        return self.add("vec3", label=label, value=list(value), on_change=on_change, on_commit=on_commit, **props)

    def color_field(self, label: str, value: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
                    on_change: Optional[Callable] = None, on_commit: Optional[Callable] = None, **props) -> "Node":
        """RGBA in [0, 1]; also alpha and hdr."""
        return self.add("color", label=label, value=list(value), on_change=on_change, on_commit=on_commit, **props)

    def text_field(self, label: str, value: str = "", on_commit: Optional[Callable] = None,
                   on_enter: Optional[Callable] = None, **props) -> "Node":
        return self.add("string", label=label, value=value, on_commit=on_commit, on_enter=on_enter, **props)

    def path_field(self, label: str, value: str = "", on_commit: Optional[Callable] = None,
                   on_browse: Optional[Callable] = None, **props) -> "Node":
        return self.add("path", label=label, value=value, on_commit=on_commit, on_browse=on_browse, **props)


class UI:
    """The editor shell's script-built panels, toasts and progress entries. An extension's
    UI builds through its own host table, so its panels go away with it."""

    def __init__(self, app, native=None):
        self._app = app
        self._own = native
        self._handlers = {}
        self._native.set_ui_handler(self._dispatch)

    @property
    def _native(self):
        # Through the app's weak proxy: holding the native app here would keep it alive past close().
        return self._own if self._own is not None else self._app._native

    def panel(self, panel_id: str, title: Optional[str] = None, dock: str = "right",
              icon: Optional[str] = None) -> Node:
        """A dockable panel ("left", "right", "bottom" or "float") listed under Window; the
        same id returns the existing panel. Its root stacks children vertically."""
        root = Node(self, self._native.nui_panel(panel_id, title or panel_id, dock), "panel")
        if icon is not None:
            root.set(icon=icon)
        return root

    def toast(self, text: str, level: str = "info") -> None:
        """A notification over the viewport; level: info, ok, warn or error."""
        self._native.toast(_LEVELS[level], text)

    def progress(self, entry_id: str, label: str, fraction: float) -> None:
        """Show or update a progress entry; fraction in [0, 1]."""
        self._native.progress(entry_id, label, fraction)

    def done(self, entry_id: str) -> None:
        """Remove a progress entry."""
        self._native.progress(entry_id, "", -1.0)

    def _listen(self, handle: int, event: str, fn: Callable) -> None:
        if handle not in self._handlers:
            self._native.nui_listen(handle)
            self._handlers[handle] = {}
        self._handlers[handle][event] = fn

    def _dispatch(self, handle: int, event_json: str) -> None:
        event = json.loads(event_json)
        fn = self._handlers.get(handle, {}).get(event["type"])
        if fn is not None:
            self._app._ui_call(lambda: fn(event["value"]))


def _dump(props: dict) -> str:
    return json.dumps({key: value for key, value in props.items() if value is not None})
