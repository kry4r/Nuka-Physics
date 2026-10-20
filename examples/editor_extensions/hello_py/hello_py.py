"""A sample Python editor extension: a panel, an action and axes drawn over the viewport."""

import math

from nuka.editor import Extension


class HelloPy(Extension):
    def on_startup(self, ctx):
        self.ctx = ctx
        self.count = 0
        self.axes = True
        self.greeting = ctx.settings.get("greeting", "Hello from Python")
        panel = ctx.ui.panel("hello_py", "Hello Python", dock="right", icon="file_py")
        panel.label("A Python extension: edit its files and it reloads.", tone="muted", wrap=True)
        self.counter = panel.stat("Greetings", "0")
        row = panel.hstack()
        row.button("Greet", icon="file_py", primary=True, on_click=self.greet)
        row.toggle("Axes", value=True, on_change=self.show_axes)
        ctx.actions.register("hello_py.greet", "Greet from Python", self.run_greet, category="tools", icon="file_py",
                             doc={"summary": "Shows the greeting as a toast and counts the greetings.", "kind": "view",
                                  "args": {"fail": "bool, fail on purpose, default false"},
                                  "returns": "{count, text}"})
        ctx.overlays.register("hello_py.axes", self.draw)
        ctx.log(f"started: {self.greeting}")

    def greet(self):
        self.count += 1
        self.counter.set(value=str(self.count))
        self.ctx.ui.toast(self.greeting, "ok")
        return {"count": self.count, "text": self.greeting}

    def run_greet(self, args):
        if args.get("fail"):
            raise RuntimeError("asked to fail")
        return self.greet()

    def show_axes(self, value):
        self.axes = bool(value)

    def draw(self, frame):
        if not self.axes:
            return
        origin = (0.0, 0.0, 0.0)
        frame.line(origin, (1.0, 0.0, 0.0), 0xFF5A5AFF, 2.5)
        frame.line(origin, (0.0, 1.0, 0.0), 0x5AFF7AFF, 2.5)
        frame.line(origin, (0.0, 0.0, 1.0), 0x5A9CFFFF, 2.5)
        bob = 0.05 * math.sin(frame.tick * 0.02)
        frame.circle((0.0, 0.0, 1.0 + bob), 6.0, 0xC6F432FF, 2.0)
        frame.text((1.05, 0.0, 0.0), 0xE8ECF0FF, "hello_py")

    def on_shutdown(self):
        self.ctx.log("stopped")
