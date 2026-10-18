"""A teleop device the smoke sets through an action, and a mode that maps the quantity "reach"
onto the elbow and the shoulder pitch."""

from nuka.editor import Extension


class TeleopProbe(Extension):
    def on_startup(self, ctx):
        self.axes = [0.0] * 6
        self.buttons = [False, False]
        self.fail = False
        self.last = None
        ctx.teleop.register_device("probe", "Probe Puck", self.read, axes=("tx", "ty", "tz", "rx", "ry", "rz"),
                                   buttons=("left", "right"), kind="six-axis")
        ctx.teleop.register_retarget("reach", "Reach", self.reach)
        ctx.actions.register("teleop_probe.set", "Set Probe", self.set, category="tools")
        ctx.actions.register("teleop_probe.last", "Last Probe State", lambda args: self.last or {}, category="tools")

    def read(self):
        return self.axes, self.buttons

    def set(self, args):
        self.axes = [float(x) for x in args.get("axes", self.axes)]
        self.buttons = [bool(x) for x in args.get("buttons", self.buttons)]
        self.fail = bool(args.get("fail", self.fail))
        return {"axes": self.axes, "buttons": self.buttons, "fail": self.fail}

    def reach(self, state):
        self.last = state
        if self.fail:
            raise RuntimeError("asked to fail")
        r = state["quantities"].get("reach", 0.0)
        out = {"elbow": -r, "shoulder_pitch": 0.5 * r}
        if state["device"]["buttons"].get("right"):
            out["nope"] = 0.0
        return out
