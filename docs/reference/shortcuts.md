# Keyboard and mouse

Shortcuts don't fire while a text field has the keyboard. The [command palette](../editor/index.md#the-command-palette)
lists every action with its shortcut, extensions' included.

## Everywhere

| Keys | Does |
| --- | --- |
| ++ctrl+n++ | New Scene |
| ++ctrl+o++ | Open Scene... |
| ++ctrl+i++ | Import Asset... |
| ++ctrl+s++ | Save |
| ++ctrl+shift+s++ | Save As... |
| ++ctrl+q++ | Exit |
| ++ctrl+z++ | Undo |
| ++ctrl+y++, ++ctrl+shift+z++ | Redo |
| ++delete++ | Delete the selection |
| ++space++ | Play / Pause |
| ++f5++ | Run the active script tab |
| ++shift+f5++ | Stop the running script |
| ++ctrl+p++, ++ctrl+shift+p++ | Command palette |
| ++ctrl+equal++, ++ctrl+minus++, ++ctrl+0++ | Larger UI, smaller UI, reset the UI scale |
| ++ctrl+shift+l++ | Switch the language |

## In the viewport

These act while the pointer is over the viewport.

| Input | Does |
| --- | --- |
| Left drag | Orbit around the target |
| Middle drag, or ++shift++ + left drag | Pan |
| Wheel | Zoom toward the target |
| ++w++ ++a++ ++s++ ++d++, ++q++ ++e++ | Fly forward, left, back, right, down, up; ++shift++ is four times faster |
| ++f++ | Frame the selection |
| ++home++ | Frame everything |
| ++shift+space++ | Show the active pane alone, and back |
| ++h++ | Show or hide the selection |
| Click | Select what is under the pointer |
| ++ctrl++ + left drag on a body | Pull it with a spring, as the Perturb tool does |

See [Viewport and camera](../editor/viewport.md) and [Pushing things around](../editor/simulation.md#pushing-things-around).

## Teleop from the keyboard

While teleop drives a robot from the **Keyboard** device, these keys are a gamepad and don't fly the
camera ([Teleop](../editor/motion.md#teleop)):

| Keys | Gamepad input | Velocity command mode |
| --- | --- | --- |
| ++w++ / ++s++ | Left stick forward / back | `vx` |
| ++a++ / ++d++ | Left stick left / right | `vy` |
| ++q++ / ++e++ | Right stick left / right | `wyaw` |
| ++page-up++ / ++page-down++ | Right stick up / down | |
| ++minus++ / ++equal++ | Left / right shoulder | |

## In panels

| Where | Keys | Does |
| --- | --- | --- |
| Stage tree | ++ctrl++ + click, ++shift++ + click, double-click | Add or remove a node, select a range, frame the node |
| Script editor | ++f5++, ++ctrl+enter++, ++shift+f5++, ++ctrl+s++ | Run, run the selection or line, stop, save ([Scripting](../editor/scripting.md)) |
| Console prompt | ++enter++, ++tab++, ++up++ / ++down++ | Run the line, complete, walk the history |
| Command palette | ++up++ / ++down++, ++enter++, ++esc++ | Move, run the highlighted action, close |
| Clip Editor | ++delete++, ++ctrl+c++ / ++ctrl+v++, ++ctrl+a++, ++ctrl+z++ / ++ctrl+y++ | Keys: delete, copy and paste, select all, undo and redo ([Clips](../editor/motion.md#clips)) |
| Packages field (import) | ++enter++ | Check the paths again |
