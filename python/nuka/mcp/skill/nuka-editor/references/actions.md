# Editor actions by group

Every editor feature is one of these actions. Run one with `action_run {id, args}`;
`action_help {id}` gives its whole entry: arguments with types and defaults, what it returns and an
example. `?` marks an optional argument. The kind in brackets is what the action touches, and so what
the user's access mode lets through (see safety.md).

This page is generated from assets/editor/actions.toml by tools/skill_actions.py.

## Documents: new, open, import, save, export

- `file.new` {template?, discard?} [file]: Replaces the document with a new untitled scene from a template (assets/editor/templates).
- `file.open` {path?, discard?} [file]: Opens a scene file as the document: .nks opens in place, other formats (URDF, MJCF, USD, meshes) import into an untitled document.
- `file.open_recent` {path, discard?} [file]: Opens a file from the recent list; the same as file.open.
- `file.forget_recent` {path} [app]: Removes a file from the recent list on the start page.
- `file.open_window` {path?, empty?} [process]: Starts another editor in a window of its own, on `path` or on an empty document; it shares this editor's profile.
- `file.revert` [file]: Reloads the document from its file, dropping unsaved changes; enabled only for a saved document with changes.
- `file.close` {discard?} [file]: Closes the document and leaves the editor on the start page.
- `file.recover` {path, discard?} [file]: Opens an autosave left by an editor that did not close cleanly, as the document it belonged to with its changes unsaved.
- `file.autosave` [file]: Writes the autosave of the document now.
- `file.unload` [file]: Unloads the scene from the world, keeping the editor open with no scene.
- `file.save` {path?} [file]: Saves the document to its file, or to `path`; an untitled document asks where with a file dialog.
- `file.save_as` {path?} [file]: Saves the document to a new file, which becomes its file; without `path` a file dialog asks.
- `file.package` {path?} [file]: Saves a copy at `path` that carries every image the scene uses in its own .nka, so it opens with no other files; the document keeps its path.
- `file.export_usd` {path?, node?, light_units?} [file]: Writes the scene, or the Stage node `node` with what hangs from it, as UsdPhysics in the format the path's extension names (.usd, .usda, .usdc, .usdz).
- `file.import_analyze` {path, profile?, fix_base?, package_paths?, light_units?} [read]: Reads an asset file (URDF, MJCF, USD, mesh, NKS) without importing it: whether it imports, its diagnostics and its size.
- `file.import` {path?, mode?, name?, pose?, ground?, profile?, fix_base?, package_paths?, light_units?, discard?} [edit]: Imports an asset file (URDF, MJCF, USD, OBJ/STL/GLB meshes, NKS).

## Edits and undo

- `edit.undo` [edit]: Undoes the last edit of the document.
- `edit.redo` [edit]: Redoes the last undone edit of the document.
- `edit.apply` {label?, ops} [edit]: Applies one edit transaction: its ops change fields of Stage nodes, all as one undoable step.
- `edit.rename` {name, path?} [edit]: Renames the node at `path`, or the selection.
- `edit.delete` {paths?, path?} [edit]: Deletes `paths`, `path` or the selection in one step; a node under another goes with it.
- `edit.reparent` {parent, path?} [edit]: Moves the node at `path`, or the selection, under another node.

## Scene: add, collision, nodes

- `scene.spawn` {kind?, pose?} [edit]: Adds a primitive to the scene.
- `scene.select` {path?, mode?, paths} [view]: Selects Stage nodes: replace makes `path` the selection (none clears it), add and toggle grow or shrink the selection, range takes `paths` with `path` leading.
- `scene.set_visible` {value?, paths?, path?} [view]: Shows or hides Stage nodes in the viewport; a hidden node keeps simulating.
- `scene.set_locked` {value?, paths?, path?} [view]: Locks or unlocks Stage nodes against picking and gizmo drags in the viewport.
- `scene.collision_preview` {paths?, path?, method?, max_pieces?, replace?, only_missing?, subtree?} [read]: What scene.generate_collision would do with the same arguments, body by body, without changing anything; the Generate Collision dialog shows it.
- `scene.generate_collision` {paths?, path?, method?, max_pieces?, replace?, only_missing?, subtree?} [edit]: Builds colliders for bodies from their visual meshes, as one undoable edit: a convex hull, a convex decomposition, an SDF or the triangle mesh itself.
- `scene.collision_dialog` [view]: Opens the Generate Collision dialog on the selection with its preview.
- `scene.terrain` {kind, size?, center?, cell?, base?, height?, bump?, grade?, rise?, width?, platform?, count?, tile?, margin?, seed?} [edit]: Sets the scene's terrain field, which the cook bakes into a static heightfield, or removes it with kind none.
- `scene.terrain_info` [read]: The scene's terrain field as scene.terrain arguments, every size filled in; null when the scene has none.
- `scene.terrain_dialog` [view]: Opens the Terrain dialog on the scene's field, or on the default bump field when there is none.
- `scene.nodes` {under?, kind?} [read]: The scene's nodes in tree order: all, those under `under`, or those of one `kind`.
- `scene.fields` {paths?, path?} [read]: The fields the Properties panel edits for `paths` (else `path`, else the selection; "" or no selection gives the world options), in groups: each group's `targets` are what an edit.apply op names as its target, each field's `field` the op's field.
- `scene.add_box` [edit]: Adds a box above the selection, else above the origin, as one undoable edit; scene.spawn with kind box.
- `scene.add_sphere` [edit]: Adds a sphere above the selection, else above the origin, as one undoable edit; scene.spawn with kind sphere.
- `scene.add_capsule` [edit]: Adds a capsule above the selection, else above the origin, as one undoable edit; scene.spawn with kind capsule.
- `scene.add_cylinder` [edit]: Adds a cylinder above the selection, else above the origin, as one undoable edit; scene.spawn with kind cylinder.
- `scene.add_plane` [edit]: Adds a plane as one undoable edit; scene.spawn with kind plane.
- `scene.expand_all` [view]: Expands every node in the Stage panel.
- `scene.collapse_all` [view]: Collapses every node in the Stage panel.

## Assets

- `asset.sources` [read]: The files the document's nodes were imported from, with the options each import used.
- `asset.reimport` {path?, discard?} [edit]: Imports the source of the node at `path` (the selection by default) again in its place, as one edit to undo; the node "" is an asset opened whole, which opens again as a new document.
- `asset.show_source` {path?} [process]: Shows the source file of the node at `path` (the selection by default) in the system's file browser.
- `assets.add_folder` {path?} [app]: Adds a folder to the Assets panel's roots, saved in the profile; without `path` a folder dialog asks.
- `assets.remove_folder` {path} [app]: Removes a folder from the Assets panel's roots.
- `assets.rescan` [view]: Scans the Assets panel's folders again for new and removed files.
- `assets.page` {page} [view]: Shows the Assets panel's `page`: files or materials.
- `assets.list` {pictures?} [read]: The asset cards as the Assets panel lists them: the bundled examples, the document's folder and the user's folders; `pictures` asks for their thumbnails as the panel does.

## Materials

- `material.list` [read]: The scene's materials: each one's fields as material.set takes them, its friction and how many shapes use it.
- `material.get` {material?, path?} [read]: One material: `material` by name, else the one the drawn parts at `path` (else the Material section's part) share.
- `material.slots` {path?} [read]: The parts the world draws at or under `path` (else the selection) and their materials, "" for the default one.
- `material.create` {name?, from?, preset?, fields?, assign?} [edit]: Makes a material as one undoable edit: a copy of `from`, else `preset`'s look, else the default, with `fields` over it.
- `material.set` {material?, path?, base_color?, metallic?, roughness?, emissive?, albedo_map?, roughness_map?, metallic_map?, normal_map?, emissive_map?, uv_scale?, triplanar?, friction?} [edit]: Sets fields of a material (`material`, else the one at `path`) as one undoable edit.
- `material.choose_map` {map, material?, path?} [edit]: Picks an image for a material's `map` slot in a file dialog.
- `material.rename` {name, material?, path?} [edit]: Renames a material (`material`, else the one at `path`) as one undoable edit.
- `material.delete` {material?, path?, replace?} [edit]: Deletes a material (`material`, else the one at `path`) as one undoable edit; the shapes that used it take `replace`, which a material in use needs.
- `material.assign` {material?, path?, paths?} [edit]: Every shape at or under `paths`, else `path`, else the selection, takes `material` ("" for the default one), as one undoable edit.
- `material.make_unique` {path?} [edit]: The part at `path` (else the Material section's) takes a copy of its material of its own, as one undoable edit.
- `material.select_users` {material} [view]: Selects the shapes that draw with `material`, and lists them.

## Robots: joints, poses, gains, identification, IK

- `robot.joints` [read]: The moving joints of the scene's robots with their limits and drives, and the shown environment's live values; the roots and whether the base floats.
- `robot.set_targets` {targets} [sim]: Sets drive targets for the shown environment by joint name, in rad or m; the next step reads them.
- `robot.reset_targets` [sim]: Puts every joint's drive target back at its initial position.
- `robot.radians` {on?} [view]: Revolute joints read in radians while on, else degrees, in the Joints panel; without `on` it toggles.
- `robot.watch` {on?} [sim]: While on, every step samples the shown environment's joints for robot.samples and the Joints panel's curves; without `on` it toggles.
- `robot.samples` {joint, count?} [read]: The newest `count` samples of one joint while robot.watch is on, oldest first; `total` counts the watch's samples.
- `robot.drive` {type?, kp?, joints?} [edit]: One undoable edit gives `joints` (all by default) a drive of `type`; a position drive takes kp.
- `robot.fix_base` {on?} [edit]: Static roots are fixed bases: one undoable edit makes every robot root static (fixed to the world) or free; without `on` it toggles.
- `robot.poses` [read]: The joint poses kept with the document, each a name and its joints' positions (rad or m).
- `robot.pose_capture` {name?} [edit]: Keeps the shown environment's joint positions as pose `name`, replacing a pose of that name; without one it is "Pose N".
- `robot.pose_apply` {name, duration?} [sim]: Blends the shown environment's drive targets to pose `name` over `duration` simulated seconds, 0 at once; joints the pose lacks keep their targets.
- `robot.pose_initial` {name} [edit]: Writes pose `name` into its joints' initial positions as one undoable edit; sim.reset starts there.
- `robot.pose_rename` {name, to} [edit]: Renames a pose.
- `robot.pose_remove` {name} [edit]: Removes a pose.
- `robot.poses_save` {path?} [file]: Writes the poses to a JSON file as {"poses": [{name, joints}]}; without `path` a file dialog asks.
- `robot.poses_load` {path?} [file]: Adds the poses of a JSON file written by robot.poses_save, replacing those of the same name; without `path` a file dialog asks.
- `robot.gain_test` {joint?, kind?, amplitude?, frequency?, duration?} [sim]: Drives one joint's target through a step or a sine from where it stands, in a copy of the world the test throws away, and fits rise, overshoot, settling, damping ratio and more.
- `robot.gain_status` [read]: Whether a gain test runs, how many runs the tuner keeps and the newest one's figures.
- `robot.gain_runs` {trace?} [read]: The gain test runs the tuner keeps, oldest first; `trace` adds each one's samples.
- `robot.gain_clear` [view]: Forgets the gain test runs the tuner keeps.
- `robot.gain_tuner` {on?} [view]: Opens or closes the Gain Tuner window; `on` sets it, else it toggles.
- `sysid.run` {episode?, params?, generations?, population?, sigma?, seed?} [sim]: Identifies joint damping, armature, friction loss or a link mass scale from a demonstration episode: CMA-ES searches each quantity between its min and max, and every candidate replays the episode's drive targets from its first step in a copy of the world, scored by the mean squared error against the joint positions it recorded.
- `sysid.status` [read]: The identification's progress and its best fit so far: each quantity's mean value before the search and the best found; the loss is the mean squared joint error in rad² or m², and the curve holds the best loss after each generation.
- `sysid.stop` [sim]: Ends the running identification with the best values found so far.
- `sysid.apply` [edit]: Writes the finished identification's best values to its joints and links in one undoable edit; a mass scale multiplies each link's mass from before the search.
- `sysid.panel` {on?} [view]: Opens or closes the System Identification window; `on` sets it, else it toggles.
- `robot.links` [read]: The shown environment's robot links in link order, each with its parent link, incoming joint and mass, and where it and its centre of mass are; the robot's total mass, centre of mass and degrees of freedom.
- `robot.asset_links` {path?, profile?} [read]: Reads an asset file without importing it and lists its links, its roots first: the mounts robot.attach_asset takes.
- `robot.attach` {child, parent, joint?, axis?, frame?, mount?, joint_name?} [edit]: Hangs the body `child`, and what its joints carry, off the link `parent` through a new joint, as one undoable edit.
- `robot.attach_asset` {path, parent, name?, joint?, axis?, frame?, mount?, joint_name?, profile?} [edit]: Adds the asset at `path` and hangs its link `mount` (default its root) off the link `parent` in one undoable edit; without `frame` the mount takes the parent's frame.
- `robot.assembler` {on?} [view]: Opens or closes the Robot Assembler; opened on a selected link, that link becomes the parent.
- `robot.ik` {link, pos, quat?, rot?, position_only?, pins?, root?, rotation_weight?, apply?} [sim]: Moves `link` (node path or joint name) to world `pos`, and `quat` or XYZ-degree `rot`, keeping `pins` put.
- `robot.ik_settings` {root?, pins?, lock_orientation?, rotation_weight?} [view]: The gizmo's IK: the chain root (empty for the articulation's), the links a drag pins, whether a move locks the orientation and the rotation weight [m/rad]; an argument given replaces its setting.

## Motion: clips, demonstrations, teleoperation, policies

- `teleop.devices` [read]: The devices teleop reads (keyboard, gamepads, the virtual device, extension devices) and the modes it maps them through, with the active ones.
- `teleop.use` {on?, device?, mode?, joint?, link?, hold?, kp?, kd?, command_base?} [sim]: Switches teleop on or off and picks the device, the mode, the joint mode's joint, the ik mode's link ("" follows the selection), PD hold and the command baseline; bare, it toggles.
- `teleop.input` {axes?, buttons?} [sim]: Sets the virtual device's axes ([-1, 1]) and buttons by name; the others keep their values.
- `teleop.map` {mode?, bindings?, reset?, deadzone?} [sim]: A mode's bindings [{input, quantity, gain, invert}]: given ones replace them and `reset` restores the defaults; `deadzone` in [0, 1) is shared by every axis.
- `teleop.pad` {attach?, id, axes?, buttons?, detach?} [sim]: A virtual gamepad on SDL's joystick path: `attach` names a new one; `id` with `axes` and `buttons` sets its inputs, and with `detach` removes it.
- `teleop.status` [read]: The active device's reading, the mode's quantities and what they drove last frame.
- `policy.load` {path?, host?} [process]: Attaches an NKML locomotion policy to the loaded robots in place of the one before, and sends it teleop's command; the dialog asks without `path`, `host` runs the host MLP.
- `policy.unload` [sim]: Detaches the loaded policy; the drive targets hold where it left them.
- `policy.enable` {on?} [sim]: Runs or stops the loaded policy; stopped, the drive targets hold where it left them.
- `policy.status` [read]: The loaded policy, its contract, the joints it drives and the command it reads.
- `policy.arrows` {on?} [view]: Shows or hides the arrows of the command and the measured velocity over the policy's robots.
- `demo.record` {on?, all?} [sim]: Records the shown environment, with `all` every displayed one, each step at the head: drive targets, joint states and the teleop device's axes.
- `demo.episodes` [read]: The recorded episodes, the recording and the replay in progress.
- `demo.play` {episode?, play?} [sim]: Writes an episode's recorded drive targets into its environment a row a step from the state now, the controller waiting meanwhile; `play` (the default) starts the transport.
- `demo.stop` [sim]: Stops a demonstration replay.
- `demo.save` {path?} [file]: The episodes as robomimic-style HDF5 through nuka.editor.demos; the dialog asks without `path`.
- `demo.load` {path?} [file]: Replaces the episodes with an HDF5 file's, its joints matched by name to the scene's; the dialog asks without `path`.
- `demo.discard` {episode?} [sim]: Drops episode `episode`, without it all of them.
- `clip.list` [read]: The joint clips saved beside the document, the edited one with its playhead and selection, and the clip playing.
- `clip.get` {clip?} [read]: A clip's tracks and keys, the edited clip's without `clip`.
- `clip.new` {name?, duration?, loop?} [edit]: A clip of `duration` seconds (1 by default) named `name`, else "Clip N"; the editor edits it.
- `clip.set` {clip?, duration?, loop?} [edit]: Sets a clip's `duration` or `loop`; keys past a shorter end stay but no longer play.
- `clip.rename` {clip?, to} [edit]: Renames a clip.
- `clip.remove` {clip?} [edit]: Removes a clip.
- `clip.key` {clip?, joint, time?, value?, interp?, in?, out?, keys?} [edit]: Sets one key from the arguments or each of `keys`, replacing any at their times.
- `clip.key_pose` {clip?, time?, pose?, from?} [edit]: Keys every joint at `time` (the playhead) from library pose `pose`, else from the shown environment's positions, or its drive targets with `from` "target".
- `clip.remove_keys` {clip?, keys?, joint?, time?} [edit]: Removes `keys` [{joint, time}], else every key of `joint` and/or at `time`, else the selection.
- `clip.move_keys` {clip?, keys?, dt?, dv?} [edit]: Shifts `keys` [{joint, time}], else the selection, by `dt` seconds and `dv` in value; keys stay within the clip and replace those where they land.
- `clip.sample` {clip?, time?, apply?} [read]: Each keyed joint's value at `time` (the playhead); with `apply` they become the shown environment's drive targets at once.
- `clip.play` {clip?, loop?, play?} [sim]: Drives the shown environment through a clip a step a row from now, over and over with `loop` (the clip's own by default); `play` (the default) starts the transport.
- `clip.stop` [sim]: Stops the clip playing.
- `clip.save` {clip?, path?} [file]: Writes a clip as {"format": "nkclip", "version": 1, "clip": ...}; the dialog asks without `path`.
- `clip.load` {path?} [file]: Adds a clip file's clip, replacing one of its name; the editor edits it.
- `clip.editor` {on?, clip?, playhead?, auto_key?, interp?, curve?, selected?} [view]: Opens or closes the Clip Editor (bare, it toggles) and sets what it shows: the `clip`, the `playhead`, `auto_key`, the `interp` of new keys, the `curve` joint and the `selected` keys.
- `clip.undo` [edit]: Steps back through the clip edits; scene edits keep their own history.
- `clip.redo` [edit]: Steps forward again through undone clip edits.

## Simulation, timeline, solver and Rerun

- `sim.play` [sim]: Starts the simulation.
- `sim.pause` [sim]: Pauses the simulation.
- `sim.toggle_play` [sim]: Plays the simulation when it is paused, else pauses it.
- `sim.step` {steps?} [sim]: Advances the paused simulation by `steps` physics steps.
- `sim.reset` [sim]: Resets every environment to the scene's initial state at tick 0 and pauses; an active randomization draws again (with the next seed unless the draw stays).
- `sim.speed` {value?} [sim]: Sets the playback speed: 1 is real time, below 1 slow motion, above 1 faster when the machine keeps up.
- `sim.envs` {count?} [sim]: {count} rebuilds the world with that many environments, all from the scene's initial state at tick 0; loads and edits keep the count.
- `sim.perturb` {path, force?, torque?, point?, target?, gain?} [sim]: A world force and torque at a world point (the COM by default), plus a spring toward `target`; no load and no target releases the body.
- `sim.grab_particle` {particle, target?} [sim]: Holds one environment particle (cloth or MPM) kinematic at `target`, where it is by default.
- `sim.release` {path?, particle?} [sim]: Lets go of the load on `path`, the held `particle`, or with neither every load.
- `sim.loads` [read]: The loads held on the shown environment as its last frame has them.
- `solver.telemetry` {mask?, alert?} [sim]: Records solver telemetry each step: per-step residuals, sweep counts and capped steps.
- `solver.heatmap` {on?} [view]: Colours the solver's per-vertex residuals over the scene; the heatmap needs the per-vertex channel and turns it on.
- `solver.set_iterations` {n} [sim]: Sets the solver's sweep budget per step, live.
- `sim.dt` {seconds?} [sim]: {seconds} rebuilds the world to step that often, its state carried over; without it, the current step.
- `solver.status` [read]: The solver telemetry's state: channels, alert, sweep budget, heatmap, records kept and capped, the newest record and the located marker.
- `solver.records` {since?, limit?, capped?} [read]: Pages forward through the kept telemetry records: `since` on, at most `limit`, oldest first.
- `solver.locate` {record?, clear?} [view]: Frames and selects the worst element of a kept record: the given one, else the newest capped one, else the newest; {"clear": true} drops the marker.
- `timeline.status` [read]: The timeline: the playhead tick, the start and head of the current branch, whether it replays, every branch, the keyframes, the ghost, the keyframe settings and memory, and a running or finished video export.
- `timeline.seek` {tick?, delta?, to?} [sim]: Moves the playhead and pauses: {tick}, {delta} ticks from the playhead, or {to: "start" | "head"}.
- `timeline.fork` {label?} [sim]: Starts a new branch at the playhead: what runs from here on is kept beside the old history.
- `timeline.branch` {id} [sim]: Switches to another branch and pauses at its playhead.
- `timeline.rename` {id?, label} [sim]: Renames a branch.
- `timeline.ghost` {on?, branch?} [view]: {on, branch}: draws another branch's bodies as a ghost over this one at the same tick; without `on` it toggles; `branch` picks what it shows, else the last parent.
- `timeline.configure` {interval?, budget_mb?} [sim]: Sets how often the timeline keeps a keyframe and how much memory its history may use.
- `timeline.clear` [sim]: Drops the recorded history and branches, keeping the world where it is.
- `timeline.hash` [read]: The world's state hash once every command queued before it applied; equal hashes mean bit-identical states.
- `timeline.save` {path?} [file]: The current branch's line of history with what it plays in, as a .nkrec; the dialog asks without `path`.
- `timeline.load` {path?} [file]: A recording of this document and world layout becomes the history, back where it was saved; the dialog asks without `path`.
- `timeline.export` {path?, fps?, from?, to?} [file]: An mp4 of ticks `from` to `to` (the start and head) at `fps` frames a simulated second, drawn by the viewport and encoded by ffmpeg from the PATH; timeline.status reports its progress.
- `rerun.stream` {on?, every?, file?, addr?} [process]: Logs the shown environment's link and body poses to Rerun every `every` ticks at the head, named as nuka.editor.nkrec names them (world/links/<node>, world/bodies/<node>) over the `tick` and `sim_time` timelines: into the .rrd `file`, else to the viewer at `addr`, else to a new local viewer.

## Sensors, labels, datasets and randomization

- `sensor.list` [read]: The scene's sensors with their mounts and settings.
- `sensor.add` {type, mount?, name?, position?, rotation_deg?, rate_hz?, camera?, lidar?, tactile?} [edit]: Adds a sensor of `type` riding the body at `mount` (default the selected sensor's body, else the selected node's body; "" mounts a camera or lidar in the world) as one undoable edit, and selects it.
- `sensor.set` {index?, sensor?} [edit]: Edits a sensor (`index` or `sensor`, else the selected one) as one undoable edit: every other argument is a field, as sensor.add takes them, plus offset, update_period and aov_mask.
- `sensor.remove` {index?, sensor?} [edit]: Removes a sensor (`index` or `sensor`, else the selected one) as one undoable edit.
- `sensor.select` {index?, sensor?} [view]: Selects a sensor, which drops the node selection; `index` -1 selects none.
- `sensor.preview` {on?, index?, sensor?, channel?} [view]: The Sensors panel's live preview: `on`, the sensor it shows (`index` or `sensor`, else the selected one) and a camera's `channel`.
- `sensor.read` {index?, sensor?, channel?, env?, path?} [read]: Renders and reads a sensor of environment `env` now: a camera's `channel`, a lidar's ranges or a state sensor's values.
- `dr.get` [read]: The domain randomization of the document: the seed, whether a reset draws again, the render jitters, the physics ranges and the live parameters' factor ranges.
- `dr.set` {seed?, reset_draws?, render?, physics?, params?} [edit]: Merges `args` in the shape dr.get reports and draws again when anything randomizes or did; saved beside the document.
- `dr.sample` {seed?} [sim]: Draws the randomization again with `seed`, else the next one.
- `label.list` {path?} [read]: The semantic labels by node path and their classes; with `path`, the label that node has and the node it comes `from`, itself or an ancestor.
- `label.set` {label, paths?, path?} [edit]: Gives the nodes at `paths` (or `path`, else the selected ones) the semantic `label` that datasets write; "" clears theirs, so they inherit their parent's again.
- `dataset.export` {out?, frames?, every?, seed?, randomize?, channels?, cameras?, lidar?, state?, reset?} [file]: Steps `frames` frames of `every` ticks from a reset (unless `reset` is false), drawing the `randomize` kinds again for each from `seed` + frame, and writes the `cameras`' `channels`, the lidars and the state sensors under `out` (a folder dialog without it); dataset.status follows it.
- `dataset.status` [read]: The dataset export running or last finished: frames written of how many, where, and any error.
- `dataset.cancel` [sim]: Stops after the frame being read and writes the frames read so far.
- `dataset.exporter` {on?} [view]: Opens or closes the Export Dataset dialog; `on` sets it, else it toggles.

## Training

- `train.tasks` [read]: The training tasks (go2_velocity, Go2 velocity tracking, and any registered from Python): each one's groups of parameters, the robot its scene holds and the step it trains at.
- `train.config` {task?} [read]: The document's task and full configuration, the values that differ from the defaults (`changed`) and what keeps them from training the scene (`errors`); `task` switches to that task's defaults.
- `train.set` {values?, path?, value, task?} [edit]: Sets `values` ({"group.key": value}), or `path` to `value`, after switching to `task` if given; a value that does not fit fails the call and changes nothing.
- `train.validate` [read]: Whether the document's training configuration can train the scene as it stands.
- `train.python` {path?} [app]: The Python runs start in and whether it can train (torch, rl_games, a GPU); `path` sets it, "" returns to the editor's own.
- `train.start` {name?, checkpoint?} [process]: Trains the scene as it stands, with the document's configuration, in another process as run `name` (the time by default), from the rl_games `checkpoint` if given.
- `train.stop` {name?} [process]: Ends run `name` (else the one started last) after its current epoch, with a final checkpoint.
- `train.status` {name?, log?} [read]: A run's state (starting, running, done, failed, lost), epoch, rewards and checkpoint count, and the last `log` lines (20) of its output.
- `train.runs` [read]: The document's runs, newest first, each with its status.
- `train.metrics` {name?, since?} [read]: A run's epoch rows (reward, episode length, losses, timings) after the first `since`; `next` is the `since` of the following call.
- `train.checkpoints` {name?} [read]: A run's saves in order: each rl_games checkpoint, its NKML policy, kind (best, epoch, final), epoch and reward.
- `train.play` {name?, policy?, kind?} [process]: Loads a run's policy onto the scene's robots and plays: `policy` (a file), else the newest checkpoint of `kind`: "best" (the default), "final", "epoch" or "last".
- `train.observe` {on?, name?, envs?, follow?} [view]: `on: true` watches run `name` (else the one started last): the world holds its first `envs` (16) environments at the run's frames under arrows of their commanded and actual velocity, and `follow` keeps the camera on the current one.
- `train.reveal` {name?} [process]: Opens run `name`'s folder (else the one started last, else the newest) in the file browser.
- `train.panel` {page?, run?, compare?} [view]: Opens the Training panel at `page` (task, runs, dashboard or checkpoints) on run `run`, with the runs `compare` drawn over it on the Dashboard; reports what the panel shows.

## View and camera

- `view.frame_all` [view]: Points the camera so the whole shown world fits the viewport.
- `view.frame_selected` [view]: Points the camera at the selection with everything under it.
- `view.reset_camera` [view]: Puts the camera back at its default view of the world.
- `view.colliders` {on?} [view]: Draws collider proxies over the scene; `on` sets it, else it toggles.
- `view.contacts` {on?} [view]: Draws contact points and normals; `on` sets it, else it toggles.
- `view.hide_overlays` [view]: Turns off the collider and contact overlays.
- `view.shadows` {on?} [view]: Casts the sun's shadows in the Lit view; `on` sets it, else it toggles.
- `view.fog` {on?} [view]: Fades distant geometry into the sky in the Lit view; `on` sets it, else it toggles.
- `view.ground_grid` {mode?} [view]: The Lit view's ground grid: mode auto (under scenes without a floor), on or off; a bare call flips what is shown.
- `view.screenshot` {path?} [file]: Writes the next drawn frame, the whole window, as a PPM.
- `view.stats` [read]: The editor's frame timings: frame count, frames per second, frame and UI time percentiles, the physics tick, how snapshots reach the renderer, whether it plays, the view mode and the viewport's pixel rectangle.
- `view.mode` {mode?} [view]: The shading the viewport draws with; without a mode it reports the current one.
- `view.layout` {layout?} [view]: The panes: one, two side by side or four; without a layout it reports the current one.
- `view.layout_single` [view]: Shows one viewport pane.
- `view.layout_split` [view]: Shows two viewport panes side by side.
- `view.layout_quad` [view]: Shows four viewport panes.
- `view.maximize` {index?} [view]: Shows one pane alone, the active one without an index, and goes back to the layout it left.
- `view.view` {index?, mode?, camera?, activate?} [view]: A pane's mode and scene camera ("" is free), each kept when left out; activate makes it the active pane.
- `view.views` [read]: The viewport panes: the layout, the active pane, whether one is maximized, the scene cameras and each pane's mode and camera.
- `view.pick` {x?, y?, select?} [view]: What the viewport shows at x, y: fractions of its width and height from the top left; `select` clicks there, making the environment it hits current and selecting the body.
- `view.envs` {show?, current?, spacing?} [view]: The environment grid: `show` cells of the current environment's page (up to 16), `current` makes one current, `spacing` sets the metres between cells (0 sizes them from the robot).
- `view.camera` {target?, distance?, yaw?, pitch?} [view]: The active pane's orbit camera: `target`, `distance` and `yaw`, `pitch` in degrees, each kept when left out.
- `view.select_rect` {x0?, y0?, x1?, y1?, mode?} [view]: Selects what the viewport shows inside x0, y0 to x1, y1 in the same fractions as view.pick; add keeps the selection.
- `view.rt_preview` {on?, max_spp?} [view]: Path-traced frames of the view over the raster one, refined up to max_spp samples while nothing moves; `on` sets it, else it toggles.
- `view.rt_status` [read]: How far the RT preview has refined: samples so far, whether it converged and is shown, pending work, its size and any error.
- `view.rt_save` {path?} [file]: The RT image the viewport shows, as a PNG; the dialog asks without `path`.
- `view.render` {path?, width?, height?, spp?} [process]: A path-traced PNG of the view at {width, height, spp}, rendered by another process from a copy of the scene and its state; view.render_status reports it.
- `view.render_status` [read]: Whether a view.render job runs, how long it has, and the last finished one.
- `view.bookmarks` [read]: The document's camera bookmarks.
- `view.bookmark_add` {name?} [view]: Keeps the view as `name`, replacing a bookmark of that name; without one it is "View N".
- `view.bookmark_goto` {name} [view]: Moves the camera to a bookmark.
- `view.bookmark_rename` {name, to} [view]: Renames a camera bookmark.
- `view.bookmark_remove` {name} [view]: Removes a camera bookmark.
- `view.bookmarks_save` {path?} [file]: Writes the bookmarks in the format --cams reads; the dialog asks without `path`.
- `view.bookmarks_load` {path?} [file]: Replaces the bookmarks with a file's; the dialog asks without `path`.
- `view.robot_overlays` {on?, frames?, axes?, com?, inertia?, translucent?} [view]: The robot overlays: link frames, joint axes, centres of mass, inertia boxes and translucent links.
- `view.sensor_overlays` {on?, frustums?, fans?, frames?, labels?} [view]: The sensor overlays: camera frustums, lidar fans, sensor frames and labels.
- `view.ui_scale` {value} [view]: Sets the UI scale, 0.5 to 3, on top of the display's.
- `view.ui_scale_up` [view]: Makes the UI 10% larger, up to 3.
- `view.ui_scale_down` [view]: Makes the UI 10% smaller, down to 0.5.
- `view.ui_scale_reset` [view]: Returns the UI scale to 1.
- `view.toggle_language` [view]: Switches the UI to the next language with a string table, English first.
- `view.language` {locale} [view]: Shows the UI in `locale`: en, or one with a string table (zh-CN ships with the editor).
- `view.missing_strings` [read]: UI strings the current language has no translation for, as shown so far.

## Windows, panels and tools

- `window.minimize` [app]: Minimizes the editor's window.
- `window.maximize` [app]: Maximizes the editor's window, or restores it when it is maximized.
- `window.restore` [app]: Restores the editor's window from minimized or maximized.
- `window.state` [read]: The editor window's state: maximized, minimized, focused, position and size in pixels, and the title bar's height and drag spans.
- `tool.perturb` {on?} [view]: Makes Perturb the viewport tool (dragging a body pulls it with a spring), or goes back to Select; `on` sets it.
- `tool.pin` {on?} [view]: Makes Pin the viewport tool (clicking a body holds it where it is), or goes back to Select; `on` sets it.
- `tool.measure` {a?, b?, at?, on?} [view]: `a` and `b` set the Measure tool's ends (a link's node path, a joint's name, or "ground" for b) and return the measurement; `at` takes link ends at their origin or com.
- `tool.measure_clear` [view]: Clears the Measure tool's ends.
- `tool.ik` {on?} [view]: While on, the gizmo on a jointed link poses the robot through IK; without `on` it toggles.
- `window.layout_authoring` [view]: Arranges the panels for building a scene: Stage and File left, Properties and Camera right, Assets, Console, Script, Stats, Timeline and Sensors below.
- `window.layout_debug` [view]: Arranges the panels for debugging the solver: Stage left, Solver, Debug, Stats and Properties right, Console, Timeline, Script and Assets below.
- `window.layout_robot` [view]: Arranges the panels for a robot: Stage left, Joints, Poses, Policy, Demos and Properties right, Console, Timeline, Stats, Assets and Sensors below.
- `window.layout_data` [view]: Arranges the panels for making data: File and Stage left, Sensors, Randomize, Stats, Camera and Properties right, Assets, Console, Script and Timeline below.
- `window.layout_training` [view]: Arranges the panels for training: Stats and Stage left, a wide Training, Policy and Camera column right, Console, Script, Assets and Timeline below.
- `window.reset_layout` [view]: Puts the panels of the current layout back where it places them.
- `window.focus` {panel} [view]: Opens a panel and brings it to the front.
- `help.about` [view]: Shows the About window: the nui API and Dear ImGui versions and the open-source credits.
- `ui.toast` {text, level?} [view]: Shows a short notice in the corner of the editor.
- `window.toggle_stage` [view]: Opens or closes the Stage panel: the scene tree, to select, rename, reparent and spawn.
- `window.toggle_property` [view]: Opens or closes the Properties panel: transform, material and physics fields of the selection.
- `window.toggle_file` [view]: Opens or closes the File panel: open, save and load scenes.
- `window.toggle_assets` [view]: Opens or closes the Assets panel: importable files to drop into the scene or open.
- `window.toggle_stats` [view]: Opens or closes the Stats panel: frame and physics timings.
- `window.toggle_debug` [view]: Opens or closes the Debug Draw panel: collider and contact overlays.
- `window.toggle_solver` [view]: Opens or closes the Solver panel: solver residuals, sweep budget alerts and the residual heatmap.
- `window.toggle_timeline` [view]: Opens or closes the Timeline panel: seek, replay and fork the run, and compare branches with a ghost.
- `window.toggle_camera` [view]: Opens or closes the Camera panel: camera pose, framing and controls.
- `window.toggle_joints` [view]: Opens or closes the Joints panel: joint states, drive targets and gains, the fixed base and teleop.
- `window.toggle_poses` [view]: Opens or closes the Poses panel: named joint poses to blend to, start from, import and export.
- `window.toggle_policy` [view]: Opens or closes the Policy panel: load a locomotion policy, run or stop it and watch its actions.
- `window.toggle_demos` [view]: Opens or closes the Demos panel: record teleoperated demonstrations, replay them and save them as HDF5.
- `window.toggle_sensors` [view]: Opens or closes the Sensors panel: preview each sensor's images, ranges and readings as the simulation runs.
- `window.toggle_randomize` [view]: Opens or closes the Randomize panel: randomize looks, masses, gravity and parameters from a seed, again on reset.
- `window.toggle_training` [view]: Opens or closes the Training panel: configure a task, start and watch runs, plot them and play their saves.
- `window.toggle_script` [view]: Opens or closes the Script panel: the Python editor.
- `window.toggle_console` [view]: Opens or closes the Console panel: script output and a Python prompt.
- `window.toggle_nui` [view]: Opens or closes the UI Inspector: the nui component tree, for extension authors.
- `tools.nui_inspector` [view]: Opens or closes the UI Inspector; the same as window.toggle_nui.
- `window.toggle_agents` [view]: Opens or closes the Agents panel: the programs driving the editor from outside, their access, requests and calls.

## Scripts and console

- `script.new` {text?} [view]: Opens a new untitled script tab holding `text`.
- `script.open` {path?, line?} [file]: Opens a Python file in a script tab, at `line` if given; without `path` a file dialog asks.
- `script.save` {tab?, path?} [file]: Saves a script tab (`tab`, else the active one) to `path`, else its own file, else where a file dialog says; with Rerun on Save it then runs.
- `script.close` {tab?, discard?} [view]: Closes a script tab (`tab`, else the active one); one with unsaved edits needs `discard`.
- `script.run` {code?, name?, path?, tab?, selection?} [process]: Runs Python in the editor, where `app`, `world`, `nuka` and `np` are bound: `code`, else the file at `path`, else a tab (`tab`, else the active one), or with `selection` that tab's selection or caret line, printing a final expression's value.
- `script.eval` {code} [process]: Runs one console line: echoes it, then runs it and prints the value of a final expression.
- `script.stop` [process]: Interrupts the running script; enabled while one runs.
- `script.result` {seq} [read]: A script run's state by the seq script.run or script.eval returned: whether it is done or failed, and its output.
- `script.complete` {code, line?, column?} [read]: Completions for Python `code` at a 1-based line and 0-based column.
- `script.tabs` [read]: The open script tabs: their files, the /script node each came from, whether they are dirty or changed on disk, the caret and the error markers.
- `script.goto` {path, line?} [view]: Shows a file at a line in the Script panel: an open tab of that name, else the file opened in a new one.
- `script.console` {since?} [read]: Console text from byte `since` on (0: all that is kept); `next` is the `since` of the following call and `dropped` says text before `since` is gone.
- `script.clear_console` [view]: Clears the console.
- `script.reload` {tab?, keep?} [file]: A script tab (`tab`, else the active one) whose file changed on disk reloads it, or with `keep` keeps its own text.
- `script.toggle_rerun` {on?} [app]: Rerun on Save: whether saving a script tab runs it; `on` sets it, else it toggles.
- `script.set_budget` {seconds?} [app]: Seconds a script may run before it is stopped, 0 for no limit; kept in the profile.
- `script.attach` {tab?, source?} [edit]: Keeps a script tab (`tab`, else the active one), or `source`, in the scene as a /script node, as one undoable edit: the node the tab came from, else a selected script node, else a new one.
- `script.load` {path?} [view]: Opens a /script node's source in a script tab: `path`, else the selection.

## The editor itself, agents and extensions

- `app.exit` {discard?} [app]: Closes the editor.
- `app.answer_prompt` {choice, path?} [file]: Answers the open save question that file.new, file.open, file.close or app.exit raised: save saves first (to `path` for an untitled document) and then reruns the action, discard reruns it dropping the changes, cancel drops it.
- `app.check` {name, value, ok?} [app]: Reports a named check from a script or smoke: it goes to the log and the start page's checks.
- `action.help` {id} [read]: Everything about one action: what it does, the arguments it reads, what it returns, an example, its kind and whether it can run now.
- `agent.mode` {mode?, allow?, deny?} [app]: Sets what programs driving the editor from outside (agents, through nuka.editor.connect or MCP) may do.
- `agent.proposals` [read]: The requests agents made that wait for the user, or were answered: proposals of edits and actions to approve, the pending ones first, then the newest.
- `agent.propose` {label?, ops, note?} [edit]: Offers an edit for the user to review in the Agents panel, with each field's value before and after; accepting applies it as one undoable step named after `label`.
- `agent.accept` {id} [edit]: Accepts an agent's proposal: applies its edit.
- `agent.approve` {request, always?} [app]: Runs the action an agent asked to run.
- `agent.reject` {id, reason?} [app]: Turns down an agent's proposal or request; the agent sees it rejected with `reason`.
- `agent.disconnect` {off?} [app]: Drops every connected agent; each finds its connection closed at its next call.
- `app.command_palette` [view]: Opens or closes the command palette, which finds and runs any action by name.
- `extensions.manager` [view]: Opens or closes the Extensions panel.
- `extensions.rescan` [app]: Looks for extensions in the extension folders again; new ones stay disabled until enabled.
- `extensions.list` [read]: The extensions found: their state and error, and the actions, panels, overlays and teleop devices and modes each one adds.
- `extensions.enable` {id} [process]: Enables an extension, and what it depends on, which loads and runs its code; the choice is kept in the profile.
- `extensions.disable` {id} [app]: Disables an extension, and what depends on it; the choice is kept in the profile.
- `extensions.reload` {id} [process]: Re-reads an extension's manifest and restarts it with its code reloaded, and the dependents that went down with it.
- `extensions.log` {id} [read]: An extension's log text.
