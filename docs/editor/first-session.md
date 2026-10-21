# A first session

This walk-through takes a robot from file to a trained, steerable policy: import it, pose it, check its
physics, configure and run a training job, then drive the result and look inside the solver. It uses the
Unitree Go2 that ships in `examples/scenes/go2_locomotion.usda`, so every step works on a fresh checkout.
Each step links to the page that covers it in full.

## 1. Open a scene

Start the editor ([Install the editor](install.md#start-the-editor)):

```bat
python -m nuka.editor
```

On the start page, pick **New → Ground Plane**: a static floor, a sun and a sky.

## 2. Import the robot

Press ++ctrl+i++ (**File → Import Asset...**) and choose `examples/scenes/go2_locomotion.usda`. The
[import wizard](import.md#the-import-wizard) reads the file before anything changes:

- Check the verdict, and that it found 13 links and 12 joints.
- Leave **Destination** at **Add to Scene** and **Placement** at **Rest on the ground**.

Press **Import**. The Go2 lands on the floor, selected.

![The import wizard](../media/editor/import-wizard.webp)

## 3. Pose it

Switch to **Window → Robot Layout**. The Joints, Poses and Policy panels open on the right
([Robots](robots.md)).

1. Turn **Fixed Base** on, so the robot holds still while you pose it.
2. Press **Position Drive**: every joint gets a position drive, so targets move it.
3. Drag along a joint's bar to set its target. Hover the row for its drive, force limit and velocity.
4. Turn on **IK Drag** on the toolbar, select a calf and drag the gizmo: the leg's joints follow
   ([Inverse kinematics](robots.md#inverse-kinematics)).
5. In **Poses**, press **+** to keep the pose. Right-click it and choose **Set as Initial** to start from it
   after every reset.

Every step is undoable with ++ctrl+z++.

## 4. Check the physics

1. Turn **Fixed Base** off and press ++space++ to play. The robot settles onto the floor on its drives.
2. Pick **Perturb** on the toolbar, press on the body and drag: a spring pulls it, and the label shows the
   force. Or hold ++ctrl++ and drag in the Select tool ([Pushing things around](simulation.md#pushing-things-around)).
3. Press **Reset** to go back to tick 0.

![Pushing bodies around](../media/editor/physics.webp)

If something looks wrong, **View → Collider Proxies** shows the collision shapes and **View → Contact
Points** the contacts ([Viewport](viewport.md#overlays)).

## 5. Configure training

Switch to **Window → Training Layout** and open the **Task** page of the Training panel
([Training](training.md)).

1. Pick **Go2 velocity tracking**. The badge should read **Ready to train**; otherwise each problem is listed
   under it.
2. Look through the groups: **run** holds the number of robots and epochs, **reward** the reward terms,
   **command** the velocities it learns to track. Hover a field for its default and its path.
3. Save the document (++ctrl+s++). The editor keeps the training configuration per document, in your
   profile.

![The Task page](../media/editor/training-task.webp)

## 6. Train and watch

1. On the **Runs** page, check that the **Trainer** dot is green
   ([The Python it trains with](training.md#the-python-it-trains-with)).
2. Type a **Run name** and press **Start**. The run trains in another process; the tiles show the epoch,
   the reward and the steps per second.
3. Press **Watch**: sixteen of the training robots stream into the viewport, each under arrows of its
   commanded and actual velocity ([Watching a run](training.md#watching-a-run)).
4. The **Dashboard** page plots reward, episode length, losses and the learning rate as it goes.

When the reward levels off, press **Stop**. The run ends after the current epoch with a final save.

![Watching a run](../media/editor/training-watch.webp)

## 7. Drive the policy

1. On the **Checkpoints** page, press **Play Best**. The best save's policy loads onto the robot and the
   simulation plays ([Checkpoints](training.md#checkpoints)).
2. Switch back to the Robot layout. The **Policy** panel shows the policy, how often it decides and its
   action norm. **Arrows** draws the commanded and measured velocity.
3. In the Joints panel's **Teleop** section, turn on **Enabled** with the **Keyboard** device and the
   **Velocity command** mode. ++w++ ++s++ drive forward and back, ++a++ ++d++ sideways, ++q++ ++e++ turn
   ([Teleop](motion.md#teleop)).
4. While it walks, push it with **Perturb** and watch it recover.

## 8. Look inside

- **Sensors.** Select the base, then **Create → Sensor** and set its **Type** to **IMU**. The Sensors panel
  plots its accelerations and rates live ([Sensors](sensors.md)).
- **Solver.** **Window → Debug Layout** opens the Solver panel. **Record** keeps each solve's residual,
  sweeps and energy; **Locate worst** frames the element that converged worst
  ([The Solver panel](inspect.md#the-solver-panel)).
- **Timeline.** Pause and drag the timeline's scrubber back. The world returns to that step, bit for bit
  while the timeline says **EXACT**. A push from there starts a new branch and keeps the old history
  ([The timeline](inspect.md#the-timeline)).

## The same from a script

Everything above is an [action](../reference/actions/index.md), so a script in the Script panel, or an
[agent](agents.md), can do it too:

```python
app.actions.run("file.new", template="ground", discard=True)
app.actions.run("file.import", path="examples/scenes/go2_locomotion.usda", mode="add", name="go2")
app.actions.run("robot.fix_base", on=True)
app.actions.run("robot.drive", type="position")
app.actions.run("robot.fix_base", on=False)
app.actions.run("sim.play")

app.actions.run("train.set", task="go2_velocity", values={"run.max_epochs": 100})
app.actions.run("train.start", name="first")
app.actions.run("train.observe", on=True, envs=16, follow=True)
# ... later
app.actions.run("train.stop")
app.actions.run("train.play", kind="best")
app.actions.run("teleop.use", on=True, device="virtual", mode="command")
app.actions.run("teleop.input", axes={"lefty": -0.6})
```

!!! tip "Rough terrain"
    The **G1 rough-terrain walking** task trains a Unitree G1 (import Unitree's URDF or USD) on the field
    **Create → Terrain...** lays down; its defaults are the bumps that task trains on
    ([Terrain](scene.md#terrain)).
