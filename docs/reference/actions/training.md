<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Training

16 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `train.tasks`

**Kind:** [read](index.md#kinds)

The training tasks (go2\_velocity, Go2 velocity tracking, and any registered from Python): each one's groups of parameters, the robot its scene holds and the step it trains at.

**Returns:** a list of {name, title, module, root, robot, groups: \[{name, params: \[{key, type, default, ...}\]}\], ...}

## `train.new_task`

**Kind:** [file](index.md#kinds)

Starts a training task of the user's own: an extension in the profile's extensions folder (my\_task, my\_task\_2, ...) whose module registers a task built on `from` plus a Goal group (speed, turn). The extension is enabled, its module opens in the Script panel and the task becomes the document's; saving the module reloads it and the Training panel shows its groups as declared.

**Arguments**

- `from`: string, the registered task to build on, default the document's task

**Returns:** {name, title, file}

**Example**

```json
{"id": "train.new_task", "args": {"from": "g1_rough"}}
```

## `train.config`

**Kind:** [read](index.md#kinds)

The document's task and full configuration, the values that differ from the defaults (`changed`) and what keeps them from training the scene (`errors`); `task` switches to that task's defaults.

**Arguments**

- `task`: string, switch the document to this task's defaults first, optional

**Returns:** {task, config, changed, errors: \[{path, message}\], ...}

## `train.set`

**Kind:** [edit](index.md#kinds)

Sets `values` ({"group.key": value}), or `path` to `value`, after switching to `task` if given; a value that does not fit fails the call and changes nothing. The configuration is kept per document in the profile.

**Arguments**

- `values`: object, "group.key" to value, optional
- `path`: string, one "group.key", optional
- `value`: value, for `path`
- `task`: string, switch to this task first, optional

**Returns:** the configuration, as train.config

**Example**

```json
{"id": "train.set", "args": {"task": "go2_velocity", "values": {"run.num_envs": 2048, "run.max_epochs": 500, "event.randomize": true}}}
```

## `train.validate`

**Kind:** [read](index.md#kinds)

Whether the document's training configuration can train the scene as it stands.

**Returns:** {ok, errors: \[{path, message}\]}

## `train.python`

**Kind:** [app](index.md#kinds)

The Python runs start in and whether it can train (torch, rl\_games, a GPU); `path` sets it, "" returns to the editor's own.

**Arguments**

- `path`: string, a python executable, "" for the editor's, optional

**Returns:** {python, ready, error, argv, ...}

## `train.start`

**Kind:** [process](index.md#kinds)

Trains the scene as it stands, with the document's configuration, in another process as run `name` (the time by default), from the rl\_games `checkpoint` if given. One run trains at a time.

**Arguments**

- `name`: string, the run's folder name, default run-&lt;date&gt;-&lt;time&gt;
- `checkpoint`: string, an rl\_games checkpoint to resume from, optional

**Returns:** the run's status, as train.status

**Example**

```json
{"id": "train.start", "args": {"name": "go2-walk-1"}}
```

## `train.stop`

**Kind:** [process](index.md#kinds)

Ends run `name` (else the one started last) after its current epoch, with a final checkpoint.

**Arguments**

- `name`: string, the run, default the one started last

**Returns:** the run's status

## `train.status`

**Kind:** [read](index.md#kinds)

A run's state (starting, running, done, failed, lost), epoch, rewards and checkpoint count, and the last `log` lines (20) of its output.

**Arguments**

- `name`: string, the run, default the one started last, else the newest
- `log`: int, how many lines of its log, default 20

**Returns:** {name, dir, state, alive, pid, epoch, max\_epochs, reward, observe, ..., log}

## `train.runs`

**Kind:** [read](index.md#kinds)

The document's runs, newest first, each with its status.

**Returns:** a list of run statuses

## `train.metrics`

**Kind:** [read](index.md#kinds)

A run's epoch rows (reward, episode length, losses, timings) after the first `since`; `next` is the `since` of the following call.

**Arguments**

- `name`: string, the run, default the one started last, else the newest
- `since`: int, rows to skip, default 0

**Returns:** {name, rows, next}

## `train.checkpoints`

**Kind:** [read](index.md#kinds)

A run's saves in order: each rl\_games checkpoint, its NKML policy, kind (best, epoch, final), epoch and reward.

**Arguments**

- `name`: string, the run, default the one started last, else the newest

**Returns:** {name, checkpoints: \[{checkpoint, policy, kind, epoch, reward}\]}

## `train.play`

**Kind:** [process](index.md#kinds)

Loads a run's policy onto the scene's robots and plays: `policy` (a file), else the newest checkpoint of `kind`: "best" (the default), "final", "epoch" or "last".

**Arguments**

- `name`: string, the run, default the one started last, else the newest
- `policy`: string, a policy file instead of the run's, optional
- `kind`: string, best, final, epoch or last, default "best"

**Returns:** {policy, status}

## `train.observe`

**Kind:** [view](index.md#kinds)

`on: true` watches run `name` (else the one started last): the world holds its first `envs` (16) environments at the run's frames under arrows of their commanded and actual velocity, and `follow` keeps the camera on the current one. `on: false` stops; without `on` it reports, `follow` changing.

**Arguments**

- `on`: bool, true starts watching and false stops, default: reports
- `name`: string, the run, default the one started last
- `envs`: int, environments to show, 1 to 16, default 16
- `follow`: bool, the camera follows the current environment, default false

**Returns:** {on, name, envs, run\_envs, frames, sequence, steps, spacing, follow, epoch, max\_epochs, reward, terrain, arrows, ...}

**Example**

```json
{"id": "train.observe", "args": {"on": true, "envs": 16, "follow": true}}
```

## `train.reveal`

**Kind:** [process](index.md#kinds)

Opens run `name`'s folder (else the one started last, else the newest) in the file browser.

**Arguments**

- `name`: string, the run, optional

**Returns:** {dir}

## `train.panel`

**Kind:** [view](index.md#kinds)

Opens the Training panel at `page` (task, runs, dashboard or checkpoints) on run `run`, with the runs `compare` drawn over it on the Dashboard; reports what the panel shows.

**Arguments**

- `page`: string, task, runs, dashboard or checkpoints, default kept
- `run`: string, the run the panel shows, default kept
- `compare`: list of strings, runs drawn over it on the Dashboard, default kept

**Returns:** {page, run, compare, runs, series, saves, log\_lines, config\_version, changed, problems, python\_ready, ...}
