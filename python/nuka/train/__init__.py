"""Training the editor configures and launches: task schemas, configurations and the run process.

`TASKS` holds the registered tasks; an extension adds one with `register(TaskSpec(...))`. A run is
`python -m nuka.train <folder>/job.json` (or the editor's `--train`), see runner. Importing this
package does not import torch.
"""

from .config import changed, coerce, defaults, resolve, scene_errors
from .tasks import TASKS, Group, Param, TaskSpec, register

__all__ = ["TASKS", "Group", "Param", "TaskSpec", "register", "changed", "coerce", "defaults", "resolve",
           "scene_errors"]
