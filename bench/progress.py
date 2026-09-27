from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


class ProgressReporter:
    def __init__(self, enabled):
        self.enabled = enabled
        self.console = Console(stderr=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
            disable=not enabled,
        )

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.progress.stop()

    def add_task(self, description, total=None):
        return self.progress.add_task(description, total=total)

    def update(self, task, advance=0, completed=None, description=None):
        values = {"advance": advance}
        if completed is not None:
            values["completed"] = completed
        if description is not None:
            values["description"] = description
        self.progress.update(task, **values)

    def finish(self, task):
        self.progress.remove_task(task)
