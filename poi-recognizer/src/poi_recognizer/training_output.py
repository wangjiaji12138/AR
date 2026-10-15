"""Render PaddleOCR batch output without persisting terminal refreshes."""

import re
import time
from typing import TextIO

_BATCH = re.compile(r"epoch: \[(\d+)/(\d+)\], global_step: (\d+), (.*)")
_EVAL = re.compile(r"eval model:.*?\|\s*(\d+)/(\d+)\s*(.*)")


class TrainingOutput:
    """Keep one live progress line and persist only summaries and diagnostics."""

    def __init__(self, task: str, log: TextIO, terminal: TextIO) -> None:
        self.task = task
        self.log = log
        self.terminal = terminal
        self.steps = 0
        self.epoch = 0
        self.last_batch = ""
        self.last_refresh = 0.0
        self.active = False

    def _write(self, line: str) -> None:
        self.log.write(line + "\n")
        self.log.flush()
        if self.terminal.isatty():
            if self.active:
                self.terminal.write("\r\033[K")
            self.terminal.write(line + "\n")
            self.terminal.flush()
            self.active = False
        else:
            self.terminal.write(line + "\n")
            self.terminal.flush()

    def _progress(self, label: str, current: int, total: int, detail: str) -> None:
        if not self.terminal.isatty():
            return
        now = time.monotonic()
        if current < total and now - self.last_refresh < 0.2:
            return
        self.last_refresh = now
        ratio = min(1.0, current / max(1, total))
        bar = "#" * int(ratio * 24)
        self.terminal.write(
            f"\r\033[K{self.task} {label} [{bar:<24}] " f"{current}/{total} {ratio:6.1%} {detail}"
        )
        self.terminal.flush()
        self.active = True

    def _finish_train(self) -> None:
        if self.last_batch:
            self._write(f"{self.task} train {self.last_batch}")
            self.last_batch = ""

    def consume(self, line: str) -> None:
        """Consume one newline or carriage-return delimited child output record."""
        line = line.rstrip()
        if not line.strip():
            return
        loader = re.search(r"train dataloader has (\d+) iters", line)
        if loader:
            self.steps = int(loader[1])
        batch = _BATCH.search(line)
        if batch:
            epoch, epochs, step = map(int, batch.group(1, 2, 3))
            if epoch != self.epoch:
                self._finish_train()
                self.epoch = epoch
            self.last_batch = batch[0]
            current = step - (epoch - 1) * self.steps if self.steps else step
            self._progress(f"train {epoch}/{epochs}", current, self.steps or current, batch[4])
            if self.steps and current >= self.steps:
                self._finish_train()
            return
        evaluation = _EVAL.search(line)
        if evaluation:
            self._finish_train()
            current, total = map(int, evaluation.group(1, 2))
            self._progress(f"eval epoch {self.epoch}", current, total, evaluation[3])
            if current == total:
                self._write(
                    f"{self.task} eval epoch {self.epoch}: {current}/{total} {evaluation[3]}"
                )
            return
        if "cur metric," in line or "best metric," in line:
            self._finish_train()
            line = f"{self.task} epoch {self.epoch}: {line}"
        self._write(line)

    def finish(self) -> None:
        """Preserve the final batch, including a partial epoch on failure."""
        self._finish_train()
        if self.active:
            self.terminal.write("\n")
            self.terminal.flush()
            self.active = False
