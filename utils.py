import numpy as np
import matplotlib.pyplot as plt


def vec3(*args):
    assert len(args) == 3
    return np.array(args)


def _fit(arr: np.array):
    if len(arr) > 1:
        x = np.arange(len(arr))
        a, b = np.polyfit(x, arr, 1)
        return a * x + b
    else:
        return arr


def draw_progress(progress, fit_line=True):
    loss_history = np.array(progress.history["loss"])
    val_loss_history = np.array(progress.history.get("val_loss") or [0.0])

    plt.figure(figsize=(10, 3))
    plt.ylim(ymin=0, ymax=max(np.max(loss_history), np.max(val_loss_history)))
    plt.plot(loss_history, "r-")
    plt.plot(val_loss_history, "g-")

    if fit_line:
        plt.plot(_fit(loss_history), "b--")
        plt.plot(_fit(val_loss_history), "b--")

    plt.show()


def draw_frame(frame: np.array):
    plt.figure(figsize=(5, 5))
    plt.imshow(frame)
    plt.axis("off")
    plt.show()


import sys
import os


class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    # https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    """
    A context manager that block stdout for its scope, usage:
    with HideOutput():
        os.system('ls -l')
    """
    DEFAULT_ENABLE = True

    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        # self.fd = sys.stdout.fileno()
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno)  # Added
