import numpy as np

from laser_measles.biweekly.base import BaseComponent


class FadeOutTracker(BaseComponent):
    """
    Component for tracking the number of nodes with fade-outs.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.fade_out_tracker = np.zeros(model.params.nticks)

    def __call__(self, model, tick: int) -> None:
        self.fade_out_tracker[tick] = np.sum(model.nodes.states[1, :] == 0)
