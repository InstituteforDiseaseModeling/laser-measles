"""
Test script for LaserFrameWrapper functionality.
"""

from typing import Protocol

import numpy as np
from laser_core.laserframe import LaserFrame

from laser_measles.wrapper import PrettyLaserFrameWrapper


class AgesAndStates(Protocol):
    age: np.ndarray
    states: np.ndarray


def test_wrapper():
    """Test the LaserFrameWrapper with a sample LaserFrame."""

    # Create a sample LaserFrame
    lf = LaserFrame(capacity=1000)
    lf.add_scalar_property("age", dtype=np.uint32)
    lf.add_vector_property("states", 4)  # S, E, I, R

    # Add some sample data
    lf_typed: AgesAndStates = lf  # type: ignore
    lf_typed.age[:] = np.random.randint(0, 100, 1000)
    lf_typed.states[0, :] = np.random.randint(0, 1000, 1000)  # S
    lf_typed.states[1, :] = np.random.randint(0, 100, 1000)  # E
    lf_typed.states[2, :] = np.random.randint(0, 50, 1000)  # I
    lf_typed.states[3, :] = np.random.randint(0, 200, 1000)  # R

    # Wrap and print
    wrapped_lf = PrettyLaserFrameWrapper(lf)
    print("Basic print:")
    print(wrapped_lf)

    print("\n" + "=" * 60 + "\n")

    print("With data preview:")
    wrapped_lf.print_laserframe(max_items=5)


if __name__ == "__main__":
    test_wrapper()
