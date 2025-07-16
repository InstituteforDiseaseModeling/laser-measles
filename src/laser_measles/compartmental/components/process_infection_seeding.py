from laser_measles.base import BaseLaserModel
from laser_measles.components.base_infection_seeding import BaseInfectionSeedingParams
from laser_measles.components.base_infection_seeding import BaseInfectionSeedingProcess


class InfectionSeedingParams(BaseInfectionSeedingParams):
    pass


class InfectionSeedingProcess(BaseInfectionSeedingProcess):
    """Process infection seeding."""

    def _seed_infections_in_patch(self, model: BaseLaserModel, patch_idx: int, num_infections: int) -> int:
        """Seed infections in a specific patch."""
        # Move from Susceptible to Infected
        model.patches.states.S[patch_idx] -= num_infections
        model.patches.states.E[patch_idx] += num_infections
        return num_infections
