from matplotlib.figure import Figure
from pydantic import BaseModel, Field
from typing import Optional

from laser_measles.base import BaseComponent
from .process_transmission import TransmissionProcess, TransmissionParams
from .process_disease import DiseaseProcess, DiseaseParams


class InfectionParams(BaseModel):
    """Combined parameters for transmission and disease processes."""
    
    beta: float = Field(default=32, description="Base transmission rate", gt=0.0)
    seasonality_factor: float = Field(default=1.0, description="Seasonality factor", ge=0.0, le=1.0)
    seasonality_phase: float = Field(default=0, description="Seasonality phase")
    exp_mu: float = Field(default=11.0, description="Exposure mean (lognormal)", gt=0.0)
    exp_sigma: float = Field(default=2.0, description="Exposure sigma (lognormal)", gt=0.0)
    inf_mean: float = Field(default=8.0, description="Mean infection duration", gt=0.0)
    inf_sigma: float = Field(default=2.0, description="Shape parameter for infection duration", gt=0.0)

    @property
    def transmission_params(self) -> TransmissionParams:
        """Extract transmission-specific parameters."""
        return TransmissionParams(
            beta=self.beta,
            seasonality_factor=self.seasonality_factor,
            seasonality_phase=self.seasonality_phase,
            exp_mu=self.exp_mu,
            exp_sigma=self.exp_sigma,
            inf_mean=self.inf_mean,
            inf_sigma=self.inf_sigma
        )
    
    @property
    def disease_params(self) -> DiseaseParams:
        """Extract disease-specific parameters."""
        return DiseaseParams(
            inf_mean=self.inf_mean,
            inf_sigma=self.inf_sigma
        )


class InfectionProcess(BaseComponent):
    """
    Combined infection process that orchestrates transmission and disease progression.
    
    This component provides a unified interface for both disease transmission
    (handled by TransmissionProcess) and disease progression through states
    (handled by DiseaseProcess), similar to the biweekly model's InfectionProcess
    but for agent-based modeling.
    """

    def __init__(self, model, verbose: bool = False, params: InfectionParams | None = None) -> None:
        """
        Initialize the combined infection process.

        Args:
            model: The model object that contains the patches and parameters.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            params: Combined parameters for both transmission and disease processes.
        """
        super().__init__(model, verbose)
        
        self.params = params if params is not None else InfectionParams()
        
        # Initialize sub-components
        self.transmission = TransmissionProcess(model, verbose, self.params.transmission_params)
        self.disease = DiseaseProcess(model, verbose, self.params.disease_params)

    def __call__(self, model, tick: int) -> None:
        """
        Execute both transmission and disease progression for the given tick.

        Args:
            model: The model object containing the population, patches, and parameters.
            tick: The current time step in the simulation.
        """
        # First handle disease progression (exposed -> infectious -> recovered)
        self.disease(model, tick)
        
        # Then handle transmission (susceptible -> exposed)
        self.transmission(model, tick)

    def on_birth(self, model, tick, istart, iend) -> None:
        """
        Handle birth events by delegating to the transmission component.
        
        Args:
            model: The simulation model containing the population data.
            tick: The current tick or time step in the simulation.
            istart: The starting index of the newborns in the population array.
            iend: The ending index of the newborns in the population array.
        """
        self.transmission.on_birth(model, tick, istart, iend)

    def plot(self, fig: Figure = None):
        """
        Plot cases and incidence using the transmission component's plotting functionality.

        Args:
            fig: A Matplotlib Figure object to plot on. If None, a new figure is created.
        """
        yield from self.transmission.plot(fig)