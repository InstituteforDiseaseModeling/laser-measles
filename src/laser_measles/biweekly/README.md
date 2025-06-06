# README
Compartmental SIR model using biweekly steps. Recommended for development and scenario building.

Example:
```python
import polars as pl
import numpy as np
lat = [0,1,2]
lon = [0,1,2]
pop=3*[10_000]
scenario = pl.DataFrame({'ids':'lat':lat, 'lon':lon, 'pop':pop})
```