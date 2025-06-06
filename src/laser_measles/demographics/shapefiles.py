from collections import defaultdict
from pathlib import Path
from shapefile import Reader, Writer

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def check_field(path: str | Path, field_name: str) -> bool:
    path = Path(path) if isinstance(path, str) else path
    with Reader(path) as sf:
        fields = [field[0] for field in sf.fields[1:]]
        if field_name in fields:
            return True
        return False


def add_dotname(
    path: str | Path,
    dot_name_fields: list[str],
    dotname_symbol: str = ":",
    append_suffix: str = "dotname",
    inplace: bool = False,
    field_name: str = "DOTNAME",
) -> None:
    """
    Add a DOTNAME to the shapefile.
    """

    def make_temp_path(path: Path, append_suffix: str, suffix: str) -> Path:
        return path.with_name(path.stem + "_" + append_suffix + suffix)

    # Resolve shapefile
    path = Path(path) if isinstance(path, str) else path

    # original shapefile
    with Reader(path) as sf:
        fields = [field[0] for field in sf.fields[1:]]
        if not all(field in fields for field in dot_name_fields):
            raise ValueError(
                f"Dot name fields {dot_name_fields} not found in shapefile {path}. Choices are {fields}"
            )

        if field_name in fields:
            return

        dotnames = [
            dotname_symbol.join(
                [shaperec.record[field].lower() for field in dot_name_fields]
            )
            for shaperec in sf.iterShapeRecords()
        ]

        # check that all dotnames are unique
        if len(dotnames) != len(set(dotnames)):
            raise ValueError(f"Dotnames are not unique in shapefile {path}")

        # create a new shapefile
        with Writer(make_temp_path(path, append_suffix=append_suffix, suffix=path.suffix)) as w:
            # add the original fields
            for i, field in enumerate(sf.fields):
                if i > 0:
                    w.field(*field)
            # add the new field
            w.field(field_name, "C", 50)

            record_cnt = 0

            for i, shaperec in enumerate(sf.iterShapeRecords()):
                dotname = dotname_symbol.join(
                    [shaperec.record[field].lower() for field in dot_name_fields]
                )
                # add the new field
                w.record(*shaperec.record, dotname)
                # add the shape
                w.shape(shaperec.shape)
                record_cnt += 1

    # copy the new shapefile to the old
    if inplace:
        for suffix in [".shp", ".shx", ".prj", ".cpg", ".prj", ".dbf"]:
            temp_path = make_temp_path(path, append_suffix=append_suffix, suffix=suffix)
            if temp_path.exists():
                temp_path.rename(path.with_suffix(suffix))


def get_dataframe(shapefile_path: str | Path) -> pl.DataFrame:
    """
    Get a Polars DataFrame containing the shapefile data with DOTNAME and shape columns.

    Args:
        shapefile_path: The path to the shapefile.

    Returns:
        A Polars DataFrame with DOTNAME and shape columns.
    """
    shapefile_path = (
        Path(shapefile_path) if isinstance(shapefile_path, str) else shapefile_path
    )
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}")

    with Reader(shapefile_path) as sf:
        # Get all records and shapes
        records = []
        shapes = []
        for shaperec in sf.iterShapeRecords():
            records.append(shaperec.record)
            shapes.append(shaperec.shape)

        record_dict = defaultdict(list)
        for record in records:
            for key, value in record.as_dict().items():
                record_dict[key].append(value)

        # Convert to DataFrame
        df = pl.DataFrame(record_dict)

        # Add shape column
        df = df.with_columns(pl.Series(name="shape", values=shapes))

        return df

        return df

def plot_dataframe(df: pl.DataFrame, ax: plt.Axes | None = None, plot_kwargs: dict | None = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
    if plot_kwargs is None:
        plot_kwargs = {}
    default_plot_kwargs = {
        "closed": True,
        "fill": False,
        "edgecolor": "black",
        "linewidth": 0.5,
    }
    default_plot_kwargs.update(plot_kwargs)
    xlim = [float("inf"), float("-inf")]
    ylim = [float("inf"), float("-inf")]
    def get_data(data: list[tuple[float, float]], index: int) -> list[float]:
        return [x[index] for x in data]
 
    for shape in df["shape"]:
        polygon = Polygon(shape.points, **default_plot_kwargs)
        ax.add_patch(polygon)
        xlim[0] = min(xlim[0], min(get_data(shape.points, 0)))
        xlim[1] = max(xlim[1], max(get_data(shape.points, 0)))
        ylim[0] = min(ylim[0], min(get_data(shape.points, 1)))
        ylim[1] = max(ylim[1], max(get_data(shape.points, 1)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1 / np.cos(np.mean(ylim) * np.pi / 180))
    ax.set_axis_off()
    return ax.figure


if __name__ == "__main__":
    df = get_dataframe("/home/krosenfeld/code/laser-measles/examples/expanding_kano/gadm41_NGA_shp/gadm41_NGA_2_shp")
    plot_dataframe(df)