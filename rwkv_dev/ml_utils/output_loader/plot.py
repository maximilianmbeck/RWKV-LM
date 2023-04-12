from typing import Any, Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ml_utils.logger import LOG_STEP_KEY
from ml_utils.utils import convert_to_simple_str, make_str_filename


def plot_sweep_summary(summary_df: pd.DataFrame,
                       x_axis: str,
                       y_axis: Union[str, List[str]],
                       compare_parameter: str,
                       compare_parameter_val_selection: List[Any] = [],
                       style_dict: Dict[str, Dict[str, Any]] = {},
                       title: str = None,
                       y_label: str = '',
                       x_label: str = '',
                       ax: Axes = None,
                       grid_alpha: float = 0.3,
                       ylim: Tuple[float, float] = (),
                       xlim: Tuple[float, float] = (),
                       legend_args: Dict[str, Any] = dict(loc='lower left', bbox_to_anchor=(1.0, 0.0)),
                       figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54),
                       savefig: bool = False) -> Figure:
    """Function for creating a sweep summary plot.
    Allows for selecting the x- and y-axis parameters separately. 
    Typical setup: 
    - x-axis: A sweep parameter, e.g. `data.dataset_kwargs.rotation_angle`
    - y-axis: A metric, e.g. `Accuracy-train_step-0`
    - compare_parameter: Compare different parameter setups, e.g. `init_model_step`

    Args:
        summary_df (pd.DataFrame): The sweep summary dataframe.
        x_axis (str): Parameter (column name in summary_df) to plot on the x-axis.
        y_axis (Union[str, List[str]): Parameter (column name in summary_df) to plot on the y-axis.
                                       Can also pass multiple values.
        compare_parameter (str): The compare parameter. Plot a line for each parameter.
        compare_parameter_val_selection (List[Any], optional): If specified, plot only these values. Plots all otherwise. Defaults to [].
        title (str, optional): The title. Defaults to None.
        ax (, optional): The Axes. Defaults to None.
        ylim (Tuple[float, float], optional): y-axis limist. Defaults to ().
        figsize (tuple, optional): Size of the Figure. Defaults to (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54).
        savefig (bool, optional): Save the figure. Defaults to False.

    Returns:
        Figure: The matplotlib figure.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if not isinstance(y_axis, list):
        y_axis = [y_axis]

    # select rows from compare parameter
    comp_param_vals = summary_df[compare_parameter].unique()
    if compare_parameter_val_selection:
        comp_val_sel = np.array(compare_parameter_val_selection)
        comp_param_vals = np.intersect1d(comp_param_vals, comp_val_sel)
    comp_param_vals.sort()
    # sort along x_axis
    summary_df = summary_df.sort_values(by=x_axis, axis=0)

    comp_param_str = compare_parameter.split('.')[-1]

    # get x and y axis
    for cpv in comp_param_vals:
        df = summary_df.loc[summary_df[compare_parameter] == cpv].drop(compare_parameter, axis=1)
        x_vals = df[x_axis].values
        for y_ax in y_axis:
            y_vals = df[y_ax].values
            label = f'{comp_param_str}={cpv}'
            if len(y_axis) > 1:
                label += f'#{y_ax}'
            if label in style_dict:
                style = style_dict[label]
            else:
                style = {'label': label}
            ax.plot(x_vals, y_vals, **style)
    if legend_args:
        ax.legend(**legend_args)

    ax.grid(alpha=grid_alpha)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    x_axis_label = x_label if x_label else x_axis
    ax.set_xlabel(x_axis_label)
    y_axis_label = y_label if y_label else y_axis
    ax.set_ylabel(y_axis_label)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)

    if savefig:
        fname = f'{x_axis_label}--{y_axis_label}'
        fname = make_str_filename(fname)
        f.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')
    return f


def plot_data_log_values(
    data_log_df: pd.DataFrame,
    y_axis_left: Union[List[str], str],
    y_axis_right: Union[List[str], str] = [],
    x_axis: str = LOG_STEP_KEY,
    style_dict: Dict[str, Dict[str, Any]] = {},
    title: str = '',
    y_label_left: str = '',
    y_label_right: str = '',
    x_label: str = '',
    grid_alpha: float = 0.3,
    ax: Axes = None,
    ylim: Tuple[float, float] = (),
    xlim: Tuple[float, float] = (),
    figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54),
    savefig: bool = False
) -> Figure:
    """Plot values from a data log dataframe from a single run.

    Args:
        data_log_df (pd.DataFrame): The data log dataframe.
        y_axis_left (Union[List[str], str]): Data for left y-axis. One or multiple column names.
        y_axis_right (Union[List[str], str], optional): Data for right y-axis. One or multiple column names. Defaults to [].
        x_axis (str, optional): The x-axis data, e.g. `train_step`. Defaults to LOG_STEP_KEY.
        style_dict (Dict[str, Dict[str, Any]], optional): Dictionary for customizing the plots. Defaults to {}.
        title (str, optional): Title. Defaults to None.
        y_label_left (str, optional): y-label left. Defaults to ''.
        y_label_right (str, optional): y-label right. Defaults to ''.
        x_label (str, optional): x-label. Defaults to ''.
        grid_alpha (float, optional): Alpha value for axes grid. Defaults to 0.3.
        ax (Axes, optional): An axis to plot on. Defaults to None.
        ylim (Tuple[float, float], optional): y-axis limits. Defaults to ().
        figsize (tuple, optional): The Figure size. Defaults to (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54).
        savefig (bool, optional): Save the figure. Defaults to False.


    Returns:
        Figure: The figure.
    """

    def plot_y_axis(ax: Axes, y_axis: List[str], y_label: str):
        for y_ax in y_axis:
            # select columns
            vals = data_log_df[[x_axis, y_ax]]
            vals = vals.dropna(axis=0)
            vals.sort_values(by=x_axis, axis=0, inplace=True)
            # plot values
            x_vals = vals[x_axis]
            y_vals = vals[y_ax]
            if y_ax in style_dict:
                style = style_dict[y_ax]
            else:
                style = {'label': y_ax}
            ax.plot(x_vals, y_vals, **style)

        if not y_label:
            y_label = convert_to_simple_str(y_axis, separator='+')
        ax.set_ylabel(y_label)
        ax.spines.top.set_visible(False)
        return y_label

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if not isinstance(y_axis_left, list):
        y_axis_left = [y_axis_left]

    if not isinstance(y_axis_right, list):
        y_axis_right = [y_axis_right]
    
    ax.set_prop_cycle(color=plt.get_cmap('tab10').colors)
    y_label_left = plot_y_axis(ax, y_axis_left, y_label_left)

    if len(y_axis_right) > 0:
        ax_right = ax.twinx()
        ax_right.set_prop_cycle(color=plt.get_cmap('Set2').colors)
        y_label_right = plot_y_axis(ax_right, y_axis_right, y_label_right)

    if len(y_axis_left) + len(y_axis_right) > 1:
        plt.figlegend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.87), ncol=3)

    ax.grid(alpha=grid_alpha)
    ax.spines.right.set_visible(False)

    if not x_label:
        x_label = x_axis
    ax.set_xlabel(x_label)

    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    
    if savefig:
        fname = f'{title}--{x_label}--{y_label_left}--{y_label_right}'
        fname = make_str_filename(fname)
        f.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')

    return f
