---
title: Matplotlib and its Legend
tags:
  - python
  - matplotlib
date: 2020-03-12 18:26:50
---

**This blog comes from an answer from [How to put the legend out of the plot](https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot). Reprint it here for note. Original author keeps all the right.**


## Placing the legend (`bbox_to_anchor`) 

A legend is positioned inside the bounding box of the axes using the `loc` argument to [`plt.legend`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend).
E.g. `loc="upper right"` places the legend in the upper right corner of the bounding box, which by default extents from `(0,0)` to `(1,1)` in axes coordinates (or in bounding box notation `(x0,y0, width, height)=(0,0,1,1)`).

<!-- more -->

To place the legend outside of the axes bounding box, one may specify a tuple `(x0,y0)` of axes coordinates of the lower left corner of the legend.

```python
plt.legend(loc=(1.04,0))
```

However, a more versatile approach would be to manually specify the bounding box into which the legend should be placed, using the **`bbox_to_anchor`** argument. One can restrict oneself to supply only the `(x0,y0)` part of the bbox. This creates a zero span box, out of which the legend will expand in the direction given by the `loc` argument. E.g.

**plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")**

places the legend outside the axes, such that the upper left corner of the legend is at position `(1.04,1)` in axes coordinates.

Further examples are given below, where additionally the interplay between different arguments like `mode` and `ncols` are shown.

[![enter image description here](OIMyM.png)](https://i.stack.imgur.com/OIMyM.png)

```python
l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
l2 = plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
l3 = plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
l4 = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
l5 = plt.legend(bbox_to_anchor=(1,0), loc="lower right", 
                bbox_transform=fig.transFigure, ncol=3)
l6 = plt.legend(bbox_to_anchor=(0.4,0.8), loc="upper right")
```

Details about how to interpret the 4-tuple argument to `bbox_to_anchor`, as in `l4`, can be found in [this question](https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib). The `mode="expand"` expands the legend horizontally inside the bounding box given by the 4-tuple. For a vertically expanded legend, see [this question](https://stackoverflow.com/questions/46710546/matplotlib-expand-legend-vertically).

Sometimes it may be useful to specify the bounding box in figure coordinates instead of axes coordinates. This is shown in the example `l5` from above, where the `bbox_transform` argument is used to put the legend in the lower left corner of the figure.

### Postprocessing

Having placed the legend outside the axes often leads to the undesired situation that it is completely or partially outside the figure canvas.

Solutions to this problem are:

* **Adjust the subplot parameters**
  One can adjust the subplot parameters such, that the axes take less space inside the figure (and thereby leave more space to the legend) by using [`plt.subplots_adjust`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplots_adjust). E.g.

  ```python
    plt.subplots_adjust(right=0.7)
  ```

leaves 30% space on the right-hand side of the figure, where one could place the legend.

* **Tight layout**
  Using [`plt.tight_layout`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.tight_layout) Allows to automatically adjust the subplot parameters such that the elements in the figure sit tight against the figure edges. Unfortunately, the legend is not taken into account in this automatism, but we can supply a rectangle box that the whole subplots area (including labels) will fit into.

  ```python
    plt.tight_layout(rect=[0,0,0.75,1])
  ```

* **Saving the figure with `bbox_inches = "tight"`**
  The argument `bbox_inches = "tight"` to [`plt.savefig`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig) can be used to save the figure such that all artist on the canvas (including the legend) are fit into the saved area. If needed, the figure size is automatically adjusted.

  ```python
    plt.savefig("output.png", bbox_inches="tight")
  ```

* **automatically adjusting the subplot params**
  A way to automatically adjust the subplot position such that the legend fits inside the canvas **without changing the figure size** can be found in this answer: [Creating figure with exact size and no padding (and legend outside the axes)](https://stackoverflow.com/a/43001737/4124317)

Comparison between the cases discussed above:

[![enter image description here](zqKjY.png)](https://i.stack.imgur.com/zqKjY.png)

## Alternatives **A figure legend**

One may use a legend to the figure instead of the axes, [`matplotlib.figure.Figure.legend`](https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.legend). This has become especially useful for matplotlib version >=2.1, where no special arguments are needed

```
fig.legend(loc=7) 
```

to create a legend for all artists in the different axes of the figure. The legend is placed using the `loc` argument, similar to how it is placed inside an axes, but in reference to the whole figure - hence it will be outside the axes somewhat automatically. What remains is to adjust the subplots such that there is no overlap between the legend and the axes. Here the point _"Adjust the subplot parameters"_ from above will be helpful. An example:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi)
colors=["#7aa0c4","#ca82e1" ,"#8bcd50","#e18882"]
fig, axes = plt.subplots(ncols=2)
for i in range(4):
    axes[i//2].plot(x,np.sin(x+i), color=colors[i],label="y=sin(x+{})".format(i))

fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.75)   
plt.show()
```

[![enter image description here](v1AU6.png)](https://i.stack.imgur.com/v1AU6.png)

**Legend inside dedicated subplot axes**
An alternative to using `bbox_to_anchor` would be to place the legend in its dedicated subplot axes (`lax`). Since the legend subplot should be smaller than the plot, we may use `gridspec_kw={"width_ratios":[4,1]}` at axes creation. We can hide the axes `lax.axis("off")` but still put a legend in. The legend handles and labels need to obtained from the real plot via `h,l = ax.get_legend_handles_labels()`, and can then be supplied to the legend in the `lax` subplot, `lax.legend(h,l)`. A complete example is below.

```python
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 6,2

fig, (ax,lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[4,1]})
ax.plot(x,y, label="y=sin(x)")
....

h,l = ax.get_legend_handles_labels()
lax.legend(h,l, borderaxespad=0)
lax.axis("off")

plt.tight_layout()
plt.show()
```

This produces a plot, which is visually pretty similar to the plot from above:

[![enter image description here](4RrYb.png)](https://i.stack.imgur.com/4RrYb.png)

We could also use the first axes to place the legend, but use the `bbox_transform` of the legend axes,

```python
ax.legend(bbox_to_anchor=(0,0,1,1), bbox_transform=lax.transAxes)
lax.axis("off")
```

In this approach, we do not need to obtain the legend handles externally, but we need to specify the `bbox_to_anchor` argument.

### Further reading and notes:

*   Consider the matplotlib [legend guide](http://matplotlib.org/users/legend_guide.html) with some examples of other stuff you want to do with legends.
*   Some example code for placing legends for pie charts may directly be found in answer to this question: [Python - Legend overlaps with the pie chart](https://stackoverflow.com/questions/43272206/python-legend-overlaps-with-the-pie-chart)
*   The `loc` argument can take numbers instead of strings, which make calls shorter, however, they are not very intuitively mapped to each other. Here is the mapping for reference:

[![enter image description here](jxecX.png)](https://i.stack.imgur.com/jxecX.png)