import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter, FixedLocator
import numpy as np


def update_matplot_rcParams(rcParams: dict):
    rcParams.update(
        {
            "axes.labelpad": 3.0,
            "axes.labelsize": 6.0,
            "axes.titlesize": 6.0,
            "axes.linewidth": 0.5,
            "figure.figsize": [7.5, 5.0],
            "figure.subplot.bottom": 0.0,
            "figure.subplot.hspace": 0.0,
            "figure.subplot.left": 0.0,
            "figure.subplot.right": 1.0,
            "figure.subplot.top": 1.0,
            "figure.subplot.wspace": 0.0,
            "font.size": 6.0,
            "lines.linewidth": 0.5,
            "legend.fontsize": 5.0,
            "legend.labelspacing": 0.1,
            "legend.borderpad": 0.1,
            "legend.handlelength": 1,
            "legend.handletextpad": 0.5,
            "lines.linewidth": 1.0,
            "xtick.labelsize": 5.0,
            "xtick.major.pad": 2.0,
            "xtick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "xtick.minor.pad": 1.4,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.4,
            "ytick.labelsize": 5.0,
            "ytick.major.pad": 2.0,
            "ytick.major.size": 2.5,
            "ytick.major.width": 0.5,
            "ytick.minor.pad": 1.4,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.4,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            #'text.usetex': True,
            #'font.family':'serif',
            #'font.serif': 'Palatino',
            #'font.sans-serif':'Helvetica',
            #'font.monospace': 'Courier',
            #'font.family': 'serif',
        }
    )


class SizedScale(mscale.ScaleBase):
    """ """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``ax.set_yscale("mercator")`` would be
    # used to select this scale.
    name = "sized"

    def __init__(self, axis, *, sizes: np.ndarray, scale_factor=10):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        size: The degree above which to crop the data.
        """
        super().__init__(axis)
        self.sizes = sizes
        self.scale_factor = scale_factor

    def get_transform(self):
        """
        The SizedTransform class is defined below as a
        nested class of this one.
        """
        return self.SizedTransform(self.sizes, self.scale_factor)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.

        In our case, the Mercator example uses a fixed locator from -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        fmt = FuncFormatter(lambda x, pos=None: f"{x}")
        axis.set(
            major_locator=FixedLocator(np.arange(self.sizes.size * self.scale_factor)),
            major_formatter=fmt,
            minor_formatter=fmt,
        )

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return vmin, vmax

    class SizedTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = output_dims = 1

        def __init__(self, sizes, scale_factor):
            mtransforms.Transform.__init__(self)
            self.sizes = sizes
            self.scale_factor = scale_factor

        def transform_non_affine(self, a):
            """
            This transform takes a numpy array and returns a transformed copy.
            Since the range of the Mercator scale is limited by the
            user-specified threshold, the input array must be masked to
            contain only valid values.  Matplotlib will handle masked arrays
            and remove the out-of-range data from the plot.  However, the
            returned array *must* have the same shape as the input array, since
            these values need to remain synchronized with values in the other
            dimension.
            """
            xp = []
            fp = []
            cumsize = 0
            for i, size in enumerate(self.sizes):
                if i == 0:
                    xp.append(-0.5)
                    fp.append(0)

                cumsize += size
                xp.append(i + 0.5)
                fp.append(cumsize)

            return np.interp((a - 5) / self.scale_factor, xp, fp)

        def inverted(self):
            """
            Override this method so Matplotlib knows how to get the
            inverse transform for this transform.
            """
            return SizedScale.InvertedSizedTransform(self.sizes, self.scale_factor)

    class InvertedSizedTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, sizes, scale_factor):
            mtransforms.Transform.__init__(self)
            self.sizes = sizes
            self.scale_factor = scale_factor

        def transform_non_affine(self, a):
            fp = []
            xp = []
            cumsize = 0
            for i, size in enumerate(self.sizes):
                if i == 0:
                    fp.append(-0.5)
                    xp.append(0)

                cumsize += size
                fp.append(i + 0.5)
                xp.append(cumsize)

            return np.interp(a, xp, fp) * self.scale_factor + 5

        def inverted(self):
            return SizedScale.SizedTransform(self.size, self.scale_factor)


# Now that the Scale class has been defined, it must be registered so
# that Matplotlib can find it.
mscale.register_scale(SizedScale)
