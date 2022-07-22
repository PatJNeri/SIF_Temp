# Predictive Modeling of Fv/Fm response to temperature
# Patrick Neri

# This is a formal document designed to work in tandem and allow 
# reproduction of figures and results displayed in the above 
# mentioned paper.

# In order to work, make sure the most recent version of the database
# measurement set is contained in the same repo as this file.

# %%
# Set from PSIIMasterAttempt2-24-21
from lmfit.model import save_model
from matplotlib import colors
import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt
import math
# from scipy.optimize import curve_fit
from lmfit import Model
from lmfit.models import RectangleModel, StepModel
from scipy.linalg.decomp_schur import rsf2csf
from sklearn.metrics import r2_score
import scipy.stats as stats
from statsmodels.tools.validation.validation import _right_squeeze

# Set from ClimatologyStats
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import math
from matplotlib import transforms
import scipy.stats as stt
#import cartopy.crs as ccrs
from lmfit.models import RectangleModel
import re
import random
import seaborn as sns

# Set from GeoAnalysis
from matplotlib import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.algorithms import rank
import scipy.stats as stt
import cartopy.crs as ccrs
import re

# Set from StatsdocTemp-7-17
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import math

# Common Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy.stats as (stt/stats)
