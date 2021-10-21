
# imports ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

# load data ====================================================================
data = pd.read_csv("./data/processed/leoss_decoded.tsv", sep="\t", low_memory = False)
print(f"data.shape = {data.shape}")


#

# BL_CRstartday_bit
# BL_CRstartdayNewCategories
# BL_LastKnownStatus == "Dead from COVID-19" 12% (738 / 6,025)

