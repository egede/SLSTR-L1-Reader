# SLSTR-L1-Reader
Python interface for reading SLSTR Level 1 information

```# Data in example was obtained from http://data.ceda.ac.uk/neodc/sentinel3a/data/SLSTR/L1_RBT/2018/07/07/

# Example of plotting the reflectance from the S1, S2 and S3 channels:
from slstr.reader import Reader
from slstr.plotter import *
r = Reader('SLSTR/2018/07/07/S3A_SL_1_RBT____20180707T000155_20180707T000455_20180707T014632_0179_033_130_3420_SVL_O_NR_003.SEN3')
ectance(r, 'S1', 'S2', 'S3')

# See all possible channels
print(r.all_channels)```
