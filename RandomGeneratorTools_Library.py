# Databricks notebook source
# MAGIC %md
# MAGIC # Random Generator Utilities
# MAGIC * This is a low-level 'utility' library notebook.  It is intended for use by other user-level utility notebooks.  Chances are that you won't need to use this directly.
# MAGIC * This notebook is designed to offer certain numpy-related random generator services but in a manner that can be rendered reproducible in a Spark context.
# MAGIC * More code to appear here as is needed.

# COMMAND ----------

from collections import OrderedDict

import numpy as np
import pandas as pd

import pyspark.sql.functions as F 
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions
# MAGIC ### ```random_df_chosen_from_list()```:  Random DataFrame with Values Chosen from a List
# MAGIC This is a reproducible scheme for creating a Spark DataFrame using the numpy random generator.  Values are chosen uniformly and at random (with replacement) from an input list of values.  With the seed specified, the scheme is reproducible.  An additional, monotonically increasing index column is provided in the output to allow these values to be joined in a reproducible manner 
# MAGIC

# COMMAND ----------

def random_df_chosen_from_list(nrows, 
                               cols,
                               choices_list=list(range(-2, 3, 1)),
                               seed=0,
                               monotone_id_col='MONOTONE_ID',
                               return_pandas=False):
    """
    Creates a Random DataFrame with Values Chosen from a List.

    This function uses the numpy random number generator, with seed specified 
    to ensure reproducibility, to select randomly and uniformly with replacement
    values from an input list.  The DataFrame is generated in pandas, and then 
    converted to a Spark DataFrame.  A monotonically increasing ID column is 
    included to support joining another Spark DataFrame to ensure reproducibility 
    in other Spark use cases.  The only restriction imposed using pandas is that 
    the length of the DataFrame must be small enough that the pandas DataFrame 
    generated will fit in memory on the master Spark node.


    :param len:  Integer, length of desired DataFrame.
    :param cols:  List of String, names of columns to be generated.
    :param choices_list:  List, set of elements to be sampled radomly.
    :param seed:  Numeric, seed for the numpy random number generator.
    :param mootone_id_col:  String, Optional, name of monotone ID column to be
                            generated.
    :param return_pandas:  Boolean, Optional (default False), return a pandas
                           rather than Spark DataFrame.
    
    :return:  Table with random entries (excepting the MONOTONIC_INDEX column).
    :rtype:   Spark (pandas) DataFrame if return_pandas is False (True).
    """
    # Generate monotonic index values.
    index_vals = np.arange(0, nrows, dtype=np.int64)

    # Set the seed for the numpy random number generator.
    np.random.seed(seed)

    # Generate single, long vector of randomly chosen values.
    random_vec = np.random.choice(choices_list,
                                  nrows * len(cols),
                                  replace=True)
    
    # Create OrderedDict to hold DataFrame entries.
    random_od = OrderedDict([(monotone_id_col, index_vals)])
    for i in range(0, len(cols)):
        random_od.update({cols[i]: random_vec[i * nrows: (i + 1) * nrows]})
    
    # Create pandas DataFrame.
    random_p_df = pd.DataFrame(random_od)

    if return_pandas:
        return random_p_df
    else:
        return spark.createDataFrame(random_p_df)