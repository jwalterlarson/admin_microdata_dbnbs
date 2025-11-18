# Databricks notebook source
# MAGIC %md
# MAGIC # Rounding Tools

# COMMAND ----------

from collections import OrderedDict

import numpy as np
import pandas as pd

import pyspark.sql.functions as F 
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Notebook Path
# MAGIC Just in case the notebook is moved and/or renamed.

# COMMAND ----------

dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Random Generator Tools Library

# COMMAND ----------

# MAGIC %run "<path-to-folder>/RandomGeneratorTools_Library"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions
# MAGIC ### ```round_to_nearest_multiple()```:  Round to Nearest Multiple of a Specified Base
# MAGIC Works for both Integer and Floating-point bases.  Note, however, that sometimes additional truncation may be required--due to binary representation precision issues--to get clean outputs (e.g., try rounding to multiples of 0.2 for an example).

# COMMAND ----------

def round_to_nearest_multiple(df,
                              cols,
                              base=10):
    """
    Rounds specified columns to nearest multiple of base value.

    This function rounds all nominated columns to the nearest multiple of a 
    provided base value.  The base value can be either an integer or floating-
    point value but floating point values should only be used if rounding 
    floating point columns to some fractional value (say for example the 
    nearest multiple of 0.05).  For each column to be rounded, an output 
    column whose name contains the suffix '_ROUNDED' will be created.

    :param df:  Spark DataFrame, input table.
    :param cols:  List of String, list of columns to be rounded.
    :param base:  Integer or Float (Optional), base for rounding.

    :return:  Table with rounded-value columns added.
    :rtype:   Spark DataFrame.
    """
    myname_ = 'round_to_nearest_multiple'

    # Detect of base type, which will determine the output type.
    if isinstance(base, int):
        output_type = IntegerType
    elif isinstance(base, float):
        output_type = FloatType
    else:
        return "{0}:: ERROR--Base Type {1} not supported.".format(myname_,
                                                                  type(base))
    
    result = df
    for col in cols:
        col_round = col + '_ROUNDED'
        result = df.withColumn(col_round,
                               (base * F.round(F.col(col) 
                                                 / base)).cast(output_type()))
    
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```random_round_up_down_nearest_multiples()``` Random choice rounding up/down to nearest multiples of a user-supplied base

# COMMAND ----------

def random_round_up_down_to_nearest_multiples(df,
                                              cols,
                                              base=10,
                                              random_seed=0):
    """
    Rounds specified columns up or down to nearest multiples of base value.

    This function rounds all nominated columns up or down with random 
    likelihood to the nearest whole-number multiples of a provided base 
    value.  The base value must be an integer.  For each column to be 
    rounded, output columns for the floor, ceiling, and rounded values 
    will be returned in columns whose names contains the suffixes '_FLOOR', 
    '_CEIL', and'_ROUNDED', respectively.

    :param df:  Spark DataFrame, input table.
    :param cols:  List of String, list of columns to be rounded.
    :param base:  Integer or Float (Optional), base for rounding.
    :param random_seed:  Numeric (Optional), random number seed.

    :return:  Table with rounded-value columns added.
    :rtype:   Spark DataFrame.
    """
    myname_ = 'random_round_up_down_to_nearest_multiples'

    # Get input row count.
    nrows = df.count()
    
    # Add monotone index and name it MONOTONE_ID.
    monotone_id_col = 'MONOTONE_ID'
    df_cols = df.columns
    result = df.rdd.zipWithIndex()
    result = result.toDF() \
                   .select(*([F.col('_1').getItem(c).alias(c) for c in df_cols] + 
                           [F.col('_2').alias(monotone_id_col)]))
    
    print(myname_, result.columns)
    # Create floor, ceiling, and choices columns.
    for col in cols:
        col_floor = col + '_FLOOR'
        col_ceil = col + '_CEIL'
        col_choices = col + '_CHOICES'
        result = result.withColumn(col_floor,
                                   (base * F.floor(F.col(col) 
                                        / base))) \
                       .withColumn(col_ceil,
                                   (base * F.ceil(F.col(col) 
                                        / base))) \
                       .withColumn(col_choices,
                                   F.array(F.col(col_floor),
                                           F.col(col_ceil)))
                   
    # Generate table of choices, one for each column to
    # be rounded.  Endow this column with a monotone index.
    choice_cols = [c + '_CHOICE' for c in cols]
    choices_df = random_df_chosen_from_list(nrows, 
                                            choice_cols,
                                            choices_list=[0, 1],
                                            seed=random_seed,
                                            monotone_id_col=monotone_id_col,
                                            return_pandas=False)
    
    # Join choices_df to result and pick randomly-generated
    # rounding choice.
    result = result.join(choices_df, 
                         monotone_id_col)
    for c in cols:
        col_choices = c + '_CHOICES'
        col_random_choice =  c + '_CHOICE'
        col_round = c + '_ROUNDED'
        result = result.withColumn(col_round,
                                   F.col(col_choices).getItem(F.col(col_random_choice)))
    
    return result
