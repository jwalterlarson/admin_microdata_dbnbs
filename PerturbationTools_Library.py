# Databricks notebook source
# MAGIC %md
# MAGIC # Perturbation Tools
# MAGIC
# MAGIC This notebook implements functions for perturbing tabular data.  The underlying random generator is the ```numpy``` generator, which has more capabilities than Spark's random generator.  The functions defined here are designed to be _reproducible if presented the same input data_.
# MAGIC
# MAGIC ### How to use this library
# MAGIC Execute the command shown below:
# MAGIC
# MAGIC ```%run "<path-to-folder>/PerturbationTools_Library"``` 

# COMMAND ----------

from collections import OrderedDict

import numpy as np
import pandas as pd

import pyspark.sql.functions as F 
from pyspark.sql.types import StringType
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
# MAGIC ## Load Random Generator Tools

# COMMAND ----------

# MAGIC %run "<path-to-folder>/RandomGeneratorTools_Library"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```additive_perturb_from_list()```:  Additive Perturbation Using Random Values from a List
# MAGIC By default, the input additive perturbations are randomly chosen (with replacement) from the list ```[-2, -1, 0, 1, 2]``` and are chosen with uniform likelihood.

# COMMAND ----------

def additive_perturb_from_list(df,
                               perturb_cols,
                               seed=0,
                               perturb_list=list(range(-2, 3, 1)),
                               monotone_id_col='MONOTONE_ID'):
    """
    Applies an additive perturbation (from a list) to select DataFrame columns.

    This function takes a list of perturbation values, samples uniformly and randomly 
    with repalcement to create perturbations, and applies these perturbations in an
    additive manner.  This is performed for each column nominated in the list of 
    columns to be perturbed.  Perturbations and the perturbed values are returned 
    in additional columns whose names are the name of a given column with the suffixes
    '_PERTURBATION_VALUE' and '_PERTURBED', respectively, appended.

    :param df:  Spark DataFrame, input table.
    :param perturb_cols:  List of String, names of columns whose values are to be
                          perturbed.
    :param seeds:  List of Numerical, Optional, seeds to be used for the numpy 
                   random generator, one seed per column to be perturbed.
    :param perturb_list:  List, Optional, numerical values comprising the set of 
                          perturbations to be sampled uniformly and randomly with 
                          replacement.
    
    :return:  Original table with perturbations and resultant perturbed values.
    :rtye:    Spark DataFrame.
    """
    # Suffixes for perturbation values and perturbed values.
    pertn_suffix = '_PERTURBATION_VALUE'
    pertbd_suffix = '_PERTURBED'

    # Get the length of the input DataFrame.
    nrows = df.count()

    # Add monotone ID column to copy of df.
    df_cols = df.columns
    result = df.rdd.zipWithIndex()
    result = result.toDF()
    result = result.select(*([F.col('_1').getItem(c).alias(c) for c in df_cols] + 
                             [F.col('_2').alias(monotone_id_col)]))
    
    # Generate perturbation columns in a separate DataFrame.
    pertn_cols = [c + pertn_suffix for c in perturb_cols]
    perturbations = random_df_chosen_from_list(nrows, 
                                               pertn_cols,
                                               choices_list=perturb_list,
                                               seed=seed,
                                               monotone_id_col=monotone_id_col,
                                               return_pandas=False)

    # Join perturbations onto the result.
    result = result.join(perturbations,
                         on=[monotone_id_col],
                         how='inner')
    
    # Apply additive perturbations.
    for col in perturb_cols:
        perturbation_val_col = col + pertn_suffix
        perturbed_col = col + pertbd_suffix
        result = result.withColumn(perturbed_col,
                                   F.col(col) + F.col(perturbation_val_col))
    
    return result
