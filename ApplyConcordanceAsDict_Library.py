# Databricks notebook source
# MAGIC %md
# MAGIC # Apply Concordance as a ```dict```
# MAGIC This operation frequently arises in what we do with unit data:  Re-coding of a column using a concordance.
# MAGIC
# MAGIC Doing this in PySpark is stratightforward but nontrivial.  Hence a dedicated notebook.
# MAGIC
# MAGIC ## Usage
# MAGIC ### Access Function Definitions
# MAGIC Run this notebook to access its function definitions, using the command
# MAGIC
# MAGIC ```%run "<path-to-folder>/ApplyConcordanceAsDict_Library"```
# MAGIC
# MAGIC ### Application
# MAGIC 1. Get a Spark (or optionally pandas) DataFrame that contains the concordance.
# MAGIC 2. Supply this concordance DataFrame to the function ```df_to_dict()```.  **Please read this function's documentation first before using it.**
# MAGIC 3. The ```dict``` resulting from step 2 above--for discussion purposes call it ```concordance_dict```--is the concordance mapping.  It is usable as-is on any pandas DataFrame ```df``` using a lambda construct
# MAGIC ```df[value] = df[key].apply(lambda x: concordance_dict[x])```
# MAGIC
# MAGIC
# MAGIC 4. Application to a Spark DataFrame requires transformation of ```concordance_dict``` into a Spark User Defined Function (UDF).  This is accomplished by invoking the function ```translate_dict_to_udf()```.  Adding a new "values" column of a Spark DataFrame ```df``` using "keys" column and the concordance is accomplished as follows:
# MAGIC
# MAGIC ```df = df.withColumn(value_col, translate_dict_to_udf(concordance_dict)(key_col))```
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import DataFrame
import pyspark.sql.functions as F 
from pyspark.sql.types import StringType
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

import os
from datetime import datetime
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### Notebook Path
# MAGIC Just in case the run command listed above doesn't work due to this notebook having been moved.

# COMMAND ----------

notebook_path = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions
# MAGIC ### ```df_to_dict()```:  Convert ```DataFrame``` to ```dict```

# COMMAND ----------

def df_to_dict(concordance_df, 
               key_column,
               value_column,
               pandas_input=False,
               return_key_list=False, 
               broadcast_result=False, 
               verbose=False):
    """
    Convert Spark DataFrame to Python dict.

    :param concordance_df: Spark DataFrame with two columns, with the 
            first (second) column corresponding to output dict instance's
            keys (values).  Can be a pandas DataFrame if pandas_input=True.
    :param key_column:  String, name of column to be used as dict keys.
    :param value_column:  String, name of column to be used as dict values.
    :param return_key_list: Boolean (optional, default False) in addition
            to the dict instance, return a list of its keys.  This is 
            relevant in situations in which one might want to filter the
            DataFrame to which the concordance is applied to obtain only 
            rows whose values fed as keys into the concordance are in fact
            amongst the concordance dict's keys.  Note the keys list will
            be broadcasted if the next argument, broadcast_result is True.
    :param broadcast_result: Boolean (optional, default False) broadcast 
            resulting dict across Spark cluster?  Do not change this unless
            one encounters problems with consistency.  Currently Spark does
            not handle broadcast dict instances properly.  This might change.

    :return: dict instance (and optionally dict instance followed by key list).
             Variables optionally broadcasted if brodcast_result = True.
    :rtype:  Python dict.
    """
     # Function name for diagnostic output.
    myname_ = 'df_to_dict'

    if verbose:
        print("Enter function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    # Dictionary comprehension to create concordance_dict.
    if not pandas_input:
        # Default, Spark DataFrame input.
        concordance_dict = {x[key_column]: x[value_column] 
                            for x in concordance_df.select(key_column, value_column).collect()}
        concordance_keys = list(concordance_dict.keys())
    else:
        # Nominated pandas DataFrame input.
        concordance_dict = dict(zip(concordance_df[key_column],
                                    concordance_df[value_column]))
        concordance_keys = list(concordance_dict.keys())

    if broadcast_result:
        concordance_keys = spark.sparkContext.broadcast(concordance_keys)
        concordance_dict = spark.sparkContext.broadcast(concordance_dict)

    if verbose:
        print("Exiting function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))

    if return_key_list:
        return concordance_dict, concordance_keys
    else:
        return concordance_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```translate_dict_to_udf()```:  Translate ```dict``` into Spark ```UDF```
# MAGIC Unlike pandas, in Spark we cannot apply a Python ```dict``` to a DataFrame column using a ```lambda``` construct.  Instead, we are forced to wrap the ```dict``` in a Spark ```UDF``` function.  The function below takes a Python ```dict``` and returns a ```UDF``` suitable for application to a Spark DataFrame column

# COMMAND ----------

def translate_dict_to_udf(lookups):
    """
    Another take on using a dict to transform a column.
    In contradiction to *nearly everything else* I've 
    read on the topic, this works if one DOES NOT 
    use a broadcasted dict.  Go figure.

    :param lookups: Python dict, not broadcasted.

    :return: UDF that will remap a columns values using
             the dict.
    :rtype:  Spark User-defined function (UDF). 
    """
    return udf(lambda col: lookups.get(col), StringType()) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## ```concordance_remap()```:  Generic Remapper Using a Concordance
# MAGIC This function puts together the previously defined parts to remap a DataFrame column's values based on a supplied concordance and its

# COMMAND ----------

def concordance_remap(df,
                      encoded_col,
                      decoded_col,
                      concordance_table,
                      concordance_key,
                      concordance_value):
    """
    Re-maps encoded column in DataFrame using supplied concordance table.

    This function reads a concordance SQL database table and uses its key/value
    columns to re-map values of an encoded column in a DataFrame.

    :param df:  Spark DataFrame
    :param encoded_col:  String, name of 'encoded' column in df.
    :param decoded_col:  String, name of new, 'decoded' column to add to df.
    :param concordance_table:  String or Spark DataFrame.  If a String, it is
                               the path to SQL database concordance table, which 
                               will be input and applied.  If a DataFrame, it is 
                               the concordance table that will be applied.
    :param concordance_key:  String, name of 'key' column in concordance.
    :param concordance_value:  String, name of 'value' column in concordance.

    :return:  Input DataFrame with decoded column added.
    :rtype:   Spark DataFrame.
    """
    myname_ = notebook_path + '::concordance_remap'
    if isinstance(concordance_table, str):
        # Load concordance.
        query = ('SELECT ' + concordance_key + ', ' + 
                concordance_value + ' FROM ' +
                concordance_table)
        conc_df = spark.sql(query)
        is_pandas_df = False
    elif isinstance(concordance_table, DataFrame):
        conc_df = concordance_table.select(concordance_key,
                                           concordance_value)
        is_pandas_df = False
    elif isinstance(concordance_table, pd.DataFrame):
        conc_df = concordance_table[[concordance_key,
                                     concordance_value]]
        is_pandas_df = True
    else:
        print("""{0}():: Error--Unsupported type \'{1}\'' for argument \'concordance_table\'.
              Acceptable types:  [str, DataFrame, pd.DataFrame]
              """.format(myname_,
                         type(concordance_table)))
        return None
    
    # Create dict.
    conc_dict = df_to_dict(conc_df,
                           concordance_key,
                           concordance_value,
                           pandas_input=is_pandas_df)
    
    # Apply mapping.
    df = df.withColumn(decoded_col, translate_dict_to_udf(conc_dict)(encoded_col))
    
    return df
