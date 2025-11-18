# Databricks notebook source
# MAGIC %md
# MAGIC # Identify Degenerate SPINE_ID Values
# MAGIC Datasets in DataLab have two unique identifiers:
# MAGIC * ```SYNTHETIC_AEUID```:  Unique identifier for a person _within a given PLIDA dataset_ (e.g., Census, ATO ITR Context, Higher Education, _et cetera_); and
# MAGIC *```SPINE_ID```:  Unique identifier suitable for linking data _across two or more PLIDA datasets_.
# MAGIC
# MAGIC Sadly, there are situations in which a ```SPINE_ID``` can have multiple associated ```SYNTHETIC_AEUID``` values.  This situation casts doubt on whether the ```SPINE_ID``` value in question is in fact an individual.  We call such values _degenerate_ ```SPINE_ID```s.
# MAGIC
# MAGIC Finding and excising degenerate ```SPINE_ID```s should be an oft-repeated task (once per dataset when one is joining multiple datasets).  This notebook provides functionality for creating and using degenerate ```SPINE_ID``` lists.
# MAGIC
# MAGIC ## Usage
# MAGIC This notebook solely contains function definitions.  These functions are available in another notebook if one runs the following command in the notebook:
# MAGIC
# MAGIC ```%run "<path-to-folder>/DegenerateSpineIDs_Library"```
# MAGIC
# MAGIC ## Setup/Module Import.

# COMMAND ----------

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

# MAGIC %scala
# MAGIC dbutils.notebook.getContext.notebookPath.get

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions
# MAGIC ### ```find_degenerate_spine_ids()```:  Identify Degernerate ```SPINE_ID``` Values

# COMMAND ----------

def find_degenerate_spine_ids(source_table,
                              spine_id_col='SPINE_ID',
                              synth_id_col='SYNTHETIC_AEUID',
                              verbose=False):
    """
    Pull ID columns from a Hive Metastore table and identify "degenerate" SPINE_IDs.

    :param source_table:  String, name of Hive Metastore source data table.
    :param spine_id_col:  Name of inter-datasets identifier linkage key.
    :param synth_id_col:  String, optional, name of intra-dataset unique identifer.
    :param verbose:  Produce diagnostic output while running?

    :return:  One-column table listing degenerate SPINE_ID values (useds name from
              input argument spine_id_col).
    :rtype:   Spark DataFrame.
    """
    myname_ = 'find_degenerate_spine_ids'

    if verbose:
        print("Enter function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
        print("{0}():: {1} | Query selecting \'{2}, {3}\' from table {4}".format(myname_,
                                                                                 datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                                 spine_id_col,
                                                                                 synth_id_col,
                                                                                 source_table))

    query = 'SELECT ' + spine_id_col + ', ' + synth_id_col + ' FROM ' + source_table + ';'
    spines_and_synths = spark.sql(query) \
                             .distinct()
    initial_spine_count = spines_and_synths.select(spine_id_col) \
                                           .distinct() \
                                           .count()
    
    if verbose:
        print("{0}():: {1} | Query successfully selected {2} distinct {3}/{4} combinations".format(myname_,
                                                                                  datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                                  spines_and_synths.count(),
                                                                                  spine_id_col,
                                                                                  synth_id_col))
        print("{0}():: {1} | Grouping by the {2} distinct {3} values and counting their associated {4} values...".format(myname_,
                                                                                                                         datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                                                                         initial_spine_count,
                                                                                                                         spine_id_col,
                                                                                                                         synth_id_col))

    synths_per_spine = spines_and_synths.groupBy(spine_id_col) \
                                        .count()
    
    degenerate_spine_ids = synths_per_spine.filter(F.col('count') > 1) \
                                           .select(spine_id_col) \
                                           .orderBy(spine_id_col)
    degenerate_spine_count = degenerate_spine_ids.count()
    
    if verbose:
        print("{0}():: {1} | Identified {2} degenerate {3} values.".format(myname_,
                                                                           datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                           degenerate_spine_count,
                                                                           spine_id_col))
        print("{0}():: {1} | Overall {2} degeneracy rate:   {3}%.".format(myname_,
                                                                          datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                          spine_id_col,
                                                                          100. * degenerate_spine_count / float(initial_spine_count)))
        print("Exiting function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
        
    return degenerate_spine_ids

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```exclude_degenerate_spine_ids()``` Filter Out Rows Containing Degenerate ```SPINE_ID``` Values

# COMMAND ----------

def exclude_degenerate_spine_ids(df,
                                 degenerate_ids_table_name,
                                 spine_id_col='SPINE_ID'):
    """
    Filter out rows containing degenerate SPINE_ID values.

    :param df:  Spark DataFrame, input table to be filtered.
    :param degenerate_ids_table_name:  String, name of hive metastore
                                       table containing degenerate 
                                       SPINE_ID values.
    :param spine_id_col:  String, optional, name of SPINE_ID column.
    """
    query = ('SELECT ' + spine_id_col + 
             ' FROM ' + degenerate_ids_table_name + ';')
    degenerate_ids_df = spark.sql(query)

    filtered_df = df.join(degenerate_ids_df,
                          [spine_id_col],
                          'leftanti') \
                    .orderBy(spine_id_col)
    
    return filtered_df
