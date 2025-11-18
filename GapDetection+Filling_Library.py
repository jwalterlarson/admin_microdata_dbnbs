# Databricks notebook source
# MAGIC %md
# MAGIC # Mind the Gap--A Library of Gap Detection/Filling Functions
# MAGIC Frequently one encounters timeseries--for example FY series--that have gaps.
# MAGIC
# MAGIC ## Module Import/Setup

# COMMAND ----------

import pyspark.sql.functions as F 
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType, IntegerType, StringType, BooleanType
from pyspark.sql.window import Window
from pyspark import StorageLevel

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
# MAGIC ### ```get_fy_ranges()```:  Determine First/Final FY for each entity.

# COMMAND ----------

def get_fy_ranges(df,
                  group_cols=['SYNTHETIC_AEUID'],
                  fy_col='FY'):
    """
    Get first/final FYs for each entity defined by a grouping.

    This function scans a list of String FY values of the form XXXX_XXXX+1
    and determines the first and final FY values in the series.

    :param df:  Spark DataFrame, input table.
    :param group_cols:  List of String, Optional, name(s) of column(s) 
                        specifying the grouping that defines each entity.
    :param fy_col:  String, Optional, name of column containing FY values 

    :return:  Table of Entities with first/final FY columns.
    :rtype:   Spark DataFrame.
    """

    fy_ranges = df.groupBy(*group_cols) \
                  .agg(F.countDistinct(fy_col).alias('FY_COUNT'),
                       F.min(F.col(fy_col)).alias('FIRST_FY'),
                       F.max(F.col(fy_col)).alias('FINAL_FY')) \
                  .orderBy(*group_cols)
    
    return fy_ranges

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```entity_fy_coverage_test()```:  Compare Distinct FY Counts with Potential FY Counts for Each Entity

# COMMAND ----------

def entity_fy_coverage_test(df,
                            group_cols=['SYNTHETIC_AEUID'],
                            fy_col='FY'):
    """
    Identify entities with FY gaps.

    This function computes FY ranges for each entity and then compares 
    the number of FY samples with the potential number of FYs based on
    the FY ranges.

    :param df:  Spark DataFrame, input table.
    :param group_cols:  List of String, Optional, name(s) of column(s) 
                        specifying the grouping that defines each entity.
    :param fy_col:  String, Optional, name of column containing FY values 

    :return:  Table of Entities with FY Counts, first/final FY values, and
              count of potential FYs.
    :rtype:   Spark DataFrame.
    """
    pass
    result_df = get_fy_ranges(df,
                              group_cols=group_cols,
                              fy_col=fy_col)
    
    result_df = result_df.withColumn('FIRST_FY_START_YEAR',
                                     F.substring('FIRST_FY', 1, 4).cast(IntegerType())) \
                          .withColumn('FINAL_FY_START_YEAR',
                                      F.substring('FINAL_FY', 1, 4).cast(IntegerType())) \
                          .withColumn('EXPECTED_FY_COUNT',
                                      F.col('FINAL_FY_START_YEAR') - 
                                       F.col('FIRST_FY_START_YEAR') + 1) \
                          .withColumn('MISSING_FY_COUNT',
                                      F.col('EXPECTED_FY_COUNT') -
                                      F.col('FY_COUNT')) \
                          .withColumn('HAS_GAP',
                                      F.when(F.col('MISSING_FY_COUNT') != 0, True) \
                                       .otherwise(False))
    
    return result_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```get_potential_fys()```:  Construct Series of FYs Given First/Final FY Values

# COMMAND ----------

def get_potential_fys(fy_ranges,
                      group_cols=['SYNTHETIC_AEUID'],
                      first_fy_col='FIRST_FY',
                      final_fy_col='FINAL_FY',
                      fy_col='FY'):
    """
    For each entity, get set of potential FY values.

    Finding which FYs we expect to see for each entity is essential
    to finding gaps in the data.

    :param fy_ranges:  Spark DataFrame, table with first/final FYs for  each
                       entity.  Output of get_fy_ranges() is suitable.
    :param group_cols:  List of String, Optional, name(s) of column(s) 
                        specifying the grouping that defines each entity.
    :param fy_col:  String, Optional, name of column containing output 
                    potential FY values.

    :return:  Table of Entities with first/final FY columns.
    :rtype:   Spark DataFrame.
    """

    potential_fys = fy_ranges.withColumn('FIRST_FY_END_YR', 
                                         F.element_at(F.split('FIRST_FY', '_'), -1).cast(IntegerType())) \
                              .withColumn('FINAL_FY_END_YR', 
                                          F.element_at(F.split('FINAL_FY', '_'), -1).cast(IntegerType())) \
                              .withColumn('POTENTIAL_END_YRS', 
                                          F.sequence(F.col('FIRST_FY_END_YR'),
                                                     F.col('FINAL_FY_END_YR')))
    potential_fys = potential_fys.select(*group_cols, 'POTENTIAL_END_YRS') \
                                 .withColumn('END_YR', F.explode('POTENTIAL_END_YRS')) \
                                 .withColumn('FY_END_YR', F.col('END_YR')) \
                                 .withColumn('FY_START_YR', F.col('FY_END_YR') - 1) \
                                 .withColumn(fy_col, 
                                             F.concat_ws('_',
                                                         F.col('FY_START_YR').cast(StringType()),
                                                         F.col('FY_END_YR').cast(StringType()))) \
                                 .drop('POTENTIAL_END_YRS',
                                       'END_YR', 
                                       'FY_START_YR', 
                                       'FY_END_YR') \
                                 .orderBy(*group_cols, fy_col)
    
    return potential_fys

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```find_gap_fys()```:  Find FY Gaps in a table for every entity in the table.

# COMMAND ----------

def find_fy_gaps(series_df,
                 group_cols=['SYNTHETIC_AEUID'],
                 fy_col='FY'):
    """
    Find Missing FYs for Each Entity in an FY-series Table.

    :param series_df:  Spark DataFrame, Original Series data.
    :param group_cols:  List of String, Optional, name(s) of column(s) 
                        specifying the grouping that defines each entity.
    :param fy_col:  String, Optional, name of column containing output 
                    potential FY values.

    :return:  Table of Entities' missing FYs, with one Entity/FY per row.
    :rtype:   Spark DataFrame.
    """

    fy_ranges = get_fy_ranges(series_df,
                              group_cols=group_cols,
                              fy_col=fy_col)
    
    potential_fys = get_potential_fys(fy_ranges,
                                      group_cols=group_cols,
                                      fy_col=fy_col)
    
    gap_fys = potential_fys.join(series_df,
                                 on=[*group_cols, fy_col],
                                 how='leftanti') \
                           .orderBy(*group_cols, fy_col)
    
    return gap_fys

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```fill_fy_gaps()```:  Flesh Out a "Gaps" Table with Fill Values

# COMMAND ----------

def fill_fy_gaps(series_df,
                 gaps_df,
                 fill_value=None):
    """
    Given a series table and a gaps table, render as a single, unioned and filled table.

    :param series_df:  Spark DataFrame, Original Series data.
    :param gaps_df:  Spark DataFrame, "Gaps" table.  Output of the function
                     find_gap_fys() is suitable.
    :param fill_value:  Optional, scalar fill value.

    :return:  A table that is the union of series_df and a fleshed-out/filled 
              version of gaps_df.
    
    :rtype:   Spark DataFrame
    """
    series_cols = series_df.columns
    gaps_cols = gaps_df.columns
    filler_cols = [c for c in series_cols if c not in gaps_cols]

    filler_df = gaps_df.withColumn(filler_cols[0],
                                   F.lit(fill_value))
    for c in filler_cols[1:]:
        filler_df = filler_df.withColumn(c,
                                         F.lit(fill_value))
    
    filler_df = filler_df.select(*series_cols)

    filled_df = series_df.union(filler_df) \
                         .orderBy(*gaps_cols)

    return filled_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```find_and_fill_fy_gaps()```:  Combined Gap Finder and Filler

# COMMAND ----------

def find_and_fill_fy_gaps(series_df,
                          group_cols=['SYNTHETIC_AEUID'],
                          fy_col='FY',
                          fill_value=None,
                          verbose=False):
    """
    Find and fill gaps in series data.

    This is a driver function that finds and fills gaps by invoking in
    succession other functions in this library:
    1. find_fy_gaps() to identify gaps; and
    2. fill_fy_gaps() to fill them.

    :param series_df:  Spark DataFrame, Original Series data.
    :param group_cols:  List of String, Optional, name(s) of column(s) 
                        specifying the grouping that defines each entity.
    :param fy_col:  String, Optional, name of column containing output 
                    potential FY values.
    :param verbose:  Boolean, Optional (default False), generate diagnostic
                     output?

    :return:  Table of entities' series, now with filled gaps.
    :rtype:   Spark DataFrame.
    """
    myname_ = 'find_and_fill_fy_gaps'

    if verbose:
        print("Enter function {0}() at {1}".format(myname_, 
                                                   datetime.now().strftime('%d %B %Y %H:%M:%S')))
        total_rows = series_df.count()
        num_entities = series_df.select(*group_cols).distinct().count()
        print("""{0}():: {1} | Commence with series comprising {2} rows of data 
               with {3} distinct entities.""".format(myname_,
                                                     datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                     total_rows,
                                                     num_entities))
        print("{0}():: {1} | Commence searching for gaps.".format(myname_,
                                                                  datetime.now().strftime('%d %B %Y %H:%M:%S')))
    gaps_df = find_fy_gaps(series_df,
                           group_cols=group_cols,
                           fy_col='FY')
    
    if verbose:
      gap_rows = gaps_df.count()
      num_gap_entities = gaps_df.select(*group_cols).distinct().count()
      print("""{0}():: {1} | Identified gaps comprising {2} rows of data 
               with {3} distinct entities.""".format(myname_,
                                                     datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                     gap_rows,
                                                     num_gap_entities))
      print("{0}():: {1} | Commence filling gaps...".format(myname_,
                                                            datetime.now().strftime('%d %B %Y %H:%M:%S')))

    filled_df = fill_fy_gaps(series_df,
                             gaps_df,
                             fill_value=fill_value)
    
    if verbose:
        final_rows = filled_df.count()
        print("""{0}():: {1} | Gap filling complete, 
                 resulting in {2} rows of data""".format(myname_,
                                                         datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                         final_rows))
        print("Exiting function {0}() at {1}".format(myname_, 
                                                     datetime.now().strftime('%d %B %Y %H:%M:%S')))
    return filled_df