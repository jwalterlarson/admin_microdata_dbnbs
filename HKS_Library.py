# Databricks notebook source
# MAGIC %md
# MAGIC # "Keep and Sweep" Data Suppression for Hierarchical Categorical Data
# MAGIC This is a data suppression scheme applicable to multilevel, hierarchical data.  Examples include, with examples listed from finest to coarsest granularity:
# MAGIC * ANZSCIC Industry Codes:  Class (4-digit), Group (3-digit), Subdivision (2-digit), and Division (Letter).
# MAGIC * ANZSCO Occupation Codes:  Occupation (6-digit), Unit Group (4-digit), Minor Group (3-digit), Sub-major Group (2-digit), and Major Group (1-digit)
# MAGIC * Australian Statistical Geography Standard: Meshblock (11-digit, mapped to parent SA1 by concordance), SA1 (11-digit), SA2 (9-digit), SA3 (5-digit), SA4 (3-digit), and State/Territory (1 digit).
# MAGIC
# MAGIC The functions implemented here are designed to be generic.  They may require wrapping to adapt them to a particular hierarchy.
# MAGIC
# MAGIC ### How to Use this Notebook
# MAGIC The functions defined in this notebook can be used by another notebook by running the following block of code:
# MAGIC
# MAGIC ```%run "<path-to-folder>/KeepAndSweep_Library"```
# MAGIC
# MAGIC After that command cell has run, the functions will be available.
# MAGIC
# MAGIC ## Module Import / Setup

# COMMAND ----------

import pyspark.pandas

import pyspark.sql.functions as F 
from pyspark.sql.types import StringType
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install and import ```prtpy```

# COMMAND ----------

# MAGIC %sh pip install prtpy

# COMMAND ----------

import prtpy

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Disclosure Risk Functions Library
# MAGIC This library computes various aggregations, accompanied by contributor counts and dominance statistics.

# COMMAND ----------

# MAGIC %run "/Shared/Projects/FoodRequest/Utils/DisclosureRiskFunctions_Library"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Notebook Path
# MAGIC Just in case the run command listed above doesn't work due to this notebook having been moved.

# COMMAND ----------

# MAGIC %scala
# MAGIC dbutils.notebook.getContext.notebookPath.get

# COMMAND ----------

# MAGIC %md
# MAGIC ## Function Definitions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hierarchical Keep-and-Sweep Stage
# MAGIC #### ```hks_segment()```:  Separate table rows into "keep" and "sweep" segments, based counts compared to a minimum, threshold value

# COMMAND ----------

def hks_segment(df,
                counts_col,
                counts_min_threshold=10,
                verbose=False):
    """
    Keep/Sweep segmentation of an input table.

    This function takes an input table and compares values in the column 
    counts_col with a minum threshold, segmenting the table into "keep" 
    (at or above the threshold, to be retained as-is) and "sweep" (below
    the threshold, to be forwarded for further processing).  

    :param df:  Spark DataFrame, input table.
    :param counts_cols:  String, name of column containing counts.
    :param counts_min_threshold:  Numerical, minimum comparision threshold.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.

    :return:  Two tables, the "keep" followed by the "sweep" (residue).
    :rtype:  Tuple of Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'hks_segment'
    keep = df.filter(F.col(counts_col) >= counts_min_threshold)
    sweep = df.filter(F.col(counts_col) < counts_min_threshold)

    return keep, sweep

# COMMAND ----------

# MAGIC %md
# MAGIC #### ```hks_contrib_dominance_segment()```:  Contributor/Dominance-risk-based Keep/Sweep Segmentation.
# MAGIC This function relies on pre-computed contributor counts and dominance statistics.

# COMMAND ----------

def hks_contrib_dominance_segment(df,
                                  disclosure_risk_rule,
                                  verbose=False):
    """
    Keep/Sweep segmentation using complex disclosure risk rules.

    This function takes an input table and examines pre-computed 
    disclosure risk--contributor counts and/or dominance statistics--and
    segments the table into "keep" (rows deemed safe and suitable for 
    output) and "sweep" (rows deemed unsafe, to be forwarded for further 
    processing) tables.  

    :param df:  Spark DataFrame, input table.  This table has had its
                source statistic and contributor counts computed using
                the function get_dominant_contributors() and the output
                run through the function apply_disclosure_risk_rules().
                Both of these functions, while defined in another notebook,
                are accessible through this notebook.
    :param disclosure_risk_rule:  String, Optional, tag nominating the
                                  disclosure risk rule(s) to be applied.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.

    :return:  Two tables, the "keep" followed by the "sweep" (residue).
    :rtype:  Tuple of Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'hks_contrib_dominance_segment'

    if verbose:
        print("""Enter function {0}() at {1}
              """.format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
        
    supported_rules = ['RoT+Contrib+Dominance',
                       'RoT+Contrib',
                       'RoT']
    if disclosure_risk_rule not in supported_rules:
        print(""""{0}():: {1} | ERROR:  Disclosure risk rule {2} not supported!
              Supported rules are:  {3}.
              Returning None.""".format(myname_, 
                                        datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                        disclosure_risk_rule,
                                        supported_rules))
    
    if disclosure_risk_rule == 'RoT+Contrib+Dominance':
        keep = df.filter((F.col('RoT_PASS') &
                          F.col('Ro5C_PASS') &
                          F.col('DoLC_PASS') &
                          F.col('Do2LC_PASS')))
        sweep = df.filter(~(F.col('RoT_PASS') &
                            F.col('Ro5C_PASS') &
                            F.col('DoLC_PASS') &
                            F.col('Do2LC_PASS')))
    elif disclosure_risk_rule == 'RoT+Contrib':
        keep = df.filter((F.col('RoT_PASS') &
                          F.col('Ro5C_PASS')))
        sweep = df.filter(~(F.col('RoT_PASS') &
                            F.col('Ro5C_PASS')))
    elif disclosure_risk_rule == 'RoT':
        keep = df.filter(F.col('RoT_PASS'))
        sweep = df.filter(~F.col('RoT_PASS'))
    else:
        # Should not be reachable!
        return None
    
    return keep, sweep

# COMMAND ----------

# MAGIC %md
# MAGIC #### ```hks_stage()```:  Implements full stage of Hierarchical Keep-and-Sweep

# COMMAND ----------

def hks_stage(df, 
              group_cols,
              counts_col,
              counts_min_threshold=10,
              verbose=False):
    """
    Execute Single Keeep-and-Sweep Stage.

    This function executes a single stage of a keep-and-sweep scheme.  
    It takes an input table, performs a groupBhmy over a specified group 
    of column names, aggregates the nominated counts column, and segments 
    the data depending on the relationship of the aggregated counts to a 
    minimum threshold value.  The 'keep' segment comprises the rows having
    aggregated counts at or above the minimum threshold.  The "sweep" segment
    are the rows having aggregate counts below the minimum threshold, which 
    will be forwarded (not in this function) for further processing.  The 
    function returns two tables, the 'keep' segment, followed by the 'sweep'
    segment.

    :param df:  Spark Dataframe, input table.
    :param group_cols:  List of String, ordered names of columns for grouping 
                        and subsequent aggregation.
    :param counts_col:  String, name of column containing input counts, also 
                        name applied to counts columns in the output tables.
    :param counts_min_threshold:  Numeric, minimum threshold for aggregated 
                                  counts; rows values at or above this threshold 
                                  are kept as-is, while those with aggregated 
                                  counts below the threshold are swept.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.
    
    :return:  Two tables, the 'keep' table, followed by the 'sweep' table.
    :rtype:   Two Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'hks_stage'
    aggregated_df = df.groupBy(*group_cols) \
                 .agg(_sum(F.col(counts_col)).alias(counts_col)) \
                 .select(*group_cols, counts_col) \
                 .orderBy(*group_cols)
    
    return hks_segment(aggregated_df,
                       counts_col,
                       counts_min_threshold=counts_min_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ```hks_contrib_dominance_stage()```:  Contributors/Dominance-based HKS Stage

# COMMAND ----------

def hks_contrib_dominance_stage(df, 
                                group_cols,
                                contribs_cols,
                                counts_col,
                                contributor_uid_col,
                                disclosure_risk_rule='RoT+Contrib+Dominance',
                                counts_min_threshold=10,
                                contributor_count_min_threshold=5,
                                largest_contrib_percent=50.,
                                largest_two_contribs_percent=67.,
                                verbose=False):
    """
    Execute Single Contributors/Dominance-based Keeep-and-Sweep Stage.

    This function executes a single stage of a contributors/dominance-based
    keep-and-sweep scheme.  It takes an input table, which is the swept 
    output from a previous keep/sweep stage, recovers from the contribs_cols 
    list the contributor IDs and their associated counts, storing these in 
    the columns contributor_uid_col and counts_col, respectively.  Once these
    columns are restored, a contributors/dominance-based aggregation of 
    counts is computed, complete with contributor and dominance statistics.  
    This aggregate table is then evaluated with respect to the supplied 
    disclosure risk criteria to determine which rows in the aggregate table
    are safe and "kept" as-is versus the rows deemed unsafe, which are "swept."
    The function returns two tables, the 'keep' segment, followed by the 'sweep'
    segment.

    :param df:  Spark Dataframe, input table.
    :param group_cols:  List of String, ordered names of columns for grouping 
                        and subsequent aggregation.
    :param contribs_cols:  List of two Strings, the names of columns containing
                           lists, respectively, of contributor IDs and their 
                           associated counts, ordered in descending order by the
                           counts.  These columns get exploded to recover IDs 
                           and counts, which get stored respectively in the 
                           "recovered" columns contributor_uid_col, and counts_col.
    :param counts_col:  String, name of column used to store recovered counts.
    :param contributor_uid_col:  String, name of column used to store recovered 
                                 contributor IDs.
    :param disclosure_risk_rule:  String, Optional, disclosure risk rules to be 
                                  applied.  Default is 'RoT+Contrib+Dominance',
                                  which is a combination of minimum count threshold,
                                  minimum contributor count threshold, and dominance
                                  statistics for the two largest contributors.  Other
                                  supported values include 'RoT+Contrib' and 'RoT; 
                                  if 'RoT' is desired, the user is better off not 
                                  using this function and using hks_stage() instead.
    :param counts_min_threshold:  Numeric, Optional, minimum threshold for deeming
                                  aggregated counts as safe; default value is 10.
    :param contributor_count_min_threshold:  Numeric, Optional, minimum threshold 
                                             for number of contributors to a statistic.
                                             Default value is 5, which is what ABS uses.
    :param largest_contrib_percent:  Numeric, Optional, percent threshold for the largest 
                                     contributor's share of the "total" statistic.  The 
                                     default value is 50%, which is ABS' standard.
    :param largest_two_contribs_percent:  Numeric, Optional, percent threshold for
                                          the share of the total statistic that is
                                          attributable to the two largest contributors
                                          combined.  The default value is 67%, which
                                          is the ABS standard.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.
    
    :return:  Two tables, the 'keep' table, followed by the 'sweep' table.
    :rtype:   Two Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'hks_contrib_dominance_stage'

    if verbose:
        print("Enter function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    # 1. Recover contributor IDs and respective contributions
    #    from the columns contribs_cols, storing counts and IDs
    #    in the columns counts_col and contributor_uid_col, 
    #    respectively.
    ranked_entity = contribs_cols[0]
    ranked_entity_contrib = contribs_cols[1]

    sample = df.select(*group_cols,
                       *contribs_cols) \
               .withColumn('ZIPPED_ENTITY_COLS',
                           F.arrays_zip(ranked_entity,
                                        ranked_entity_contrib)) \
               .drop(ranked_entity,
                     ranked_entity_contrib) \
               .select(*group_cols,
                       F.explode('ZIPPED_ENTITY_COLS').alias('SPLUNGE')) \
               .withColumn(contributor_uid_col,
                           F.col('SPLUNGE').getItem(ranked_entity)) \
               .withColumn(counts_col,
                           F.col('SPLUNGE').getItem(ranked_entity_contrib)) \
               .drop('SPLUNGE') \
               .sort(*group_cols)

    # 2. Compute contributor/dominance statistics and apply
    #    disclosure risk rules to the results.
    agg_stat_col = counts_col
    sample_agg = get_dominant_contributors(sample,
                                           group_cols,
                                           agg_func=F.sum,
                                           agg_col=counts_col,
                                           agg_stat_col=agg_stat_col,
                                           contrib_id_col=contributor_uid_col) \
                    .sort(*group_cols)
    
    agg_risks = apply_disclosure_risk_rules(sample_agg,
                                            total_stat_col=agg_stat_col,
                                            num_contribs_col='TOTAL_CONTRIBUTORS',
                                            test_dominance=True,
                                            first_contrib_count_col='CONTRIB_1ST_COUNT',
                                            second_contrib_count_col='CONTRIB_2ND_COUNT',
                                            sorting_keys=group_cols)
    
    # 3. Based on the risks, segment into keep/sweep tables.
    keep, sweep = hks_contrib_dominance_segment(agg_risks,
                                                disclosure_risk_rule=disclosure_risk_rule,
                                                verbose=verbose)
    if verbose:
        print("Exiting function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    return keep, sweep

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Stage:  Bin-covering Suppression
# MAGIC #### ```cflz_bin_cover_preproc()```:  Transform Table into Csirik-Frenk-Lebbe-Zhang Scheme Input Format

# COMMAND ----------

def cflz_bin_cover_preproc(df,
                           group_cols,
                           labels_col,
                           counts_col,
                           counts_list_col='COUNTS',
                           counts_to_labels_dict_col='COUNTS_TO_LABELS',
                           verbose=False):
    """
    Transform pandas DataFrame into format amenable to applying CFLZ bin-covering.

    At this point, the input table df has been filtered to ensure that all rows
    are collectively able in principle to support multiple suppression bins.  This
    was performed using the residual segmentation function .

    This function takes the input table and transforms its columns into a format 
    that is amenable with using the function cflz_bin_cover_labelled.  Specifically,
    the rows are grouped by group_cols, and the following aggregations are applied:
    1. The counts_col values--the counts--are collapsed into a list.  This 
       will be stored in the column counts_list_col. 
    2. The entries in labels_col--the labels--are organised with their respective
       counts_col values in a dict, with the dict keys being distinct counts/weight
       values and the dict values are lists of labels having that count/weight value
       associated with them.  This will be stored in the column counts_to_labels_dict_col.
    A single pandas DataFrame in the above format is returned, which has the grouping
    columns, followed by counts_list_col and counts_to_labels_dict_col.

    :param df:  pandas DataFrame, input table that contains at a minimum the 
                columns listed below (even if the arguments are Optional).
    :param group_cols:  List of String, names of grouping columns.
    :param labels_col:  String, name of column containing labels.
    :param counts_col:  String, name of column containing input counts/weights.
    :param counts_list_col:  String, Optional, name of column containing output
                             counts/weights list.
    :param counts_to_labels_dict_col:  String, Optional, name of column containing
                                       output counts-to-labels list dict.
    :param counts_sum_col:  String, Optional, name of column containing, for each row, 
                            the sum of the list items in the column counts_list_col.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.   

    :return:  Transformed table whose columns will work using CFLZ using an .apply()
              construct.
    :rtype:   pandas DataFrame
    """
    # Function name for diagnostic output.
    myname_ = 'cflz_bin_cover_preproc'
    # Collect labels (in alphabetic order but grouped first by 
    # grouping columns and count).
    labels_df = df.groupby([*group_cols,
                            counts_col]) \
                  .agg({labels_col: lambda x: sorted(list(x))}) \
                  .reset_index()
    labels_df = labels_df.groupby(group_cols) \
                         .agg({counts_col: lambda x: list(x),
                               labels_col: lambda x: list(x)}) \
                         .reset_index()
    labels_df[counts_to_labels_dict_col] = labels_df.apply(lambda x: dict(zip(x[counts_col], 
                                                                             x[labels_col])), 
                                                    axis=1)
    labels_df = labels_df[[*group_cols, counts_to_labels_dict_col]]

    # Collect counts.
    counts_df = df.groupby(group_cols) \
                  .agg({counts_col: lambda x: sorted(list(x))}) \
                  .reset_index()
    counts_df = counts_df.rename(columns={counts_col: counts_list_col})
    result_df = pd.merge(counts_df,
                         labels_df,
                         on=group_cols,
                         how='inner') \
                  .sort_values(by=group_cols)
    # Add checksum--sum of list of counts--column.
    checksum_col = counts_list_col + '_SUM'
    result_df[checksum_col] = result_df[counts_list_col].apply(lambda x: sum(x))
    
    return result_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### ```cflz_bin_cover_labelled()```:  Bin-covering Scheme for Labelled Data
# MAGIC Low-level function that takes non-tabular ```list```/```dict``` raw data and applies a modified CFLZ bin-covering scheme.

# COMMAND ----------

def cflz_bin_cover_labelled(weights,
                            weights_to_labels,
                            bin_size=10,
                            algorithm=prtpy.covering.threequarters,
                            verbose=False):
    """
     Assign labelled weights to cover a set of bins with a minimum size.

     This function takes a set of weights and labels and uses the prtpy 
     implementation of the Csirik-Frenke-Lebbe-Zhang (CFLZ; 1999) algorithm
     to assign the weights to as many bins as possible, provided the bins' 
     sizes are bounded below by a minimum size.  It does this by retaining 
     some labelling information, namely that any equal weights' labels are
     treated as interchangeable.  The function goes beyond the prtpy CFLZ
     implementation to ensure that *each input weight is assigned to a bin*;
     that is, no weights are discarded.  The function returns the following 
     nested lists (in the order listed below):
    1.  The Bins.  The outer list is the list of bins, which correspond 
        each to an inner list of their assigned weights.  The bins are 
        returned in order of increasing total weight.  
    2.  The Labels.  The outer list is the list of bins (in the same ordering
         as the weights were returned).
    3.  Sums of Bins' Weights:  This is a list, whose elements are in the same
        order as the corresponding bins, and contains the respective sums of 
        each bin's weights.

    :param weights:  List, Weights to be binned.
    :param weights_to_labels:  Dict of List, with the key corresponding to weight 
                               and the value a list of (presumably String but any 
                               object will do) labels associated with that weight.                                                                                                                                            
    :param bin_size:  Integer, Optional, minimum bin size.
    :param algorithm: prtpy object, optional, covering algorithm to be employed.
                      Users are advised not stray from the default value unless 
                      they are very familiar with the prtpy package.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.
    
    :return:  Three lists, as described in the enumarated list above.
    :rtype:  Three list instances, which the user will have to unpack.
    """
    # Function name for diagnostic output.
    myname_ = 'cflz_bin_cover_labelled'
    import copy
    
    if verbose:
        print("Enter function {0}() at {1}".format(myname_, 
                                                   datetime.now().strftime('%d %B %Y %H:%M:%S')))
    # First-cut binning from prtpy.pack,
    # which may not allocate all of the weights.
    bins = prtpy.pack(algorithm=algorithm, 
                      binsize=bin_size, 
                      items=weights)
    bins = sorted(bins, key=sum)

    if len(bins) == 0:
        print("""{0}():: {1} | prtpy.pack() returned 0 bins!
                 weights = {2}
                 bins = {3}""".format(myname_,
                                      datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                      weights,
                                      bins))
    
    # Identify weights not assigned to a bin.  Sort them in
    # decreasing order and allocate one each to the bins, 
    # which were of course sorted in increasing order by sum.
    # This should tend to keep the bin sizes as uniform as 
    # possible.
    bins_unpacked = [x for xs in bins for x in xs]
    # unallocated = list(reversed(sorted([x for x in weights 
    #                                     if x not in bins_unpacked])))
    unallocated = list((Counter(weights) -
                        Counter(bins_unpacked)).elements())
    unallocated = list(reversed(unallocated))
    augmented_bins = bins.copy()
    if len(unallocated) > 0:
        for i in range(0, len(unallocated)):
            augmented_bins[i % len(augmented_bins)].append(unallocated[i])

    # Allocate labels to bins.
    labels = copy.deepcopy(weights_to_labels)
    bin_labels = []
    for bin in augmented_bins:
        labs = []
        for item in bin:
            labs.append(labels[item].pop(0))
        bin_labels.append(labs)

    # Compute sum of weights for each bin.
    augmented_bins_sums = [sum(x) for x in augmented_bins]

    if verbose:
        print("Exiting function {0}() at {1}".format(myname_, 
                                                     datetime.now().strftime('%d %B %Y %H:%M:%S')))
    # First-cut binning.
    
    return augmented_bins, bin_labels, augmented_bins_sums

# COMMAND ----------

# MAGIC %md
# MAGIC #### ```cflz_bin_cover_driver()```:  Single-level Data Suppression using Csirik-Frenk-Lebbe-Zhang Scheme for Labelled Data

# COMMAND ----------

def cflz_bin_cover_driver(df,
                          group_cols,
                          labels_col,
                          counts_col,
                          counts_min_threshold=10,
                          verbose=False):
    """
    Driver for bin-covering data suppression.

    Typically this function is invoked as the final stage
    of a hierarchical keep-and-sweep scheme (as such being 
    invoked from the function hks_driver()).  That said, this
    function is also usable stand-alone simply as an automatic, 
    single-level, entity-merging data suppression solution.  
    
    This function takes the input table and, based on sums of 
    counts by grouping columns, partitions the table into rows 
    that are collectively (within a group) of bin-covering data 
    suppression (groups whose counts sum to the counts_min_threshold 
    value or greater) and residuual rows (groups whose counts sum to
    less than the counts_min_threshold value).  The table that is 
    amenable to bin-covering suppression is kept and routed to CFLZ bin
    covering processing scheme.  The residues table is kept as-is.
    Two Spark DataFrames are returned in the following order:  
    1. The 'kept' table that has been subjected to CFLZ bin-covering
       suppression.
    2. The 'swept' table of residues.

    :param df:  Spark DataFrame, input table.
    :param group_cols:  List of String, names of grouping columns.
    :param labels_col:  String, name of column containing labels.
    :param counts_col:  String, name of column containing counts.
    :param counts_min_threshold:  Numeric, Optional, minimum threshold
                                  value at which or above aggregated 
                                  counts data will be 'kept' for CFLZ 
                                  bin-covering schem processing rather 
                                  than 'swept' into the residues table.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.
    
    :return:  Two tables, the CFLZ-suppressed table, followed by the 
              residues table.
    :rtype:   Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'cflz_bin_cover_driver'

    if verbose:
        print("""Enter function {0}() at {1}
              Grouping columns = {2}
              Labels column = {3}
              Counts column = {4}
              Counts minimum threshold = {5}""".format(myname_, 
                                                       datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                       group_cols,
                                                       labels_col,
                                                       counts_col,
                                                       counts_min_threshold))

    # Convert input DataFrame to pandas immediately so that all the
    # processing of substance occurs in pandas rather than Spark.  
    # This is because the manipulations involved for CFLZ are more
    # readily implemented using pandas (the assumption being that
    # at this point we are dealing with remnant data that can be 
    # processed on the root node of a Spark cluster).

    if verbose:
          print("""{0}():: {1} | Convert incoming DataFrame to pandas...
              Total incoming rows = {2}
              Total incoming counts = {3}""".format(myname_, 
                                                    datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                    df.count(),
                                                    df.agg(F.sum(counts_col)).collect()[0][0]))
    p_df = df.toPandas()
    
    if verbose:
          print("""{0}():: {1} | Conversion to pandas complete.
              Total rows = {2}
              Total counts = {3}""".format(myname_, 
                                                    datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                    len(p_df),
                                                    p_df[counts_col].sum()))
    
    # Determine which rows will be kept for CFLZ and which will be
    # swept into residues.  Form group counts discriminators 
    # group_keep and group_sweep.
    group_keep = p_df.groupby(group_cols)[counts_col] \
                     .sum() \
                     .reset_index()
    group_keep = group_keep[group_keep[counts_col] >= counts_min_threshold]
    
    group_sweep = p_df.groupby(group_cols)[counts_col] \
                      .sum() \
                      .reset_index()
    group_sweep = group_sweep[group_sweep[counts_col] < counts_min_threshold]

    if verbose:
          print("""{0}():: {1} | Group Aggregation on {2} and Keep/Sweep partition.
              Total keep rows = {3}
              Total keep counts = {4}
              Total sweep rows = {5}
              Total sweep counts = {6}""".format(myname_, 
                                                 datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                 group_cols,
                                                 len(group_keep),
                                                 group_keep[counts_col].sum(),
                                                 len(group_sweep),
                                                 group_sweep[counts_col].sum()))
    if verbose:
          print("""{0}():: {1} | Use Group Aggregation keep/sweep segments to partition table."""\
                  .format(myname_, 
                          datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    # Use the group_keep and group_sweep results to 
    # partition the table into two segments:
    #   1) The "keep" segment that is amenable to bin-covering; and
    #   2) The "sweep" segment that has insufficient counts to support
    #      filling even a single bin.
    cflz_keep = pd.merge(p_df,
                         group_keep[group_cols],
                         on=group_cols,
                         how='inner')
   
    # The "sweep" segment.  Reduce via aggregation to one
    # row per group_cols value tuple.  This is all the 
    # processing the residuals get.
    residue_df = pd.merge(p_df,
                          group_sweep[group_cols],
                          on=group_cols,
                          how='inner') \
                   .groupby(group_cols) \
                   .agg({counts_col: 'sum',
                         labels_col: lambda x: list(x)}) \
                   .reset_index()

    if verbose:
          print("""{0}():: {1} | Partitioning completed.
                Segment with sufficient total {2} bin counts to be forwarded for bin-covering:
                  Total rows = {3}
                  Total counts = {4}
                Residual segment with insufficient total {5} counts for bin-covering:
                  Total rows = {6}
                  Total counts = {7}""".format(myname_, 
                                               datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                               group_cols,
                                               len(cflz_keep),
                                               cflz_keep[counts_col].sum(),
                                               group_cols,
                                               len(residue_df),
                                               residue_df[counts_col].sum()))
    
    # Convert the 'keep' segment into CFLZ input format.
    if verbose:
          print("""{0}():: {1} | Preprocessing to CFLZ bin-covering input format..."""\
                  .format(myname_, 
                          datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    cflz_df = cflz_bin_cover_preproc(cflz_keep,
                                     group_cols,
                                     labels_col,
                                     counts_col)
    if verbose:
        print("""{0}():: {1} | Conversion to CFLZ bin-covering input format complete.
                 Total counts = {2}""".format(myname_, 
                                              datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                              cflz_df['COUNTS'].apply(lambda x: sum(x)).sum()))
    
    # Apply CFLZ bin-covering.  The return column OUTPUT
    # is a 3-element tuple with the CFLZ bin-covering 
    # results, which will be unpacked below.
    if verbose:
          print("""{0}():: {1} | Apply CFLZ bin-covering..."""\
                  .format(myname_, 
                          datetime.now().strftime('%d %B %Y %H:%M:%S')))
    cflz_df['OUTPUT'] = cflz_df.apply(lambda x: 
                                      cflz_bin_cover_labelled(x['COUNTS'], 
                                                              x['COUNTS_TO_LABELS']), 
                                      axis=1)
    # Comment out the bin assignments column for now--all we need
    # for the suppression scheme are the bins' labels and total counts.
    # cflz_df['BINS'] = cflz_df['OUTPUT'].apply(lambda x: x[0])
    if verbose:
          print("""{0}():: {1} | Extract CFLZ bin-covering results..."""\
                  .format(myname_, 
                          datetime.now().strftime('%d %B %Y %H:%M:%S')))
    cflz_df[labels_col] = cflz_df['OUTPUT'].apply(lambda x: x[1])
    cflz_df[counts_col] = cflz_df['OUTPUT'].apply(lambda x: x[2])
    cflz_df = cflz_df[[*group_cols, labels_col, counts_col]]
    cflz_df = cflz_df.explode([labels_col, counts_col], 
                              ignore_index=True)
    if verbose:
          print("""{0}():: {1} | Bin covering resuts:
                  Total rows = {2}
                  Total counts = {3}""".format(myname_, 
                                               datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                               len(cflz_df),
                                               cflz_df[counts_col].sum()))
    
    # Convert back to Spark.
    # The from_pandas() calls are commented out.  In
    # theory they should work but don't due to a Db
    # runtime issue (as per stackoverflow).
    # keep = pyspark.pandas.from_pandas(cflz_df)
    # sweep = pyspark.pandas.from_pandas(residue_df)
    keep = spark.createDataFrame(cflz_df)
    sweep = spark.createDataFrame(residue_df)

    if verbose:
        print("Exiting function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))

    return keep, sweep

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hierarchical Multilevel Suppression Drivers
# MAGIC ### ```hks_driver()```:  HKS Driver Using Purely Counts-based Suppression
# MAGIC This is set to run by default using ABS' Rule of Ten.

# COMMAND ----------

def hks_driver(df,
               core_group_cols,
               hierarchy_cols,
               hierarchy_levels,
               counts_col,
               counts_min_threshold=10,
               enable_bin_covering=False,
               verbose=False):
    """
    Driver for the hierachical keep-and-sweep data suppression scheme.

    This is the code users will typically call to manage a multistage
    keep-and-sweep suppression scheme. It starts with an input table 
    that has already been grouped/aggregated by the core columns and
    the full hierarchy (resulting of course in counts for the finest
    level of granularity in the hierarchy).  It iterates keep-and-sweep
    at each level of the hierarchy, accumulating the 'keeps' in a list
    of tables while subjecting the 'sweeps' to processing at the next 
    coarsest level in the hierarchy.  Once it has reached the coarsest
    level it processess the remaining 'sweeps' using a bin-covering 
    scheme.  The various 'keeps' tables and final, bin-covered 'sweeps'
    table are then formatted to have the same columns and the tables 
    are then concatenated into a single table, sorted in lexicographic
    order with keys core_group_cols + hiearchy_cols.

    N.B.:  The HKS scheme relies on progressive grouping and aggregation
    at hierarchy levels.  As such, the DataFrame df should contain only
    the columns essential to this scheme--the core_group_cols list, the
    hierarchy_cols list, and the counts_col list.  Other information will 
    be lost in the grouping/aggregation process.

    :param df:  Spark DataFrame, Input table.
    :param core_group_cols:  List of String, Core grouping columns (those 
                             independent of the hierarchy).
    :param hierarchy_cols:  List of String, Column names defining the 
                            hierarchy listed in order from coarsest to finest
                            granularity.
    :param hierarchy_levels:  Dict, with values in hierarchy_cols as keys, with
                              hierarchy level name as corresponding values.
    :param counts_col:  String, name of column containing counts.
    :param counts_min_threshold:  Numeric, minimum counts threshold value, at 
                                  or above which rows are 'kept', below which
                                  rows are 'swept' for processing at the next 
                                  higher coarseness level.
    :param enable_bin_covering:  Boolean, optional (default False), use the 
                                 CFLZ bin-covering scheme as a final step after
                                 all levels of the hierarchy have been processed.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.  These diagnostics are 
                     compute-intensive and should be disabled if optimal 
                     performance is desired.

    :return:  Two tables, the "keep" segment to which bin-covering has been 
              applied, followed by the "sweep" segment that are residuals that 
              could not be assigned a sufficiently filled bin.  The latter 
              segment should represent a very small portion of the original data.
    :rtype:  Tuple of two Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'hks_driver'

    if verbose:
        print("Enter function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    # Snapshot of input table schema (for reconstructing 
    # suppressed columns and formatting).
    full_columns = df.columns

    # Empty list to hold "kept" tables.
    kept_tables = []

    # First Stage:  Keep-and-Sweep on the initial data.
    if verbose:
        print("""{0}():: {1} | Start HKS on \'{2}\' level...
              Total incoming rows = {3}
              Total incoming counts = {4}""".format(myname_, 
                                                    datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                    hierarchy_cols[-1],
                                                    df.count(),
                                                    df.agg(F.sum(counts_col)).collect()[0][0]))
    keep, sweep = hks_segment(df,
                              counts_col,
                              counts_min_threshold=counts_min_threshold)
    
    kept_tables.append(keep)

    if verbose:
        print("""{0}():: {1} | HKS \'{2}\' level completed.
              {3} Rows deemed safe and kept as-is.
              Total kept counts = {4}
              {5} Rows swept to next level.
              Total swept counts = {6}""".format(myname_, 
                                                 datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                 hierarchy_cols[-1],
                                                 keep.count(),
                                                 keep.agg(F.sum(counts_col)).collect()[0][0],
                                                 sweep.count(),
                                                 sweep.agg(F.sum(counts_col)).collect()[0][0]))

    # Loop over hierarchy.
    # Setup.  List of hierarchy grouping columns and columns
    # to be recovered.
    remaining_hierarchy_cols = hierarchy_cols.copy()
    hierarchy_recover_cols = [remaining_hierarchy_cols.pop(-1)]
    while len(remaining_hierarchy_cols) >= 1:
        
        if verbose:
            print("{0}():: {1} | Start HKS on \'{2}\' level...".format(myname_, 
                                                                       datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                       remaining_hierarchy_cols[-1]))
        # Set stage grouping columns for keep/sweep.
        stage_group_cols = core_group_cols + remaining_hierarchy_cols
        
        # Hierarchical keep and sweep stage.
        keep, sweep = hks_stage(sweep, 
                                stage_group_cols,
                                counts_col,
                                counts_min_threshold=counts_min_threshold)
        
        # Determine level label for fill-in columns and fill.
        current_level_col = remaining_hierarchy_cols[-1]
        level_prefix = hierarchy_levels[current_level_col] + ' '
        for col in hierarchy_recover_cols:
            keep = keep.withColumn(col,
                                   F.concat(F.lit(level_prefix),
                                            F.col(current_level_col)))
        # Format and store.
        keep = keep.select(*full_columns)
        kept_tables.append(keep)

        if verbose:
            print("""{0}():: {1} | HKS \'{2}\' level completed.
                  {3} Rows deemed safe and kept as-is.
                  Total kept counts = {4}
                  {5} Rows swept to next level.
                  Total swept counts = {6}""".format(myname_, 
                                                     datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                     remaining_hierarchy_cols[-1],
                                                     keep.count(),
                                                     keep.agg(F.sum(counts_col)).collect()[0][0],
                                                     sweep.count(),
                                                     sweep.agg(F.sum(counts_col)).collect()[0][0]))

        # Prepare for the next hierarchy level.
        if len(remaining_hierarchy_cols) == 1:
            # Reached the end of the hierarchical iterations.
            break
        else:
            hierarchy_recover_cols.append(remaining_hierarchy_cols.pop(-1))

    # At this point, keep-and-sweep has been performed at
    # each level of the hierarchy.  What remains is the set
    # of residuals remaining at the top level.  If desired, 
    # subject these residuals to the bin-covering scheme.

    if enable_bin_covering:

        if verbose:
            print("{0}():: {1} | Start Bin Covering...".format(myname_, 
                                                               datetime.now().strftime('%d %B %Y %H:%M:%S')))
        
        keep, sweep = cflz_bin_cover_driver(sweep,
                                            core_group_cols,
                                            remaining_hierarchy_cols[0],
                                            counts_col,
                                            counts_min_threshold,
                                            verbose=verbose)
        
        # Recover missing hierarchy columns using same logic as the final
        # step of the iterated hierarchies (same values of current_level_col,
        # level_prefix, and hierarchy_recover_cols).
        for col in hierarchy_recover_cols:
            keep = keep.withColumn(col,
                                   F.concat(F.lit(level_prefix), 
                                            F.lit('['),
                                            F.concat_ws(', ', 
                                                        F.col(current_level_col),
                                                        F.lit(']')))) \
                    .withColumn(col, 
                                F.regexp_replace(F.col(col),
                                                 ', ]', ']'))
        # For Union type compatibility, reformat final hierarchy 
        # column from array to string.
        keep = keep.withColumn(remaining_hierarchy_cols[0],
                               F.concat(F.lit('['),
                                        F.concat_ws(', ', 
                                                    F.col(remaining_hierarchy_cols[0]),
                                                    F.lit(']')))) \
                    .withColumn(remaining_hierarchy_cols[0], 
                                F.regexp_replace(F.col(remaining_hierarchy_cols[0]),
                                                       ', ]', ']')) \
                    .select(*full_columns)
        kept_tables.append(keep)

        if verbose:
            print("""{0}():: {1} | Bin Covering completed.
                {2} Rows resulting from consolidation.
                Total counts in consolidated rows = {3}
                {4} Rows swept as unsafe.
                Total swept counts = {5}""".format(myname_, 
                                                   datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                   keep.count(),
                                                   keep.agg(F.sum(counts_col)).collect()[0][0],
                                                   sweep.count(),
                                                   sweep.agg(F.sum(counts_col)).collect()[0][0]))

    # Assemble and sort.
    if verbose:
        print("{0}():: {1} | Union safe segments...".format(myname_, 
                                                            datetime.now().strftime('%d %B %Y %H:%M:%S')))
    keep = kept_tables[0]
    for tab in kept_tables[1:]:
        keep = keep.union(tab)
    
    group_cols = core_group_cols + hierarchy_cols
    keep = keep.orderBy(*group_cols)

    if verbose:
        print("Exiting function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    return keep, sweep

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```hks_driver_contrib_dominance()```:  HKS Driver that Uses Contributor Counts and Dominance Statistics
# MAGIC This driver does not invoke bin-covering as a final step.  This is because the complexity of implementing bin-covering informed by contributor counts and dominance statistics is difficult compared to the implementation used with a counts threshold.

# COMMAND ----------

def hks_driver_contrib_dominance(df,
                                 core_group_cols,
                                 hierarchy_cols,
                                 hierarchy_levels,
                                 counts_col,
                                 contributor_uid_col,
                                 disclosure_risk_rule='RoT+Contrib+Dominance',
                                 counts_min_threshold=10,
                                 contributor_count_min_threshold=5,
                                 largest_contrib_percent=50.,
                                 largest_two_contribs_percent=67.,
                                 verbose=False):
    """
    Contributors-and-dominance-based hierachical keep-and-sweep driver.

    This is the code users will typically call to manage a multistage
    keep-and-sweep suppression scheme. It starts with an input table 
    that has already been grouped/aggregated by the core columns, the
    full hierarchy (resulting of course in counts for the finest level 
    of granularity in the hierarchy) and for each contributing  entity. 
    THE INITIAL AGGREGATION TO CREATE THE TABLE df MUST BE DONE USING THE 
    LIBRARY FUNCTION get_dominant_contributors(), a function that has been
    loaded from another notebook by this one and is accessible to the user
    through this notebook.  The initial aggregation must be performed using
    get_dominant_contributors() because contributor counts and dominance 
    statistics are required in all keep/sweep stages.

    It iterates keep-and-sweep at each level of the hierarchy, accumulating 
    the 'keeps' in a list of tables while subjecting the 'sweeps' to 
    processing at the next coarsest level in the hierarchy.  The various 
    'keeps' tables are then formatted to have the same columns and the 
    tables are then concatenated into a single table, sorted in lexicographic 
    order with keys core_group_cols + hiearchy_cols.

    N.B.:  The HKS scheme relies on progressive grouping and aggregation
    at hierarchy levels.  As such, the DataFrame df should contain only
    the columns essential to this scheme--the core_group_cols list, the
    hierarchy_cols list, the counts_col name, and the contributor_uid_col 
    name.  Other information will be lost in the grouping/aggregation 
    process.

    :param df:  Spark DataFrame, Input table.
    :param core_group_cols:  List of String, Core grouping columns (those 
                             independent of the hierarchy).
    :param hierarchy_cols:  List of String, Column names defining the 
                            hierarchy listed in order from coarsest to finest
                            granularity.
    :param hierarchy_levels:  Dict, with values in hierarchy_cols as keys, with
                              hierarchy level name as corresponding values.
    :param counts_col:  String, name of column containing counts.
    :param contributor_uid_col:  String, column containing contributor unique 
                                 ID.
    :param disclosure_risk_rule:  String, Optional, rule for evaluating 
                                  dislosure risk to decide which rows are 
                                  to be kept/swept.  Supported values are:
                                    * 'RoT+Contribs+Dominance':  Use Rule
                                      minimum count threshold (by default the 
                                      ABS' Rule of Ten), Contributin Entities 
                                      minimum count (by default ABS' Rule of 
                                      Five contributors to a statistic), and 
                                      Dominance Statistics Combined.
                                    * If users only want to use minimum counts
                                      threshold (e.g., RoT), the use the function
                                      hks_driver() instead.
    :param counts_min_threshold:  Numeric, minimum counts threshold value, at 
                                  or above which rows are 'kept', below which
                                  rows are 'swept' for processing at the next 
                                  higher coarseness level.
    :param contributor_count_min_threshold:  Numeric, minimum contributors
                                             count required required for a 
                                             the statistic in counts_col can 
                                             be considered 'safe.'
    :param largest_contrib_percent:  Numeric, Optional, percent threshold for the largest 
                                     contributor's share of the "total" statistic.  The 
                                     default value is 50%, which is ABS' standard.
    :param largest_two_contribs_percent:  Numeric, Optional, percent threshold for
                                          the share of the total statistic that is
                                          attributable to the two largest contributors
                                          combined.  The default value is 67%, which
                                          is the ABS standard.
    :param verbose:  Boolean, optional, flag to enable diagnostic output
                     while this function is running.  These diagnostics are 
                     compute-intensive and should be disabled if optimal 
                     performance is desired.

    :return:  Two tables, the "keep" segment to which bin-covering has been 
              applied, followed by the "sweep" segment that are residuals that 
              could not be assigned a sufficiently filled bin.  The latter 
              segment should represent a very small portion of the original data.
    :rtype:  Tuple of two Spark DataFrames.
    """
    # Function name for diagnostic output.
    myname_ = 'hks_driver_contrib_dominance'

    if verbose:
        print("""Enter function {0}() at {1}
              """.format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
        
    supported_rules = ['RoT+Contrib+Dominance',
                       'RoT+Contrib',
                       'RoT']
    if disclosure_risk_rule not in supported_rules:
        print(""""{0}():: {1} | ERROR:  Disclosure risk rule {2} not supported!
              Supported rules are:  {3}.
              Returning None.""".format(myname_, 
                                        datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                        disclosure_risk_rule,
                                        supported_rules))
    
    # Snapshot of input table schema (for reconstructing 
    # suppressed columns and formatting).
    full_columns = df.columns
    
    # List of contributor-related columns that have
    # been generated automatically by the function
    # get_dominant_contributors() and are required
    # for the HKS iteration process.
    ranked_contribs_cols = ['RANKED_' + contributor_uid_col,
                            'RANKED_ENTITY_CONTRIB'] 

    # Empty list to hold "kept" tables.
    kept_tables = []

    # First Stage:  Keep-and-Sweep on the initial data.
    if verbose:
        print("""{0}():: {1} | Start HKS on \'{2}\' level...
              Total incoming rows = {3}
              Total incoming counts = {4}""".format(myname_, 
                                                    datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                    hierarchy_cols[-1],
                                                    df.count(),
                                                    df.agg(F.sum(counts_col)).collect()[0][0]))
    keep, sweep = hks_contrib_dominance_segment(df,
                                                disclosure_risk_rule=disclosure_risk_rule,
                                                verbose=verbose)
    
    kept_tables.append(keep)

    if verbose:
        print("""{0}():: {1} | HKS \'{2}\' level completed.
              {3} Rows deemed safe and kept as-is.
              Total kept counts = {4}
              {5} Rows swept to next level.
              Total swept counts = {6}""".format(myname_, 
                                                 datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                 hierarchy_cols[-1],
                                                 keep.count(),
                                                 keep.agg(F.sum(counts_col)).collect()[0][0],
                                                 sweep.count(),
                                                 sweep.agg(F.sum(counts_col)).collect()[0][0]))

    # Loop over hierarchy.
    # Setup.  List of hierarchy grouping columns and columns
    # to be recovered.
    remaining_hierarchy_cols = hierarchy_cols.copy()
    hierarchy_recover_cols = [remaining_hierarchy_cols.pop(-1)]
    while len(remaining_hierarchy_cols) >= 1:
        
        if verbose:
            print("{0}():: {1} | Start HKS on \'{2}\' level...".format(myname_, 
                                                                       datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                                       remaining_hierarchy_cols[-1]))
        # Set stage grouping columns for keep/sweep.
        stage_group_cols = core_group_cols + remaining_hierarchy_cols
        
        # Hierarchical keep and sweep stage.
        keep, sweep = hks_contrib_dominance_stage(sweep, 
                                                  stage_group_cols,
                                                  ranked_contribs_cols,
                                                  counts_col,
                                                  contributor_uid_col,
                                                  disclosure_risk_rule=disclosure_risk_rule,
                                                  counts_min_threshold=counts_min_threshold,
                                                  contributor_count_min_threshold=contributor_count_min_threshold,
                                                  largest_contrib_percent=largest_contrib_percent,
                                                  largest_two_contribs_percent=largest_two_contribs_percent)
        
        # Determine level label for fill-in columns and fill.
        current_level_col = remaining_hierarchy_cols[-1]
        level_prefix = hierarchy_levels[current_level_col] + ' '
        for col in hierarchy_recover_cols:
            keep = keep.withColumn(col,
                                   F.concat(F.lit(level_prefix),
                                            F.col(current_level_col)))
        # Format and store.
        keep = keep.select(*full_columns)
        kept_tables.append(keep)

        if verbose:
            print("""{0}():: {1} | HKS \'{2}\' level completed.
                  {3} Rows deemed safe and kept as-is.
                  Total kept counts = {4}
                  {5} Rows swept to next level.
                  Total swept counts = {6}""".format(myname_, 
                                                     datetime.now().strftime('%d %B %Y %H:%M:%S'),
                                                     remaining_hierarchy_cols[-1],
                                                     keep.count(),
                                                     keep.agg(F.sum(counts_col)).collect()[0][0],
                                                     sweep.count(),
                                                     sweep.agg(F.sum(counts_col)).collect()[0][0]))

        # Prepare for the next hierarchy level.
        if len(remaining_hierarchy_cols) == 1:
            # Reached the end of the hierarchical iterations.
            break
        else:
            hierarchy_recover_cols.append(remaining_hierarchy_cols.pop(-1))

    # Assemble and sort.
    if verbose:
        print("{0}():: {1} | Union safe segments...".format(myname_, 
                                                            datetime.now().strftime('%d %B %Y %H:%M:%S')))
    keep = kept_tables[0]
    for tab in kept_tables[1:]:
        keep = keep.union(tab)
    
    group_cols = core_group_cols + hierarchy_cols
    keep = keep.orderBy(*group_cols)

    if verbose:
        print("Exiting function {0}() at {1}".format(myname_, datetime.now().strftime('%d %B %Y %H:%M:%S')))
    
    return keep, sweep
