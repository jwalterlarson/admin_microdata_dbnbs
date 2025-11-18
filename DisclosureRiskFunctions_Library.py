# Databricks notebook source
# MAGIC %md
# MAGIC # Disclosure Risk Functions Library

# COMMAND ----------

import pyspark.sql.functions as F 
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType, IntegerType, StringType, BooleanType
from pyspark.sql.window import Window

import os

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
# MAGIC ### ```get_contributors_count()```:  Count Contributors to a Statistic

# COMMAND ----------

def get_contributors_count(sample_df,
                           group_cols,
                           agg_func=F.countDistinct,
                           agg_col='SYNTHETIC_ID',
                           agg_stat_col='HEADCOUNT',
                           contrib_id_col='ABN_HASH_TRUNC'):
    """
    Compute Aggregation Statistic and Get Contributor Counts.

    This function should be used when dominance tests are not relevant.

    :param sample_df:  Spark DataFrame, input sample
    :param group_cols:  List of String, names of columns for grouping
                        and aggregation.
    :param agg_func:  Spark function (from pyspark.sql.functions), Optional,
                      Aggregation function.
    :param agg_col:  String or list of Strubgs, Optional, name of column to 
                     which aggregation function is to be applied.
    :param agg_stat_col:  String, Optional, name of column to be used for
                          resulting aggregation statistic.  
    :param contrib_id_col:  String, Optional, name of column containing 
                            contributor identification information and 
                            used to enumerate contributors.
    
    :return:  Table with the total statistic and contributor count added.
    :rtype:   Spark DataFrame
    """
    contrib_counts = sample_df.groupby(*group_cols) \
                              .agg(agg_func(*agg_col).alias(agg_stat_col),
                                   agg_func(contrib_id_col).alias('TOTAL_CONTRIBUTORS'))
    
    return contrib_counts

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```get_dominant_contributors()```:  Identify and Quantify Impact of Dominant Contributors

# COMMAND ----------

def get_dominant_contributors(sample_df,
                              group_cols,
                              agg_func=F.countDistinct,
                              agg_col='SYNTHETIC_AEUID',
                              agg_stat_col='HEADCOUNT',
                              contrib_id_col='ABN_HASH_TRUNC'):
    """
    Get contributor counts and identify the two most dominant contributors.

    :param sample_df:  Spark DataFrame, input sample
    :param group_cols:  List of String, names of columns for grouping
                        and aggregation.
    :param agg_func:  Spark function (from pyspark.sql.functions), Optional,
                      Aggregation function.
    :param agg_col:  String, Optional, name of column to which aggregation 
                     function is to be applied.
    :param agg_stat_col:  String, Optional, name of column to be used for
                          resulting aggregation statistic.  This is computed
                          both for comparison purposes with the two most 
                          dominant contributors' counts and as a cross-
                          check on aggregation computed outside of this 
                          function.
    :param contrib_id_col:  String, Optional, name of column containing 
                            contributor identification information.
    
    :return:  Table of dominance statistics, including:
              * The total statistic, for which contributor counts and 
                dominance statistics are to be computed.
              * Total number of distinct contrib_id_col values (contributor
                count)
              * ID of largest contributor (as per contrib_id_col)
              * Contribution of largest contributor to total
              * ID of second largest contributor (as per contrib_id_col)
              * Contribution of second largest contributor to total
    :rtype:   Spark DataFrame
    """
    # 1. Reproduce Total Statistic and get Total Contributor Count.
    contrib_counts = sample_df.groupby(*group_cols) \
                              .agg(agg_func(agg_col).alias(agg_stat_col),
                                   F.countDistinct(contrib_id_col).alias('TOTAL_CONTRIBUTORS'))
    
    # 2. Get largest two contributors' contributions.
    all_contribs = sample_df.groupBy(*group_cols, contrib_id_col) \
                            .agg(agg_func(agg_col).alias('ENTITY_CONTRIB')) \
                            .orderBy(*group_cols, F.col('ENTITY_CONTRIB').desc())

    win = Window.partitionBy(*group_cols) \
                .orderBy(F.desc('ENTITY_CONTRIB')) \
                .rowsBetween(Window.unboundedPreceding,
                             Window.unboundedFollowing)  
    
    ranked_ids = 'RANKED_' + contrib_id_col
    ranked_contribs = 'RANKED_' + 'ENTITY_CONTRIB'
    ranked = all_contribs.withColumn(ranked_ids,
                                     F.collect_list(F.col(contrib_id_col)).over(win)) \
                         .withColumn(ranked_contribs,
                                     F.collect_list(F.col('ENTITY_CONTRIB')).over(win)) \
                         .select(*group_cols,
                                 ranked_ids, 
                                 ranked_contribs) \
                         .distinct() \
                         .orderBy(*group_cols)
    
    first_contrib_id = 'CONTRIB_1ST_' + contrib_id_col
    second_contrib_id = 'CONTRIB_2ND_' + contrib_id_col
    dom_df = ranked.withColumn(first_contrib_id,
                               F.col(ranked_ids).getItem(0)) \
                   .withColumn('CONTRIB_1ST_COUNT',
                               F.col(ranked_contribs).getItem(0)) \
                   .withColumn(second_contrib_id,
                               F.when(F.size(F.col(ranked_ids)) >= 2,
                                      F.col(ranked_ids).getItem(1)) \
                                .otherwise(None)) \
                   .withColumn('CONTRIB_2ND_COUNT',
                               F.when(F.size(F.col(ranked_contribs)) >= 2,
                                      F.col(ranked_contribs).getItem(1)) \
                                .otherwise(None)) \
                   .select(*group_cols,
                           first_contrib_id, 
                           second_contrib_id,
                           'CONTRIB_1ST_COUNT', 
                           'CONTRIB_2ND_COUNT',
                           ranked_ids,
                           ranked_contribs) \
                   .orderBy(*group_cols)
    
    # 3. Join in contributor counts.
    result_df = contrib_counts.join(dom_df,
                                    group_cols,
                                    'inner') \
                              .orderBy(*group_cols)

    return result_df              

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```apply_disclosure_risk_rules()```:  Apply Disclosure Risk Rules to Mark Rows for Inclusion/Exclusion

# COMMAND ----------

def apply_disclosure_risk_rules(dom_stats_df,
                                total_stat_col='HEADCOUNT',
                                total_stat_threshold=10,
                                num_contribs_col='TOTAL_CONTRIBUTORS',
                                num_contribs_threshold=5,
                                test_dominance=True,
                                first_contrib_count_col='CONTRIB_1ST_COUNT',
                                first_contrib_pct_threshold=50.,
                                second_contrib_count_col='CONTRIB_2ND_COUNT',
                                largest_two_contribs_pct_threshold=67.,
                                sorting_keys=None):
    """
    Apply disclosure risk rules to table containing dominance statistics.

    This function applies the following rules to the dominance statistics
    table:
    1.  "Rule of Ten" (RoT): specifically a cell containing count values  
        must have counts >= total_stat_threshold.
    2.  Total Contributors (Ro5C):  the number of organisations/entities 
        that contribute to the statistic must be >= num_contribs_threshold.
    3.  Dominance of Largest Contributor (DoLC):  The largest contributor  
        must contribute less than first_contrib_pct_threshold% to the total 
        statistic.
    4.  Dominance of Two Largest Contributors (Do2LC):  The combined 
        contribution of the two largest contributors must contribute less 
        than largest_two_contribs_pct_threshold% to the total statistic.
    
    Application of each of these rules results in a corresponding column
    in the output table, whose name is given by the abbreviated rule name
    (in parentheses above) with the suffix '_PASS' appended.  Rule outcomes 
    are marked as True/False, depending on whether the data in a row PASS 
    the rule's criterion.  These four columns are combined logical OVERALL_PASS
    column, which has values True if *all* of the corresponding rule pass columns 
    in a row are True, False if *any* of the corresponding rule pass columns in 
    a row is False.  There is one final column added named 'SUPPRESS_ROW', which 
    is True when 'OVERALL_PASS' is False, and otherwise False.

    This table can be used to filter the raw results dataset to determine 
    which rows are safe to submit for clearance and which will have to be
    suppressed.

    :param dom_stats_df:  Spark DataFrame containing columns for the total 
                          statistic, total contributors count, and counts 
                          for the each of the two largest contributors.
    :param total_stat_col:  String, Optional, name of the column containing
                            values for the "total" statistic.
    :param total_stat_threshold:  Numeric, Optional, safe threshold value for 
                                  the "total" statistic.  Default value is 10,
                                  which is "the safest number ever."
    :param num_contribs_col:  String, Optional, name of the column containing
                              the total number of entities contributing to the
                              total statistic.
    :param num_contribs_threshold:  Numeric, Optional, safe threshold value for
                                    contributor counts.  Default value is 5, 
                                    which is used by ABS.  
    :param test_dominance:  Boolean, Optional (default True), apply dominance 
                            tests?  If true, the 50%/67% tests are applied using
                            the columns containing the two largest contributors'
                            respective counts.
    :param first_contrib_count_col:  String, Optional, name of the column with 
                                     counts associated with the largest contributor
                                     to the sample for the total statistic.  Ignored
                                     if test_dominance is False.
    :param first_contrib_pct_threshold:  Numeric, Optional, safe threshold value 
                                         (in percent of the total statistic) that 
                                         can be attributable to the largest 
                                         contributor.  The default value is 50%,
                                         again, standard for ABS.
    :param second_contrib_count_col:  String, Optional, name of the column with 
                                      counts associated with the second largest 
                                      contributor to the sample for the total 
                                      statistic.   Ignored if test_dominance is 
                                      False.
    :param largest_two_contribs_pct_threshold:  Numeric, Optional, safe threshold
                                                value (in percent of the total 
                                                statistic) that is attributable to
                                                the two largest contributors combined.
                                                The default value is 67%, which is ABS'
                                                standard.
    :param sortig_keys:  List of String, Optional, keys for lexicographic sort of
                         output table (if sorting is desired).
    
    :return:  Table containing principal/second contributor percentages (and their 
              sum), rule outcomes, and overall pass and suppression recommendations.
    :rtype:   Spark DataFrame.
    """

    rules_results = dom_stats_df.withColumn('RoT_PASS',
                                            F.when(F.col(total_stat_col) >= total_stat_threshold, 
                                                   True) \
                                             .otherwise(False)) \
                                .withColumn('Ro5C_PASS',
                                            F.when(F.col(num_contribs_col) >= num_contribs_threshold, 
                                                   True) \
                                             .otherwise(False)) 
    if test_dominance:
        rules_results = rules_results.withColumn('CONTRIB_1ST_PCT',
                                                 100. * F.col(first_contrib_count_col) 
                                                 / F.col(total_stat_col)) \
                                     .withColumn('CONTRIB_2ND_PCT',
                                                 100. * F.col(second_contrib_count_col) 
                                                 / F.col(total_stat_col)) \
                                     .withColumn('CONTRIB_1ST_PLUS_2ND_PCT',
                                                 F.col('CONTRIB_1ST_PCT') +
                                                 F.col('CONTRIB_2ND_PCT')) \
                                     .withColumn('DoLC_PASS',
                                                 F.when(F.col('CONTRIB_1ST_PCT') < first_contrib_pct_threshold, 
                                                        True) \
                                                 .otherwise(False)) \
                                     .withColumn('Do2LC_PASS',
                                                 F.when(F.col('CONTRIB_1ST_PLUS_2ND_PCT') < largest_two_contribs_pct_threshold, 
                                                        True) \
                                                 .otherwise(False)) \
                                     .withColumn('OVERALL_PASS',
                                                 F.when((F.col('RoT_PASS') & 
                                                     F.col('Ro5C_PASS') &
                                                     F.col('DoLC_PASS') &
                                                     F.col('Do2LC_PASS')), True) \
                                                 .otherwise(False)) \
                                     .withColumn('SUPPRESS_ROW',
                                                 ~F.col('OVERALL_PASS'))
    else:
        rules_results = rules_results.withColumn('OVERALL_PASS',
                                                 F.when((F.col('RoT_PASS') & 
                                                         F.col('Ro5C_PASS')), True) \
                                                  .otherwise(False)) \
                                     .withColumn('SUPPRESS_ROW',
                                                 ~F.col('OVERALL_PASS'))
    
    if sorting_keys is None:
        return rules_results
    else:
        rules_results = rules_results.orderBy(*sorting_keys)
        return rules_results                     

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```suppress_entity_test()```:  Determine Whether an Aggregation Entity Needs Suppression

# COMMAND ----------

def suppress_entity_test(disc_risks,
                         entity_col,
                         supp_row_col='SUPPRESS_ROW'):
    """
    Determine whether an Aggregation Entity needs suppression

    The logic here is that when aggregation occurs over multiple variables, 
    there may be cases in which one or more occurrences of a given entity 
    defining an aggregation 'bucket' fails disclosure risk tests one or more 
    times at the row level. The safest and most consistent suppression strategy 
    is to suppress the entity if any row in the risks table involving the entity 
    is marked for suppression.

    This function performs an entity suppression test by grouping data with 
    respect to values of a nominated entity (entity_col) and examining values
    in a row suppression flag column (supp_row_col), marking each value in 
    the entity column as True if *any* of its associated rows need to be 
    suppressed, and False if none need suppression.  Results are returned in
    a two-column table, whose columns are named:
    * entity_col:  Input values of entity_col
    * 'SUPPRESS_' + entity_col:  Boolean, True if the associated entity value
                                should be suppressed, False if otherwise.
    
    :param disc_risks:  Spark DataFrame, Disclosure Risks table
    :param entity_col:  String, name of column naming entity whose values are 
                        to be tested for suppression.
    :param supp_row_col:  String, Optional, name of column containing Boolean 
                          row suppression flags.
    
    :return:  Two-column table of entity values and True/False suppression
              recommendations.
    :rtype:   Spark DataFrame.
    """
    results_col = 'SUPPRESS_' + entity_col
    test_results = disc_risks.groupBy(entity_col) \
                             .agg(F.max(supp_row_col).alias(results_col)) \
                             .orderBy(entity_col)
    
    return test_results

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```round_to_ten()``` Rounding to the nearest multiple of ten or up to ten for values less than ten.

# COMMAND ----------

def round_to_ten(df,
                 col_name,
                 rounded_col_name):
    """
    Round counts to nearest multiple of ten, up to ten if smaller than ten.

    :param df:  Spark DataFrame, input table.
    :param col_name:  String, name of column to be rounded.
    :param rounded_col_name:  String, name of column for rounded results.

    :return:  Table with an extra column containing rounded results.
    :rtype:   Spark DataFrame
    """

    result_df = df.withColumn(rounded_col_name,
                              F.when(F.col(col_name) < 10, 10) \
                               .otherwise(F.round(F.col(col_name),
                                                  scale=-1).cast(IntegerType())))

    return result_df