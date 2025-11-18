# Databricks notebook source
# MAGIC %md
# MAGIC # A Library for Performing Factor-based Aggregation Analyses
# MAGIC Frequently we are faced with doing aggregations based on one or more factors, and sometimes combined with temporal aggregation.  This library implements kernels for doing this.  This notebook also loads libraries for perturbation and rounding operations.

# COMMAND ----------

import pyspark.sql.functions as F 
from pyspark.sql.types import StringType
from pyspark.sql.functions import rank, col, row_number, when
from pyspark.sql.functions import count, lit, max, min, mean 
from pyspark.sql.functions import lpad
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

import numpy as np
import pandas as pd

import seaborn as sns
sns.set_theme(style='white')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Perturbation Functions Library

# COMMAND ----------

# MAGIC %run "<path-to-folder>/PerturbationTools_Library"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Rounding Library

# COMMAND ----------

# MAGIC %run "<path-to-folder>/Rounding_Library"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Library for Disclosure Risk Evaluation

# COMMAND ----------

# MAGIC %run "<path-to-folder>/DisclosureRiskFunctions_Library"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions
# MAGIC ### ```time_factor_counts()```:  Get Demographic Factors Table

# COMMAND ----------

def time_factor_counts(df,
                       time_col,
                       factors_cols,
                       counts_col='COUNT',
                       perturb_and_round=False,
                       perturbation_seed=0,
                       rounding_base=10,
                       compute_percentages=False):
    """
    Compute Demographic Factor(s) Breakdown vs. Time

    This function takes an input table and performs a single-
    or multiple demographics factor aggregation with respect 
    to time.  The result is a table showing the factor counts
    as a timeseries.

    :param df:  Spark DataFrame, input table.
    :param time_col:  String, name of column containing time
                      coordinate values.
    :param factors_cols:  List of String, names of column(s) 
                          containing demographic factor(s) 
                          over which counts are to be computed.
    :param counts_col:  String, Optional, default value 'COUNT'.
                        Name of column in which computed counts 
                        will be stored in the output table.
    :param perturb_and_round: Boolean, Optional, default False.
                              Perturb and round counts?  If True,
                              counts are perturbed using values 
                              from the uniform discrete distribution
                              {-2, -1, 0, 1, 2} and rounded up/down 
                              at random to the nearest whole-number 
                              multiples of the base specified in the
                              argument rounding_base.
    :param perturbation_seed:  Numeric, Optional, default value 0.
                               Random number generator seed for the
                               perturbation scheme.
    :param rounding_base:  Numeric, Optional, base to which rounding
                           is to be performed.  Default value 10.
    :param comupute_percentages:  Boolean, Optional, default False.
                                  If True, perturbation and rounding,
                                  if requested, is performed before 
                                  the percentages are computed.
    
    :return:  Table with demographic breakdown counts/percentages 
              versus time.
    :rtype:   Spark DataFrame.
    """
    time_counts_df = df.groupBy(time_col,
                                *factors_cols) \
                       .agg(F.count("*").alias(counts_col)) \
                       .sort(time_col,
                             *factors_cols)
    
    if perturb_and_round:
        time_counts_df = additive_perturb_from_list(time_counts_df,
                                                    [counts_col],
                                                    seed=perturbation_seed)
        counts_perturbed_col = counts_col + '_PERTURBED'
        time_counts_df = random_round_up_down_to_nearest_multiples(time_counts_df,
                                                                   [counts_perturbed_col],
                                                                   base=rounding_base,
                                                                   random_seed=perturbation_seed)
    
    if compute_percentages:
        counts_perturbed_rounded_col = counts_perturbed_col + '_ROUNDED'
        counts_pert_round_total_col = counts_perturbed_rounded_col + '_TOTAL'
        counts_pert_round_percent = counts_perturbed_rounded_col + '_PERCENT'
        time_totals_df = time_counts_df.groupBy(time_col) \
                                       .agg(F.sum(counts_perturbed_rounded_col)\
                                             .alias(counts_pert_round_total_col))
        time_counts_df = time_counts_df.join(time_totals_df,
                                             time_col,
                                             'left') \
                                       .filter(F.col(counts_perturbed_rounded_col) != 0) \
                                       .withColumn(counts_pert_round_percent,
                                                   F.round(100. * (F.col(counts_perturbed_rounded_col) /
                                                                   F.col(counts_pert_round_total_col)), 
                                                           2))
    
    time_counts_df = time_counts_df.drop('MONOTONE_ID') \
                                   .sort(time_col,
                                         *factors_cols)
    
    return time_counts_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```time_factor_counts_disc_risks()```:  Time Factor Counts with Disclosure Risk Calculations Included

# COMMAND ----------

def time_factor_counts_disc_risks(df,
                                  time_col,
                                  factors_cols,
                                  agg_func=F.count,
                                  agg_col="*",
                                  agg_stat_col='COUNT',
                                  contrib_id_col='ABN_HASH_TRUNC',
                                  total_stat_threshold=10,
                                  num_contribs_col='TOTAL_CONTRIBUTORS',
                                  num_contribs_threshold=5,
                                  test_dominance=True,
                                  first_contrib_count_col='CONTRIB_1ST_COUNT',
                                  first_contrib_pct_threshold=50.,
                                  second_contrib_count_col='CONTRIB_2ND_COUNT',
                                  largest_two_contribs_pct_threshold=67.,
                                  sorting_keys=None,
                                  perturb_and_round=False,
                                  perturbation_seed=0,
                                  rounding_base=10,
                                  compute_percentages=False):
    """
    Compute Demographic Factor(s) Breakdown vs. Time

    This function takes an input table and performs a single-
    or multiple demographics factor aggregation with respect 
    to time.  The result is a table showing the factor counts
    as a timeseries.

    :param df:  Spark DataFrame, input table.
    :param time_col:  String, name of column containing time
                      coordinate values.
    :param factors_cols:  List of String, names of column(s) 
                          containing demographic factor(s) 
    :param agg_func:  Spark function (from pyspark.sql.functions), Optional,
                      Aggregation function.
    :param agg_col:  String, Optional, name of column to which aggregation 
                     function is to be applied.
    :param agg_stat_col:  String, Optional, default value 'COUNT', name of 
                          the column to be used for resulting aggregation 
                          statistic.  
    :param contrib_id_col:  String, Optional, name of column containing 
                            contributor identification information.
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
    :param perturb_and_round: Boolean, Optional, default False.
                              Perturb and round counts?  If True,
                              counts are perturbed using values 
                              from the uniform discrete distribution
                              {-2, -1, 0, 1, 2} and rounded up/down 
                              at random to the nearest whole-number 
                              multiples of the base specified in the
                              argument rounding_base.
    :param perturbation_seed:  Numeric, Optional, default value 0.
                               Random number generator seed for the
                               perturbation scheme.
    :param rounding_base:  Numeric, Optional, base to be used for 
                           rounding.  Default value is 10.
    :param comupute_percentages:  Boolean, Optional, default False.
                                  If True, perturbation and rounding,
                                  if requested, is performed before 
                                  the percentages are computed.
    
    :return:  Table with demographic breakdown counts/percentages 
              versus time.
    :rtype:   Spark DataFrame.
    """
    grouping_cols = [time_col] + factors_cols
    time_counts_df = get_dominant_contributors(df,
                                               grouping_cols,
                                               agg_func=agg_func,
                                               agg_col=agg_col,
                                               agg_stat_col=agg_stat_col,
                                               contrib_id_col=contrib_id_col)
    counts_risk_df = apply_disclosure_risk_rules(time_counts_df,
                                                 total_stat_col=agg_stat_col,
                                                 total_stat_threshold=total_stat_threshold,
                                                 num_contribs_col=num_contribs_col,
                                                 num_contribs_threshold=num_contribs_threshold,
                                                 test_dominance=test_dominance,
                                                 first_contrib_count_col=first_contrib_count_col,
                                                 first_contrib_pct_threshold=first_contrib_pct_threshold,
                                                 second_contrib_count_col=second_contrib_count_col,
                                                 largest_two_contribs_pct_threshold=largest_two_contribs_pct_threshold,
                                                 sorting_keys=sorting_keys)
    
    if perturb_and_round:
        counts_risk_df = additive_perturb_from_list(counts_risk_df,
                                                    [agg_stat_col],
                                                    seed=perturbation_seed)
        counts_perturbed_col = agg_stat_col + '_PERTURBED'
        counts_risk_df = random_round_up_down_to_nearest_multiples(counts_risk_df,
                                                                   [counts_perturbed_col],
                                                                   base=rounding_base,
                                                                   random_seed=perturbation_seed)
    
    if compute_percentages:
        counts_perturbed_rounded_col = counts_perturbed_col + '_ROUNDED'
        counts_pert_round_total_col = counts_perturbed_rounded_col + '_TOTAL'
        counts_pert_round_percent = counts_perturbed_rounded_col + '_PERCENT'
        time_totals_df = counts_risk_df.groupBy(time_col) \
                                       .agg(F.sum(counts_perturbed_rounded_col)\
                                             .alias(counts_pert_round_total_col))
        counts_risk_df = counts_risk_df.join(time_totals_df,
                                             time_col,
                                             'left') \
                                       .filter(F.col(counts_perturbed_rounded_col) != 0) \
                                       .withColumn(counts_pert_round_percent,
                                                   F.round(100. * (F.col(counts_perturbed_rounded_col) /
                                                                   F.col(counts_pert_round_total_col)), 
                                                           2))
    
    counts_risk_df = counts_risk_df.drop('MONOTONE_ID') \
                                   .sort(time_col,
                                         *factors_cols)
    
    return counts_risk_df
