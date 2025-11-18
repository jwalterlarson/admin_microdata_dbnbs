# Databricks notebook source
# MAGIC %md
# MAGIC # Indexing Resultant Hierarchical Keep-and-Sweep "Safe" Categories to All the Potential "Unsafe" Categories They Obscure
# MAGIC We need to do this in order to tell users of published HKS output how an HKS 'safe' category is _specific to  the context in which it is constructed_; that is, though short category names may appear similar from one situation to the next, they almost certainly are not.  
# MAGIC
# MAGIC This is distinct from constructing an index showing _only_ the suppressed  behind a "safe" category.
# MAGIC
# MAGIC ## Setup
# MAGIC ### Module Import

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

from collections import OrderedDict
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
# MAGIC ## Function Definitions

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```get_all_pot_cat_combos()```:  Build Table of Potential Finest-scale Category Values with Respect to ID Columns
# MAGIC This function forms a Cartesian product of the categories hierarchy with values of a tuple of ID columns.  This is done because HKS category names must be viewed in the context in which they are computed.

# COMMAND ----------

def get_all_pot_cat_combos(hks_output_df,
                           id_cols,
                           cats_hierarchy_df):
    """
    Generate table of potential categories corresponding to input dataset.

    This function takes an HKS-suppression output dataset and, based on this 
    table's ID columns and a reference table of the categories hierarchy, 
    generate a table of all possible categories (full hierarchy).  This is 
    simply a Cartesian product of the ID columns with the reference categories 
    hierarchy table.

    :param hks_output_df:  Spark DataFrame, HKS suppression output table.
    :param id_cols:  List of String, names of ID columns for table hks_output_df.
    :param cats_hierarchy_df:  Spark DataFrame, reference table of categories. 
                               hierarchy that was used in the HKS data suppression
                               scheme that resulted in hks_output_df.
    
    :return:  Cartesian product table of the ID columns with all possible categories.
    :rtype:   Spark DataFrame.
    """
    distinct_ids = hks_output_df.select(*id_cols) \
                                .distinct() \
                                .orderBy(*id_cols)
    cats_cols = cats_hierarchy_df.columns

    all_pot_cat_combos = distinct_ids.crossJoin(cats_hierarchy_df) \
                                     .distinct() \
                                     .orderBy(*id_cols, 
                                              *cats_cols)
    
    return all_pot_cat_combos

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```map_hierarchy_cols()```:  Build Translators Between HKS Output Category Names and Reference Table Category Names
# MAGIC This function returns two ```dict``` instances:
# MAGIC 1. Translator from HKS output table category names to reference table category names
# MAGIC 2.  Translator from reference table category names to HKS output table category names

# COMMAND ----------

def map_hierarchy_cols(hks_output_col_names,
                       ref_hierarchy_col_names):
    """
    Get mappings between HKS-suppressed output column names and reference counterparts.

    This function takes two lists of strings, which are, respectively the list of 
    category hierarchy column names in an HKS-suppression output table and their 
    counterparts in a reference table.  The lists must be ordered the same way in
    terms of coarsest- to finest-scale granularity.  Obviously, the two lists must 
    also be of the same length.

    :param hks_output_col_names:  List of String, column names for category hierarchy 
                                  in an output table.
    :param ref_hierarchy_col_names:  List of String, column names for category hierarchy
                                     in reference table.
    
    :return:  Two mappings:
                1.  Translator from HKS output category names to reference category names
                2.  Translator from reference category names to HKS output category names
    :rtype:   Two Python dict instances.
    """
    hkso_to_ref = OrderedDict(zip(hks_output_col_names,
                                  ref_hierarchy_col_names))
    ref_to_hkso = OrderedDict(zip(ref_hierarchy_col_names,
                                  hks_output_col_names))
    
    return hkso_to_ref, ref_to_hkso

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```make_cat_combos_trap()```:  Construct Trap Filter to Catch Category/ID Value Combinations
# MAGIC This function builds a fitering table that traps "naked" categories at a certain level in a hierarchy, for each ID column value.  The trap captures rows from another table through an inner join (and excludes this rows through a left antijoin).

# COMMAND ----------

def make_cat_combos_trap(cat_combos,
                         id_cols,
                         ref_cat_lev_col,
                         ref_finest_cat_col,
                         hsko_cat_lev_col):
    """
    Make trapping table for collecting by hierarchy category level

    This function takes a full table of remaining potential category 
    id combinations, narrows it to the desired category hierarchy level,
    and fashions a trap that can be used to filter (by inner join) the
    HKS output table to find rows that have 'naked' category values at 
    the desired level.

    :param cat_combos:  Spark DataFrame, table of potential category 
                        combinations
    :param id_cols:  List of String, list of names of ID columns in the
                     category combinations table
    :param ref_cat_lev_col:  String, name of hierarchy level category name
                         in potential category combinations table
    :param ref_finest_cat_col:  String, name of finest-granularity reference
                                category level.
    :param hsko_cat_lev_col:  String, name of corresponding hierarchy level 
                          category name in the HKS output table (not supplied
                          to this function but the one for which this filter 
                          trap is being constructed).
    """
    if ref_cat_lev_col == ref_finest_cat_col:
        # Need the reference level category column twice under different names,
        # for both trapping and indexing purposes.
        cat_trap = cat_combos.select(*id_cols, 
                                     ref_cat_lev_col) \
                            .withColumn(hsko_cat_lev_col,
                                        F.col(ref_cat_lev_col)) \
                            .select(*id_cols, 
                                    hsko_cat_lev_col, 
                                    ref_finest_cat_col ) \
                            .sort(*id_cols, 
                                  hsko_cat_lev_col, 
                                  ref_finest_cat_col)
    else:
        # The level and finest-granularity column category names are distinct,
        # so select and manage separately.
            cat_trap = cat_combos.select(*id_cols, 
                                         ref_cat_lev_col,
                                         ref_finest_cat_col) \
                                 .withColumnRenamed(ref_cat_lev_col, 
                                                    hsko_cat_lev_col) \
                                .select(*id_cols, 
                                        hsko_cat_lev_col, 
                                        ref_finest_cat_col ) \
                                .sort(*id_cols, 
                                      hsko_cat_lev_col, 
                                      ref_finest_cat_col)
    
    return cat_trap 

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```trap_naked_cat_combos()```:  Partition Table Using Categories Trap Filter
# MAGIC This function takes input sample and trap filter tables and segments the sample table into two tables:
# MAGIC 1. A 'trapped' table that has 'naked' category values at the desired granularity level; and
# MAGIC 2. A residuals table that will be forwarded for further processing to the next indexing stage.

# COMMAND ----------

def trap_naked_cat_combos(sample_df,
                          trap_df,
                          id_cols,
                          hkso_curr_lev_cat_col,
                          ref_finest_lev_cat_col,
                          hkso_cat_cols):
    """
    Partition table using categories trap filter.

    This function partitions the input table into two talbes:
    a'trapped' table that has 'naked' category values at the 
    desired granularity level; and a residuals table that will 
    be forwarded to the next indexing stage.  This partitioning 
    is accomplished by performing join operations between the 
    sample table and the filter trap table--an inner join to 
    produce the trapped segment, and a left anti join to produce
    the residuals segment.

    :param sample_df:  Spark DataFrame, sample table from HKS output.
    :param trap_df:  Filtering table (to be used as a inner join to the
                     sample table)
    :param id_cols:  List of String, list of ID column names.
    :param hkso_curr_lev_cat_col:  String, name of current hierarchy 
                                   level column name in the sample table.
    :param ref_finest_lev_cat_col:  String, name of finest-granularity 
                                    level reference category name.
    :param hkso_cat_cols:  List of String, names hierarchy level columns 
                           in the sample table, ordered from coarsest to 
                           finest granularity.
    :return:  Two tables, in the order listed below:
      1. A 'trapped' table that has 'naked' category values at
         the desired granularity level; and
      2. A residuals table that will be forwarded to the next
         indexing stage.
    """
    hkso_finest_lev_cat_col = hkso_cat_cols[-1]
    pot_fin_hkso_col = 'POTENTIAL_' + hkso_finest_lev_cat_col + 'S'
    naked_cats_df = sample_df.join(trap_df,
                                on=[*id_cols,
                                    hkso_curr_lev_cat_col],
                                how='inner') \
                           .withColumn(pot_fin_hkso_col,
                                       F.col(ref_finest_lev_cat_col)) \
                           .select(*id_cols, 
                                   *hkso_cat_cols,
                                   pot_fin_hkso_col) \
                           .sort(*id_cols, 
                                 *hkso_cat_cols,
                                 pot_fin_hkso_col)
    residue = sample_df.join(trap_df,
                             on=[*id_cols,
                                 hkso_curr_lev_cat_col],
                             how='leftanti')  \
                       .select(*id_cols, 
                               *hkso_cat_cols) \
                       .sort(*id_cols, 
                             *hkso_cat_cols)
    
    return naked_cats_df, residue

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```prune_pot_cat_combos()```:  Prune Potential CategoryID Combinations to Eliminate Trapped, Naked Categories
# MAGIC This function fashions a filter from a naked categories table output from the function ```trap_naked_cat_combos()``` and applies this filter (a left anti-join) to the potential category combinations table to eliminate the rows having the identified category/ID column values.

# COMMAND ----------

def prune_pot_cat_combos(pot_cat_combos,
                         naked_cats_df,
                         id_cols,
                         ref_cat_cols,
                         nc_lev_col,
                         hkso_to_ref):
    """
    Backfilter category combinations to eliminate ones just identified.
    
    This function takes a table of naked categories identified by the 
    function trap_naked_cat_combos() and uses eliminates the corresponding
    entries from a table of potential category/ID values.  

    :param pot_cat_combos: Spark DataFrame, table of potential category/ID 
                           combinations to be filtered.
    :param naked_cats_df:  Spark DataFrame, table of "naked" categories 
                           to be used to prune the potential categories 
                           table.
    :param id_cols:  List of String, names of "ID" columns, common to both
                     pot_cat_combos and naked_cats_df but not part of the 
                     categories hierarchy.
    :param ref_cat_cols:  List of String, names of reference category hier-
                          archy names, ordered from coarsest- to finest-scale
                          granularity.
    :param nc_lev_col:  String, name of the "naked" category level valeus 
                        identified in the table naked_cats_df.
    :param hkso_to_ref:  dict, translator between HKS category hierarchy 
                         level names and their reference counterparts.

    :return:  Table of remaining potential category/ID combinations.
    :rtype:   Spark DataFrame.
    """
    ref_lev_col = hkso_to_ref[nc_lev_col]
    nc_backfilter = naked_cats_df.select(*id_cols,
                                         nc_lev_col) \
                                 .withColumnRenamed(nc_lev_col,
                                                    ref_lev_col)
    remaining_pccs = pot_cat_combos.join(nc_backfilter,
                                         on=[*id_cols,
                                             ref_lev_col],
                                         how='leftanti') \
                                   .select(*id_cols,
                                           *ref_cat_cols) \
                                   .sort(*id_cols,
                                         ref_lev_col)
    return remaining_pccs

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```naked_cat_seg_format()```:  Format Index Table from Naked Categories/ID Combinations Segment
# MAGIC This function converts a naked category/ID combinations segment into the layout used in the final index product.

# COMMAND ----------

def naked_cat_seg_format(naked_cats_df,
                         id_cols,
                         hkso_cat_cols):
    """
    Format index table from naked categories segment.

    This function takes a naked categories table and re-formats it
    as an index of potential fine-scale categories.

    :param naked_cats_df:  Spark DataFrame, output table from the 
                           function trap_naked_cat_combos().
    :param id_cols:  List of String, names of ID columns.
    :param hkso_cat_cols:  List of String, column names of HKS 
                           category level hierarchy, ordered 
                           from coarsest- to finest-level 
                           granularity.
    
    :return:  Index table showing how an HKS-suppression finest
              granularity level relates to a list of all of the
              potential finest-level category values it represents.
    :rtype:   Spark DataFrame
    """
    finest_hkso_col = hkso_cat_cols[-1]
    pot_fin_hkso_col = 'POTENTIAL_' + finest_hkso_col + 'S'
    pot_fin_hkso_count_col = pot_fin_hkso_col + '_COUNT'
    naked_cats_df = naked_cats_df.groupBy(*id_cols, *hkso_cat_cols) \
                                 .agg(F.collect_list(pot_fin_hkso_col).alias(pot_fin_hkso_col)) \
                                 .withColumn(pot_fin_hkso_count_col,
                                             F.size(F.col(pot_fin_hkso_col))) \
                                 .select(*id_cols, *hkso_cat_cols,
                                         pot_fin_hkso_count_col,
                                         pot_fin_hkso_col) \
                                 .sort(*id_cols, *hkso_cat_cols)
    
    return naked_cats_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```index_bin_covering_output()```  Index Segment of HKS Outut Generated by Bin-covering
# MAGIC The bin-covering scheme used in the final stage of HKS presents a challenge to the indexing infrastructure applicable to the hierarchy levels:  Bin-covering does not generate standard category values but instead _lists_ of category values _enclosed in square brackets._  In order to index these fusion categories, we need to decompose the into their consituent coarse category values and then index these category values with respect to the select fine-scale category values they _could_ represent, and then relate them back to the bin-covering-generated fusion categories.  This function performs this task.

# COMMAND ----------

def index_bin_covering_output(bco_df,
                              pot_cat_combos,
                              id_cols,
                              bco_cat_cols,
                              ref_cat_cols):
    """
    Index Segment of HKS Outut Generated by Bin-covering

    :param bco_df:  Spark DataFrame, bin-covering output table.
    :param pot_cat_combos:  Spark DataFrame, table of potential 
                            remaining category/ID value combinations.
    :param id_cols:  List of String, ID column names, which must match 
                     for both of the above tables.
    :param bco_cat_cols:  List of String, List of names of the category 
                          hierarchy columns in the bin-covering output 
                          table, ordered from coarsest- to finest-scale
                          granularity.
    :param ref_cat_cols:  List of String, List of names of the category 
                          hierarchy columns in the potential category/ID 
                          values table, ordered from coarsest- to finest-
                          scale granularity.

    :return:  Index relating bin-covering output fusion categories to a
              list of potential finest-scale categories present in the 
              fusion category. 
    :rtype:   Spark DataFrame
    """

    residue = bco_df.withColumn(ref_cat_cols[0],
                                F.regexp_replace(F.col(bco_cat_cols[0]),
                                                 '\[|\]', '')) \
                    .withColumn(ref_cat_cols[0],
                                F.split(F.col(ref_cat_cols[0]), ', ')) \
                    .withColumn(ref_cat_cols[0],
                                F.explode(F.col(ref_cat_cols[0])))
    pot_bco_finest_cats = 'POTENTIAL_' + bco_cat_cols[-1] + 'S'
    pot_bco_finest_cats_count = pot_bco_finest_cats + '_COUNT'
    bc_index = residue.join(pot_cat_combos.select(*id_cols,
                                                  ref_cat_cols[0],
                                                  ref_cat_cols[-1]),
                            on=[*id_cols,
                                ref_cat_cols[0]],
                            how='inner') \
                      .drop(ref_cat_cols[0]) \
                      .withColumnRenamed(ref_cat_cols[-1],
                                         pot_bco_finest_cats) \
                      .sort(*id_cols, 
                            *bco_cat_cols,
                            pot_bco_finest_cats)
    bc_index = bc_index.groupBy(*id_cols, 
                                *bco_cat_cols) \
                       .agg(F.collect_list(pot_bco_finest_cats).alias(pot_bco_finest_cats)) \
                       .withColumn(pot_bco_finest_cats_count,
                                   F.size(F.col(pot_bco_finest_cats))) \
                       .select(*id_cols, 
                               *bco_cat_cols,
                               pot_bco_finest_cats_count,
                               pot_bco_finest_cats) \
                       .sort(*id_cols, 
                             *bco_cat_cols)

    return bc_index

# COMMAND ----------

# MAGIC %md
# MAGIC ### ```index_hks()```:  Construct an Index to Potential Finest-granularity Categories in HKS Output

# COMMAND ----------

def index_hks(hks_output_df,
              ref_cats_hier_df,
              id_cols,
              hkso_cat_cols,
              ref_cat_cols):
    """
    Index HKS output to potential finest-granularity categories covered

    This function is the top-level driver that takes output from the HKS
    data suppression scheme and constructs an index that relates each of 
    the HKS "channels" to its respective list of potential finest-level-
    granularity categories in the categories hierarchy.  This is to make
    concrete the possible contents of these automatically-generated HKS
    "channels"/"fusion caegories."  

    :param hks_output_df:  Spark DataFrame, HKS data suppression output.
    :param ref_cats_hier_df:  Spark DataFrame, reference categories 
                              hierarchy table.
    :param id_cols:  List of String, names of the ID columns in the HKS
                     output table.
    :param hkso_cat_cols:  List of String, hierarchy category names in 
                           the HKS output table, ordered from coarsest-
                           to finest-scale granularity.
    :param ref_cat_cols:  List of String, hierarchy category names in 
                          the reference categories hierarchy table, 
                          ordered from coarsest- to finest-scale 
                          granularity.
    
    :return:  Index table relating all HKS output channels to their 
              respective lists of all potential finest-granlarity 
              categories they represent.
    :rtype:   Spark DataFrame
    """
    # Create translator dicts between HKS output and reference 
    # hierarchy names.
    hkso_to_ref, ref_to_hkso = map_hierarchy_cols(hkso_cat_cols,
                                                  ref_cat_cols)
    
    # Get list of all potential category/ID column combinations.
    pot_cat_combos = get_all_pot_cat_combos(hks_output_df,
                                            id_cols,
                                            ref_cats_hier_df)
    
    # Create empty list to hold index output segments as they
    # are generated.
    index_segments = []

    # Loop over hierarchy levels, from finest- to coarsest-
    # scale granularity to index "naked" category values at
    # each level as they are identified.
    res = hks_output_df
    for ref_cat_lev in reversed(ref_cat_cols):

        # Make naked categories trap filter for this level.
        cat_trap = make_cat_combos_trap(pot_cat_combos,
                                        id_cols,
                                        ref_cat_lev,
                                        ref_cat_cols[-1],
                                        ref_to_hkso[ref_cat_lev])
        
        # Use the trap to partition the potential category/ID 
        # table into two segments:
        #   1.  "naked" categories at this hierarchy level.
        #   2.  residues to be processed further.
        trapped, res = trap_naked_cat_combos(res,
                                             cat_trap,
                                             id_cols,
                                             ref_to_hkso[ref_cat_lev],
                                             ref_cat_cols[-1],
                                             hkso_cat_cols)
        
        # Prune potential category/ID combinations based 
        # on those just trapped.
        pot_cat_combos = prune_pot_cat_combos(pot_cat_combos,
                                              trapped,
                                              id_cols,
                                              ref_cat_cols,
                                              ref_to_hkso[ref_cat_lev],
                                              hkso_to_ref)
        
        # Format trapped segment for inclusion in index and
        # append to segments list.
        trapped = naked_cat_seg_format(trapped,
                                       id_cols,
                                       hkso_cat_cols)
        index_segments.append(trapped)

    # Finally, index the residuals, which at this point 
    # should all be the result of the HKS bin-covering 
    # refinement stage.
    bc_ind = index_bin_covering_output(res,
                                       pot_cat_combos,
                                       id_cols,
                                       hkso_cat_cols,
                                       ref_cat_cols)
    
    # Union of index segments and sort.
    hks_index = bc_ind
    for seg in index_segments:
        hks_index = hks_index.unionByName(seg)
    
    hks_index = hks_index.sort(*id_cols, 
                               *hkso_cat_cols)

    return hks_index
