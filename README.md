# ```admin_microdata_dbnbs```
Databricks "Library" Notebooks for Working with Administrative Data

## About

These are Databricks notebooks, saved as Python source code, that provide useful functions for working with administrative microdata.  They have run successfully in ABS' DataLab environment and have been used successfully to create published datasets.

The codes are organized by functionality and cover many tasks commonly encountered when working with unit microdata.  All of the libraries are implemented in PySpark and are capable of running at scale on a Spark cluster.

## Functionalities

### Identifying Individuals

* ```DegenerateSpineIDs_Library.py ```:  When administrative data are combined from records spanning multiple agencies, individuals are identified by the data integration authority and assigned a unique id called a _linkage spine_.  Ideally, for data from a given agency, there should be a 1-1 correspondence between linkage spine and the identity key that agency assigns the individual.  When a linkage spine corresponds to multiple id keys for a single agency, this can compromise analyses that rely on tracking an individual's trajectory in the data.  Such ambiguous linkage spines can be viewed as _degenerate_.  This notebook provides functions for finding and eliminating degenerate linkage spines from a sample.

### Remapping Categorical Variables Using Concordances

* ```ApplyConcordanceAsDict.py```.  Uses supplied concordance tables (either created from files or read from a database table) to re-map variables.

### Finding and Filling Gaps in Individuals' Longitudinal Records

* ```GapDetection+Filling_Library.py```.  Finds data gaps in temporally-ordered records for each individual and offers capabilities for filling the gaps either with placeholder values or inferential patching from records immediately adjacent to the gap.

### Disclosure Risk

#### Identifying Disclosure Risks

* ```DisclosureRiskFunctions_Library.py```.  Functions for evaluating cell counts/values for varous types of disclosure risks including:  Low count values; Low Contributor counts; and Dominance statistics.  All thresholds can be user-configured as parameters supplied to functions.  Default values correspond to ABS disclosure risk rules.

#### Mitigating Disclosure Risks

##### Perturbation

* ```RandomGeneratorTools_Library.py```.  Generate Spark DataFrame of random variates using numpy's random generator facility.  At present, only numpy.random.choice() is supported.  The point is to generate random variates in a way that _can_ be reproducible under Spark parallelism.  This is accomplished by specifying a seed to the random generator and building the random variates table on the master node and assigning a monotonically increasing index that can be joined onto a sample Spark DataFrame that also has a monotonically increasing index.

* ```PerturbationTools_Library.py```.  Takes an input Spark DataFrame, adds a monotonically increasing index to it (if one is not already present), and joins a set of indexed random variates onto it and applys the desired transformation (at present only additive perturbations are supported).

##### Rounding

* ```Rounding_Library.py```.  Rounding to arbitrary integer or non-integer (presumably rational number) base.

##### Data Suppression Exploiting Categorical Data Hierarchies

* ```HKS_Library.py```.  Hierarchical Keep-and-Sweep (HKS), an adaptive data suppression scheme for categorical data.  Has been used to render "safe" counts aggregated by ANZSCO occupation codes, ANZSIC industry codes, and ASCED field of education codes.  Can be used for any hierarchical data.  Is capable of working with the types of disclosure risks diagnosed by functions in the ```DisclosureRiskFunctions_Library.py``` notebook.

* ```HKS_Indexing_Library.py```.  Builds an inverse index of output from ```HKS_Library.py``` to enumerate suppressed finer-scale categorical values that _might_ be rolled into each suppression class.  The point of this is to demonstrate how difficult--if not impossible!--the suppression scheme has rendered the problem of trying to reverse-engineer the suppression classes to reveal "unsafe" values.

#### Risk-Aware Aggregation

* ```FactorAggs_Library.py```.  Aggregation by demographic (or other categorical factors) and time.  Used to evaluate risks involved with slicing/dicing data by multiple factors.
