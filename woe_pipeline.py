from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StructField, StructType
from pyspark.ml import Estimator, Transformer, Pipeline
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from optbinning import OptimalBinning
import pandas as pd
from pyspark.sql.functions import col, udf
import random

class WOETransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, config: dict = None):
        super(WOETransformer, self).__init__()
        self.config = config
        self.iv_values_ = {}
        self.unique_counts_ = {}

    def _transform(self, df: DataFrame) -> DataFrame:
        for col, options in self.config.items():
            binning = OptimalBinning(name=col, **options)
            pandas_df = df.select(col, "label").toPandas()
            x = pandas_df[col]
            y = pandas_df["label"]
            binning.fit(x, y)
            bin_info = binning.binning_table.build()
            woe_values = bin_info['WoE'].values

            def get_woe(val):
                for i, bin_range in enumerate(binning.splits):
                    if val <= bin_range:
                        return woe_values[i]
                return woe_values[-1]

            def calculate_iv(bin_info):
                iv = 0
                for index, row in bin_info.iterrows():
                    iv += (row['Distribution good'] - row['Distribution bad']) * row['WoE']
                return iv

            # Calculate IV for the column
            self.iv_values_[col] = calculate_iv(bin_info)
            
            # Calculate the number of unique values for the column
            self.unique_counts_[col] = df.select(col).distinct().count()
            
            woe_udf = udf(get_woe, DoubleType())
            df = df.withColumn(f"{col}_woe", woe_udf(df[col]))

        return df

    def get_iv_values(self):
        return self.iv_values_
    
    def get_unique_counts(self):
        return self.unique_counts_


class WOEEstimator(Estimator, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, config: dict = None):
        super(WOEEstimator, self).__init__()
        self.config = config

    def _fit(self, df: DataFrame):
        transformer = WOETransformer(config=self.config)
        transformer._transform(df)  # Pre-transform to calculate IVs
        return transformer


def process_woe_pipeline(df: DataFrame, config: dict):
    # Create and fit the estimator
    estimator = WOEEstimator(config=config)

    # Create the pipeline
    pipeline = Pipeline(stages=[estimator])

    # Fit the pipeline
    model = pipeline.fit(df)

    # Transform the DataFrame
    woe_df = model.transform(df)
    
    # Get the IV values and unique counts
    iv_values = model.stages[0].get_iv_values()
    unique_counts = model.stages[0].get_unique_counts()

    # Convert IV values and unique counts to a pandas DataFrame
    iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV'])
    unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Feature', 'Unique Values'])
    
    # Merge the two DataFrames
    result_df = pd.merge(iv_df, unique_counts_df, on='Feature')
    return result_df, woe_df


# Example usage:

# Initialize Spark session
spark = SparkSession.builder.appName("WOEBinning").getOrCreate()

# Sample DataFrame
schema = StructType([
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("label", DoubleType(), True)
])
data = [(random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1)) for _ in range(1000)]
df = spark.createDataFrame(data, schema)

# Configuration for optbinning
config = {
    "feature1": {"dtype": "numerical", "solver": "cp"},
    "feature2": {"dtype": "numerical", "solver": "cp"}
}

# Process the pipeline and get IV values
iv_df, woe_df = process_woe_pipeline(df, config)
print(iv_df)
woe_df.show()
