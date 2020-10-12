from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("PayTM Challenge") \
                    .config("spark.some.config.option", "some-value") \
                    .getOrCreate()

weather_data = spark.read.csv('data/2019', header=True, inferSchema=True)
station_list = spark.read.csv('stationlist.csv', header=True, inferSchema=True)
country_list = spark.read.csv('countrylist.csv', header=True, inferSchema=True)

# UDFs to extract year, month, date and tornados from weather data
def extract_tornados(ind):
    data = str(ind)
    if len(data) > 4:
        return data[4] == "1"
    else:
        return False

extract_year = udf(lambda date: int(str(date)[:4]), IntegerType())
extract_month = udf(lambda date: int(str(date)[4:6]), IntegerType())
extract_day = udf(lambda date: int(str(date)[6:]), IntegerType())
extract_tornado = udf(extract_tornados, BooleanType())

# Remove columns that will not be needed for the computation purpose by selecting required columns.
weather_data = weather_data.select('STN---', 'TEMP', 'WDSP', 'FRSHTT', 'YEARMODA')

weather_data = (weather_data.withColumn(
                            'YEAR', extract_year('YEARMODA')
                            ).withColumn(
                            'MONTH', extract_month('YEARMODA')
                            ).withColumn(
                            'DAY', extract_day('YEARMODA')
                            ).withColumn(
                                "Tornadoes", extract_tornado('FRSHTT')
                            ).drop('YEARMODA').drop('FRSHTT')
                )

# Join country list with station list
country_stations = country_list.join(station_list,
                  country_list.COUNTRY_ABBR == station_list.COUNTRY_ABBR,
                 'inner').select(country_list.COUNTRY_FULL,
                                 country_list.COUNTRY_ABBR,
                                 station_list.STN_NO)

# Join country station dataset with weather dataset.
final_weather_data = country_stations.join(weather_data,
                      country_stations.STN_NO == weather_data['STN---'],
                     'inner').drop(weather_data['STN---'])


def get_mean_average_sorted(df, mean_col, group_by_col, missing_val):
    """
    Given dataframe it applies group by operation on group_by_col and
    computes mean of the mean_col. Returns data sorted in descending order
    on final average of mean_col.

    Arguments:
        dataframe spark.DataFrame
            Dataframe on which the mean operation is to be applied on
        mean_col string
            Column on which mean is to be computed on
        group_by_col list(string)
            String of column by which the group by is supposed to happen
        missing_val int
            Missing value replacement (used to filter data)

    Returns:
        aggregated_df spark.DataFrame
            Aggregated spark dataframe sorted in descending order
    """
    aggregated_df = (df.filter(
                    df[mean_col] != missing_val
                ).groupby(
                    group_by_col
                ).mean(
                    mean_col
                ).withColumnRenamed(
                    f"avg({mean_col})",f"mean_{mean_col}"
                )
           )
    return aggregated_df.orderBy(aggregated_df[f"mean_{mean_col}"].desc())



"""
Which country had the hottest average mean temperature over the year?

Approach:
Missing temprature data should be filtered out. And average temprature should be computed using groupby query of year and country and applying aggregation function called mean

Uses get_mean_average_sorted function to do the computatoion.
"""
average_temp = get_mean_average_sorted(
                    final_weather_data,
                    'TEMP',
                    ['YEAR', 'COUNTRY_FULL'],
                    9999.9
                )
country = average_temp.select('COUNTRY_FULL').take(1)
print("==============================================================================")
print(f"Country with hottest mean average temprature is {country[0].COUNTRY_FULL}")


"""
Which country had the most consecutive days of tornadoes/funnel cloud
formations?

I do not have enough time for this one. But my approach would be to use window and lag
and row_count function to compute the consecutive streak.

Once the streak is computed return the one with highest streak. Simple max function can
be used to do so.
"""
# window = Window.partitionBy("COUNTRY_FULL").orderby("YEARMODA")
# processed_data = (final_weather_data.withColumn(
#                     "lag_date",  lag("YEARMODA", 1, "").over(w1))
#                         .withColumn(
#                     "lag_val", lag("Tornadoes", 1, "").over(w1))
#                   .withColumn("index", sum("jump").over(w1))
#                   .withColumn('RowCount', )
#                  )



"""
Which country had the second highest average mean wind speed over the year?

Approach:
Missing wind speed data should be filtered out. Then average winds spped over the year can be computed as groupby query of year and country and applying aggregation function called mean.

Uses get_mean_average_sorted function to do the computatoion.
"""
average_wind_speed = get_mean_average_sorted(
                    final_weather_data,
                    'WDSP',
                    ['YEAR', 'COUNTRY_FULL'],
                    999.9
                )
country = average_wind_speed.select('COUNTRY_FULL').take(2)
print(f"Country with second highest mean wind speed over the year is {country[1].COUNTRY_FULL}")
print("==============================================================================")