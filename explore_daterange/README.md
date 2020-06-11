# daterange investigation

The goal was to try to identify a date range where the pharmacy building in influx matched the data in skyspark. The issue is that there are constant changes in SkySpark.

`generaldaterange_investigate.ipynb` is a data exploration notebook that concludes using data starting on 2020-01-08 onwards will be fine and recommends the date range 2020-01-08 through 2020-06-01 (end date isn't too important as long as it is consistent in the project)

`PharmacySkySparkvsInfluxData.ipynb` is focused on identifying how much of the data in influxDB can be salvaged and moved into the new structure when the database is updated. Essentially what can still be linked back to SkySpark and what can not. This was created to help Jiachen avoid having to requery all of the datafrom SkySpark.

