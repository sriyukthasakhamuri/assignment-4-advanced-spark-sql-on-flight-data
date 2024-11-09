from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate scheduled and actual travel times
    flights_df = flights_df.withColumn("scheduled_travel_time", F.col("ScheduledArrival") - F.col("ScheduledDeparture"))
    flights_df = flights_df.withColumn("actual_travel_time", F.col("ActualArrival") - F.col("ActualDeparture"))
    
    # Calculate the discrepancy as the absolute difference
    flights_df = flights_df.withColumn("discrepancy", F.abs(F.col("scheduled_travel_time") - F.col("actual_travel_time")))
    
    # Define a window specification to rank flights by discrepancy for each carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(F.desc("discrepancy"))

    # Apply row_number to rank flights by discrepancy for each carrier
    flights_df = flights_df.withColumn("Rank", F.row_number().over(window_spec))
    
    # Filter to get the top-ranked discrepancy for each carrier
    largest_discrepancy = flights_df.filter(F.col("Rank") == 1).select(
        "FlightNum", "CarrierCode", "Origin", "Destination",
        "scheduled_travel_time", "actual_travel_time", "discrepancy"
    )

    # Join with carriers_df to add Carrier Name
    largest_discrepancy = largest_discrepancy.join(
        carriers_df, "CarrierCode", "left"
    ).select(
        "FlightNum", "CarrierName", "Origin", "Destination", 
        "scheduled_travel_time", "actual_travel_time", "discrepancy"
    )

    # Show the result as a table
    largest_discrepancy.show()

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Calculate departure delay in minutes
    flights_df = flights_df.withColumn(
        "departure_delay_minutes",
        (F.col("ActualDeparture").cast("long") - F.col("ScheduledDeparture").cast("long")) / 60
    )
    
    # Calculate the standard deviation of departure delays for each airline
    consistent_airlines = flights_df.groupBy("CarrierCode").agg(
        F.stddev("departure_delay_minutes").alias("stddev_departure_delay"),
        F.count("FlightNum").alias("flight_count")
    ).filter(F.col("flight_count") > 100).orderBy("stddev_departure_delay")
    
    # Rank carriers by their standard deviation of departure delays (lower stddev = more consistent)
    consistent_airlines = consistent_airlines.withColumn(
        "rank", F.row_number().over(Window.orderBy("stddev_departure_delay"))
    )

    # Join with carriers_df to get carrier names
    consistent_airlines = consistent_airlines.join(
        carriers_df, "CarrierCode", "left"
    ).select(
        "rank", "CarrierName", "flight_count", "stddev_departure_delay"
    )

    # Show the result as a table
    consistent_airlines.show()

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Check if flights are canceled by adding a derived column
    flights_df = flights_df.withColumn(
        "IsCancelled",
        F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0)
    )

    # Calculate cancellation rate and total flights by origin-destination pair
    canceled_routes = flights_df.groupBy("Origin", "Destination").agg(
        F.avg("IsCancelled").alias("cancellation_rate"),
        F.count("FlightNum").alias("total_flights")  # Count total flights for each origin-destination pair
    )

    # Filter routes with more than 50 flights
    canceled_routes = canceled_routes.filter(F.col("total_flights") > 50).orderBy(F.desc("cancellation_rate"))

    # Alias the airports DataFrame to differentiate between origin and destination
    origin_airports = airports_df.alias("origin_airport")
    destination_airports = airports_df.alias("destination_airport")

    # Join with airports_df to get airport names and cities for origin and destination
    canceled_routes = canceled_routes.join(
        origin_airports, canceled_routes["Origin"] == origin_airports["AirportCode"], "left"
    ).join(
        destination_airports, canceled_routes["Destination"] == destination_airports["AirportCode"], "left"
    ).select(
        F.col("origin_airport.AirportName").alias("Origin_AirportName"),
        F.col("origin_airport.City").alias("Origin_City"),
        F.col("destination_airport.AirportName").alias("Destination_AirportName"),
        F.col("destination_airport.City").alias("Destination_City"),
        "cancellation_rate"
    )

    # Create a Window spec to rank routes by cancellation_rate in descending order
    window_spec = Window.orderBy(F.desc("cancellation_rate"))

    # Apply the rank function
    canceled_routes_with_rank = canceled_routes.withColumn(
        "rank", F.row_number().over(window_spec)
    )

    # Show the result as a table
    canceled_routes_with_rank.show()

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Extract hour from ActualDeparture to determine time of day
    flights_df = flights_df.withColumn("DepartureHour", F.hour("ActualDeparture"))

    # Define time of day based on DepartureHour
    flights_df = flights_df.withColumn(
        "time_of_day",
        F.when((F.col("DepartureHour") >= 6) & (F.col("DepartureHour") < 12), "Morning")
         .when((F.col("DepartureHour") >= 12) & (F.col("DepartureHour") < 18), "Afternoon")
         .when((F.col("DepartureHour") >= 18) & (F.col("DepartureHour") < 24), "Evening")
         .otherwise("Night")
    )

    # Calculate average departure delay for each CarrierCode and time_of_day
    carrier_performance = flights_df.groupBy("CarrierCode", "time_of_day").agg(
        F.avg(F.col("ActualDeparture").cast("long") - F.col("ScheduledDeparture").cast("long")).alias("avg_departure_delay")
    )

    # Create a window specification to rank by average departure delay within each time_of_day
    window_spec = Window.partitionBy("time_of_day").orderBy(F.desc("avg_departure_delay"))

    # Rank carriers based on average departure delay within each time of day
    carrier_performance_with_rank = carrier_performance.withColumn(
        "rank", F.row_number().over(window_spec)
    )

    # Join with carriers_df to add carrier names, using alias to avoid ambiguity
    carrier_performance_with_rank = carrier_performance_with_rank.alias("cp").join(
        carriers_df.alias("cd"), F.col("cp.CarrierCode") == F.col("cd.CarrierCode")
    ).select(
        F.col("cp.CarrierCode").alias("CarrierCode"),
        F.col("cd.CarrierName").alias("CarrierName"),
        F.col("cp.time_of_day").alias("time_of_day"),
        F.col("cp.avg_departure_delay").alias("avg_departure_delay"),
        F.col("cp.rank").alias("rank")  # Include the rank in the output
    )

    # Show the result as a table
    carrier_performance_with_rank.show()

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()