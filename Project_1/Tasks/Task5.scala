// Import the needed libraries
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler


// Read the fermentation.csv file saved in my local pc
val ferm = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("/Users/aaryan/Desktop/6. Semester/TDT4305/TDT4305/Project_1/fermentation.csv");

// Create a new dataframe with only the columns we are interested in
val all_columns = Array("Glucose concentration", "Acetate concentration", "Ethanol concentration", "Specific oxygen uptake rate", "Specific carbon dioxide evolution rate", "Biomass");
val df = ferm.select(all_columns.map(col): _*);

// Transform the features columns in into a single vector
val features = Array("Glucose concentration", "Acetate concentration", "Ethanol concentration", "Specific oxygen uptake rate", "Specific carbon dioxide evolution rate");
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
val assembledDF = assembler.transform(df);

// Create a final dataframe with just the transformed features and the target
val final_df = assembledDF.withColumnRenamed("Biomass", "target").select("features", "target");

// Split the data into training and test sets
val splitRatio = Array(0.7, 0.3);
val Array(train_df, test_df) = final_df.randomSplit(splitRatio);

// To verify that the split was successful
println("Train DataFrame size: " + train_df.count());
println("Test DataFrame size: " + test_df.count());

// Create a Linear Regression model and fit it to the training data and predict the test data
val lr = new LinearRegression().setLabelCol("target").setFeaturesCol("features");
val model = lr.fit(train_df);
val preds = model.transform(test_df);

// Evaluate the model
val evaluator = new RegressionEvaluator().setLabelCol("target").setPredictionCol("prediction").setMetricName("rmse");
val rmse = evaluator.evaluate(preds);

println(s"Root Mean Squared Error (RMSE) on test data = $rmse");