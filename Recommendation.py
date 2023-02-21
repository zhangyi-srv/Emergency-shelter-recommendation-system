import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk-17"
sc = SparkContext
spark = SparkSession.builder.appName('Recommendations').getOrCreate()

location = spark.read.csv('避难位置数据1.csv', header=True)
data = spark.read.csv('避难位置数据副本.csv', header=True)

data.show()
data.printSchema()

data = data.withColumn('userIdInt', col('userIdInt').cast('integer')).withColumn('locationIdInt',
                                                                                 col('locationIdInt').cast(
                                                                                     'integer')).withColumn('ratingInt',
                                                                                                            col('ratingInt').cast(
                                                                                                                'integer')).drop(
    'long', 'lati')
# Count the total number of ratings in the dataset
numerator = data.select("ratingInt").count()
# Count the number of distinct userIds and distinct locIds
num_users = data.select("userIdInt").distinct().count()
num_locations = data.select("locationIdInt").distinct().count()
# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_locations
# Divide the numerator by the denominator
sparsity = (1.0 - (numerator * 1.0) / denominator) * 100
print("评分矩阵 is ", "%.2f" % sparsity + "% empty.")
# Group data by userId, count ratings
userId_ratings = data.groupBy("userIdInt").count().orderBy('count', ascending=False)
userId_ratings.show()

locationId_ratings = data.groupBy("locationIdInt").count().orderBy('count', ascending=False)
locationId_ratings.show()
(train, test) = data.randomSplit([0.8, 0.2], seed=5)
print(train, test)

als = ALS(userCol="userIdInt", itemCol="locationIdInt", ratingCol="ratingInt", nonnegative=True, implicitPrefs=True,
          coldStartStrategy="drop")

type(als)

param_grid = ParamGridBuilder().addGrid(als.rank, [10, 50, 100, 150]).addGrid(als.regParam, [.01, .05, .1, .15]).build()
#             .addGrid(als.maxIter, [5, 50, 100, 200]) \

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratingInt", predictionCol="prediction")
print("Num models to be tested: ", len(param_grid))

cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

print(cv)

model = cv.fit(train)

best_model = model.bestModel

print(type(best_model))

print("  Rank:", best_model._java_obj.parent().getRank())

# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())

# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())

test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

test_predictions.show()

# 生成推荐列表
nrecommendations = best_model.recommendForAllUsers(15)
nrecommendations.limit(15).show()

nrecommendations = nrecommendations.withColumn("rec_exp", explode("recommendations")).select('userIdInt',
                                                                                             col("rec_exp.locationIdInt"),
                                                                                             col("rec_exp.rating"))
nrecommendations.limit(15).show()

nrecommendations.join(location, on='locationIdInt').filter('userIdInt = 5').show()

data.join(location, on='locationIdInt').filter('userIdInt = 5').sort('ratingInt', ascending=False).limit(15).show()

# def precision_at_k(num_users, nrecommendations, topk):
#     sum_precision = 0.0
#     num_users = len(nrecommendations.__getitem__("userIdInt"))
#     for i in range(num_users):
#         act_set = set(num_users[i])
#         pred_set = set(nrecommendations[i][:topk])
#         sum_precision += len(act_set & pred_set) / float(topk)
#
#     return sum_precision / num_users
#
#
# print(precision_at_k(num_users, nrecommendations, 5))
#
#
# def recall_at_k(num_users, nrecommendations, topk):
#     sum_recall = 0.0
#     num_users = len(nrecommendations.__getitem__("userIdInt"))
#     true_users = 0
#     for i in range(num_users):
#         act_set = set(num_users[i])
#         pred_set = set(nrecommendations[i][:topk])
#         if len(act_set) != 0:
#             sum_recall += len(act_set & pred_set) / float(len(act_set))
#             true_users += 1
#     return sum_recall / true_users

# print(recall_at_k(num_users, nrecommendations, 5))

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
