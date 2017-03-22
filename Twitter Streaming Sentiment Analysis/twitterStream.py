from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")

    counts = stream(ssc, pwords, nwords, 100)
    print counts
    make_plot(counts)

def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    posX=[]
    posY=[]
    negX=[]
    negY=[]
	
    count=1
    for i in counts:
	if len(i)!=0:	
		posX.append(count)
	        posY.append(i[0][1])
		negX.append(count)
	        negY.append(i[1][1])
	        count=count+1
	
    line1, =plt.plot(posX,posY,marker="o",label="Positive",color="g")
    line2, =plt.plot(negX,negY,marker="o",label="Negative",color="r")
    plt.xlabel('Time Step')
    plt.ylabel('Word Count')
    plt.title('Basic Twitter Sentiment Analytics')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.show()

def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    with open(filename) as f:
    	data = f.read().splitlines()
    return data

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE

    def getWordType(word):
	if word in pwords:
		return "Positive"
	elif word in nwords:
		return "Negative"

    def updateFunction(newValues, runningCount):
	if runningCount is None:
		runningCount = 0
	return sum(newValues, runningCount)

    def removeNone(x):
	if x[0] in ["Positive","Negative"]:
		return True
	else:
		return False

    words = tweets.flatMap(lambda line:line.split(" "))
    pairs = words.map(lambda word: (getWordType(word), 1))
    wordCounts = pairs.reduceByKey(lambda x, y: x + y)
    wordCounts = wordCounts.filter(removeNone)
    runningCounts = pairs.filter(removeNone)
    runningCounts = pairs.updateStateByKey(updateFunction)
    runningCounts.pprint()

    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    wordCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts

if __name__=="__main__":
    main()
