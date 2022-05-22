# POLiTweets dataset 
Authors: Joanna Baran, Michał Kajstura, Maciej Zi ́ołkowski, Krzysztof Rajda

Heuristically annotated collection of Polish tweets
designed for the task of political profiling in a multi-party setup.

More information can be read in our LREC 2022 paper from PoliticalNLP workshop.

## Dataset description
POLiTweets is the first publicly open Polish dataset for political affiliation classification in a multi-party setup. Most labels were obtained using a novel universal method of semi-automated political leaning discovery.  It relies on a heuristical data annotation procedure based on user's likes distribution across posts from political parties' representatives.
In total, this dataset consists of over 147k tweets from almost 10k Polish-writing users annotated heuristically and almost 40k tweets from 166 users annotated manually. The last ones create test splits for evaluation purposes.

The data is made available as six CSV files containing TweetID instead of explicit text, according to Twitter's guidelines about redistributing their content for scientific purposes.
Files description:
1) train.csv - 132 435 tweets with heuristically generated labels, 9452 users,
2) validation.csv - 14 716 tweets with heuristically generated labels, 4867 users,
3) manual_test.csv - 29 960 tweets manually annotated, 133 users,
4) heuristics_test.csv - 29 960 tweets with heuristically generated labels, 133 users,
5) manual_ambiguous_test.csv - 9 757 manually annotated tweets from 33 users whose political views are hard to define.
