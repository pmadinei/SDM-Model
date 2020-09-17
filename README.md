# Introduction
State-Dependent Memory (SDM) is the term that explains remembering a particular memory will occur most accurately when an individual is in the same state of consciousness as it was at the time of memory formation. The present study was defined to construct an accurate model based on the data extracted from multiple articles that elucidated the influences of various drugs in memory retrieval under state-dependent memory. The model assists scientists in the field to appraise their new hypotheses and inspire for further experiment-design without neither animal usage nor colossal amount of time and money consumption.

# Requirement
* Python > 3.7
* Jupyter Notebook
* PyTorch 1.6.0
* Keras 2.3.0
* Scikit-Learn 0.23.2
* Numpy 1.19
* Pandas 1.1.2
* Matplotlib 3.3.2

# Implementation
Throughout the researches mentioned above, various drugs - including Saline, Morphine, Nicotine, MDMA, Vehicle, Mecamylamine, (S)WAY100135, ACPA, AM251, Dextromethorphan, Ethanol, and WIN 55.212-2 - were injected pre-test or post-train in either Central Amigdala, Basolateral Amygdala, medial Prefrontal Cortext, or Peritoneum. A data-set has been created to combine all the information from the researches. Each row in the data-set represented a single experiment on a male Wistar rat that a combination of drugs with specific doses was injected through its body. Additionally, the last column of each particular experiment was the latency of the passive avoidance task in the range of 0 and 300 seconds to assess memory retrieval based on injections’ combination. Trials with detailed information were randomly split into train-set and test-set with a portion of 0.8-0.2. As a result, the models were fed with 80% of the data to be evaluated on both training rows and the rows that the models perceived for the first time in testing. Various machinelearning algorithms were applied to establish the most accurate model in the pursuit of predicting the latency of the passive avoidance task based on injection doses. The hyper-parameters of every utilized algorithm, including Linear Regression, Support-Vector Machine, Decision Tree, Random Forrest, K Nearest Neighbor, and Neural Network, were tuned by using grid-search functions.

Moreover, You can see all reports and code in jupyter notebook in eithor [HERE](https://github.com/pmadinei/SDM-Model/blob/master/SDL%20Model.ipynb) as ipynb or [HERE](https://github.com/pmadinei/SDM-Model/blob/master/Code%20in%20HTML.html) as HTML in your local computer.

# Result
The best performance of a model was observed in a compound voting model that returned the mean value of latency output from “Support-Vector Regressor,” “Decision Tree Regressor,” “Random Forrest Regressor,” and “K-Nearest Neighbor Regressor.” The voting model attained the R2 score of 0.803 on the train-set and 0.781 on the test-set. Moreover, the model reached the Root-Mean-Squared-Error of 49.36 on the train-set and 55.39 on the test-set. Since the latency numbers can be inferred as a binary conclusion of amnesia or solid memory, and by assuming that latency of 150 seconds or less is referring to amnesia, the model has an accuracy of 93% in predicting amnesia or solid memory based on injections’ doses.

# Reference
* [CENTRAL AMYGDALA NICOTINIC AND 5-HT1A RECEPTORS MEDIATE THE REVERSAL EFFECT OF NICOTINE AND MDMA ON MORPHINE-INDUCED AMNESIA](https://www.sciencedirect.com/science/article/abs/pii/S0306452214005776)
* [Interactive effects of morphine and nicotine on memory function depend on the central amygdala cannabinoid CB1 receptor function in rats](https://www.sciencedirect.com/science/article/abs/pii/S0278584617304554)
* [Medial Prefrontal Cortical Cannabinoid CB1 Receptors Mediate Morphine–Dextromethorphan Cross State-Dependent Memory: The Involvement of BDNF/cFOS Signaling Pathways](https://www.sciencedirect.com/science/article/abs/pii/S0306452218306754)
* [Basolateral amygdala CB1 cannabinoid receptors are involved in cross state-dependent memory retrieval between morphine and ethanol](https://www.sciencedirect.com/science/article/pii/S0091305716301083)
* [Role of hippocampal and prefrontal cortical signaling pathways in dextromethorphan effect on morphine-induced memory impairment in rats](https://www.sciencedirect.com/science/article/abs/pii/S1074742715002245)
