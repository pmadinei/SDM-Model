# UPDATE: This Project Won "The Golden Ahwazi Young Investigator Award 2020", The Most Authoritative Award in Neuroscience and Cognitive Science in Iran.

# Introduction
State-dependent memory (SDM) is the term that explains recalling particular information occurs most accurately when a subject (humans or animals) is in the same physiological state of consciousness as it was at the time of memory formation. One way of evaluation of memory consolidation and retrieval is the Passive-Avoidance Task. It is a fear-aggravated test used to evaluate learning and memory in rodent models of CNS disorders. In this test, subjects learn to avoid an environment in which an aversive stimulus (such as a foot-shock) was previously delivered. The chamber is divided into a lit compartment and a dark compartment, with a gate between the two. Animals are allowed to explore both compartments on the first day. On the following day, they are given a mild foot shock in one of the compartments. Animals will learn to associate certain properties of the chamber with the foot shock. In order to test their learning and memory, the mice are then placed back in the compartment where no shock was delivered. Mice with normal learning and memory will avoid entering the chamber where they had previously been exposed to the shock. This is measured by recording the latency to cross through the gate between the compartments.

![Passive Avoidance Task](https://github.com/pmadinei/SDM-Model/blob/master/Results/PAT.png)

The present study was defined to construct an accurate model based on the data extracted from multiple articles that elucidated the influences of various drugs in memory retrieval under state-dependent memory. The model assists scientists in the field to appraise their new hypotheses and inspire for further experiment-design without neither animal usage nor colossal amount of time and money consumption.

# Requirement
* Python > 3.7
* Jupyter Notebook
* PyTorch 1.6.0
* Keras 2.3.0
* Scikit-Learn 0.23.2
* Numpy 1.19
* Pandas 1.1.2
* Matplotlib 3.3.1

# Implementation
The primary dataset for the present modeling was generated by combining the data extracted from five pieces of our previous researches (published in neuroscience journals) that elucidated the influences of various states of consciousness in memory retrieval. Throughout these studies, multiple drugs including morphine, nicotine, MDMA, ethanol, mecamylamine, S-WAY100135 (an 5-HT1A receptor antagonist), ACPA/AM251 (cannabinoid CB1 receptor agonist/antagonist), WIN 55,212-2 (a cannabinoid CB1/CB2 receptor agonist) and dextromethorphan were injected into the different brain sites during post-training and/or pre-test phases of passive avoidance learning task to measure memory consolidation or retrieval in male Wistar rats. Each row in the data-set represented a single experiment on an animal that a combination of drugs with specific doses was administered via systemic or intracerebral injections. The last column of each row was the latency of the passive avoidance task in the range of 0 - 300 seconds to assess memory retrieval based on injections’ combination. Trials were randomly split into train-set and test-set with a portion of 0.8-0.2; correspondingly, the models were fed with the train-set. Various machine-learning algorithms, including Linear Regression, Support-Vector Machine, Decision Tree, Random Forrest, K Nearest Neighbor, and Neural Networks were applied to establish the most accurate model to predict the latency of the passive avoidance task. The hyper-parameters of every utilized algorithm were tuned by using grid-search functions and illustrative plots.

![GS Random Forrest](https://github.com/pmadinei/SDM-Model/blob/master/Results/Random%20Forrest%20GS.png)

As an instance, the image above illustrates the effects of alterations in two hyper parameters on the MSE of Random-Forrest model. It could be easily concluded from the image that the best given hyper parameters would be 20 for Number of stimators, and 11 for Max Depth. Moreover, You can see all reports and code in jupyter notebook in eithor [HERE](https://github.com/pmadinei/SDM-Model/blob/master/SDL%20Model.ipynb) as ipynb or [HERE](https://github.com/pmadinei/SDM-Model/blob/master/Code%20in%20HTML.html) as HTML in your local computer.

# Result
The best performance of a model was observed in a compound voting model that returned the mean value of latency output from “Support-Vector,” “Decision Tree,” “Random Forrest,” and “K-Nearest Neighbor” regressors. The image bellow shows the Mean Squared Error for each constructed model that's very clear that the best model is our voting model.

![Models Comparisons](https://github.com/pmadinei/SDM-Model/blob/master/Results/Models%20Comparison.png)

The voting model attained the R2 score of 0.803 on the train-set and 0.781 on the test-set. Moreover, the model reached the Root-Mean-Squared-Error of 49.36 on the train-set and 55.39 on the test-set. 

![Regressor Reports](https://github.com/pmadinei/SDM-Model/blob/master/Results/Best%20Model%20Reports.png)

Since the latency numbers can be inferred as a binary conclusion of amnesia or solid memory, and by assuming that latency of 150 seconds or less is referring to amnesia, the model has an accuracy of 93% in predicting amnesia or solid memory based on injections’ doses.

![Classification Reports](https://github.com/pmadinei/SDM-Model/blob/master/Results/Classification%20Report.png)

Finally, the plot bellow illustrated a comparison between real latency values and the prediction of the compound voting model.

![Model Predictions](https://github.com/pmadinei/SDM-Model/blob/master/Results/Best%20Model%20Predictions.png)

### Note
Neural-Networks usually perform perfectly through Machine-Learning modelings; however, since we had a not a large dataset for the project, adding more epochs and complexity to the model just cause overfitting without any help on test-set. Accordingly, the created voting model does not gain any number from the multilayered perception that was created. The code bellow illustrates the design of the created Deep Neural-Network.

```Python
class Model(nn.Module):
    def __init__(self, class_num, act=F.relu):

        super(Model, self).__init__()

        self.layer1 = nn.Linear(1 * 20, 4000)
        self.act1 = act

        self.layer2 = nn.Linear(4000, 2000)
        self.act2 = act


        self.layer3 = nn.Linear(2000, 1000)
        self.act3 = act

        self.layer4 = nn.Linear(1000, 500)
        self.act4 = act

        self.layer5 = nn.Linear(500, 250)
        self.act5 = act

        self.layer6 = nn.Linear(250, 1)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        #Make it one-dimentional

        x = self.layer1(x)
        x = self.act1(x)

        x = self.layer2(x)
        x = self.act2(x)

        x = self.layer3(x)
        x = self.act3(x)

        x = self.layer4(x)
        x = self.act4(x)

        x = self.layer5(x)
        x = self.act5(x)

        x = self.layer6(x)
        return x
```

In this case, models like decision trees work better for smaller datasets. The image bellow illustrates a part of the designed decision tree with a max depth of 8:

![Decision Tree](https://github.com/pmadinei/SDM-Model/blob/master/Results/Decision%20Tree.png)

# Conclusion: 
Since the latency numbers can be inferred as a binary conclusion of amnesia or solid memory, and by assuming that latency of 150 seconds or less is referring to amnesia, the model has an accuracy of 93% in predicting amnesia or memory based on injections’ doses. Even though the data-set included diverse states of consciousness, additional data from other related articles can lead the model to an even more complex model that appreciates the interactions between added states more precisely to assist scientists multifacetedly.

## Poster:
This project's poster, which has been presented and became ranked 1st in BCNC Congress 2020, is attached hereunder:

![BCNC Poster](https://github.com/pmadinei/SDM-Model/blob/master/Results/SDM%20Modeling%20-%20BCNC%20Poster.jpg)

# Reference
* [CENTRAL AMYGDALA NICOTINIC AND 5-HT1A RECEPTORS MEDIATE THE REVERSAL EFFECT OF NICOTINE AND MDMA ON MORPHINE-INDUCED AMNESIA](https://www.sciencedirect.com/science/article/abs/pii/S0306452214005776)
* [Interactive effects of morphine and nicotine on memory function depend on the central amygdala cannabinoid CB1 receptor function in rats](https://www.sciencedirect.com/science/article/abs/pii/S0278584617304554)
* [Medial Prefrontal Cortical Cannabinoid CB1 Receptors Mediate Morphine–Dextromethorphan Cross State-Dependent Memory: The Involvement of BDNF/cFOS Signaling Pathways](https://www.sciencedirect.com/science/article/abs/pii/S0306452218306754)
* [Basolateral amygdala CB1 cannabinoid receptors are involved in cross state-dependent memory retrieval between morphine and ethanol](https://www.sciencedirect.com/science/article/pii/S0091305716301083)
* [Role of hippocampal and prefrontal cortical signaling pathways in dextromethorphan effect on morphine-induced memory impairment in rats](https://www.sciencedirect.com/science/article/abs/pii/S1074742715002245)
* [Stanford Behavioral and Functional Neuroscience Laboratory](https://med.stanford.edu/sbfnl/services/bm/lm/bml-passive.html)
