import math 

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from pyibl import Agent 

# Load human data: Midterm\paperCode\data\data_all_wClickInfo.csv
humanData = pd.read_csv("./paperCode/data/data_all_wClickInfo.csv")

print(humanData.head)

""" List all column names here for reference
workerId                                       1
date_time                          1589983143089
game                                           1
trial                                          1
informed                                   False
numRelevantDimensions                          2
rt                                           NaN
reward                                       NaN
ifRelevantDimension_color                   True
rewardingFeature_color                     green
selectedFeature_color                        NaN
randomlySelectedFeature_color                NaN
builtFeature_color                           NaN
ifRelevantDimension_shape                  False
rewardingFeature_shape                       NaN
selectedFeature_shape                        NaN
randomlySelectedFeature_shape                NaN
builtFeature_shape                           NaN
ifRelevantDimension_pattern                 True
rewardingFeature_pattern                   plaid
selectedFeature_pattern                      NaN
randomlySelectedFeature_pattern              NaN
builtFeature_pattern                         NaN
order_color                                    1
idxFirstClick_color                          1.0
order_shape                                    3
idxFirstClick_shape                          NaN
order_pattern                                  2
idxFirstClick_pattern                        2.0
numSelectedFeatures                          NaN
postGameAnswer_color                       green
postGameConfidence_color                      49
postGameAnswer_shape                    triangle
postGameConfidence_shape                      51
postGameAnswer_pattern             not-important
postGameConfidence_pattern                   100
"""
performanceDfColumns = ["Agent", "Reward", "Game", "Trial", "Relevant Dimensions", "Selected Features"]
performanceDf = pd.DataFrame([], columns=performanceDfColumns)

a = Agent(name="Contextual Bandit", attributes=["Shape", "Color", "Texture"], default_utility=0.5)
choices = [{"Shape":"Square"},{"Shape":"Circle"},{"Shape":"Triangle"}, 
           {"Color":"Red"},{"Color":"Blue"},{"Color":"Green"}, 
           {"Texture":"Hatched"}, {"Texture":"Dotted"},{"Texture":"Wavy"},
           {"Shape":"Square", "Color":"Red"},{"Shape":"Circle", "Color":"Red"},{"Shape":"Triangle", "Color":"Red"},
           {"Shape":"Square", "Color":"Blue"},{"Shape":"Circle", "Color":"Blue"},{"Shape":"Triangle", "Color":"Blue"},
           {"Shape":"Square", "Color":"Green"},{"Shape":"Circle", "Color":"Green"},{"Shape":"Triangle", "Color":"Green"},
           {"Shape":"Square", "Texture":"Hatched"},{"Shape":"Circle", "Texture":"Hatched"},{"Shape":"Triangle", "Texture":"Hatched"},
           {"Shape":"Square", "Texture":"Dotted"},{"Shape":"Circle", "Texture":"Dotted"},{"Shape":"Triangle", "Texture":"Dotted"},
           {"Shape":"Square", "Texture":"Wavy"},{"Shape":"Circle", "Texture":"Wavy"},{"Shape":"Triangle", "Texture":"Wavy"},
           {"Color":"Red", "Texture":"Hatched"},{"Color":"Blue", "Texture":"Hatched"},{"Color":"Green", "Texture":"Hatched"},
           {"Color":"Red", "Texture":"Dotted"},{"Color":"Blue", "Texture":"Dotted"},{"Color":"Green", "Texture":"Dotted"},
           {"Color":"Red", "Texture":"Wavy"},{"Color":"Blue", "Texture":"Wavy"},{"Color":"Green", "Texture":"Wavy"},{"Shape":"Square", "Color":"Red", "Texture":"Hatched"},{"Shape":"Circle", "Color":"Red", "Texture":"Hatched"},{"Shape":"Triangle", "Color":"Red", "Texture":"Hatched"},
            {"Shape":"Square", "Color":"Blue", "Texture":"Hatched"},{"Shape":"Circle", "Color":"Blue", "Texture":"Hatched"},{"Shape":"Triangle", "Color":"Blue", "Texture":"Hatched"},
            {"Shape":"Square", "Color":"Green", "Texture":"Hatched"},{"Shape":"Circle", "Color":"Green", "Texture":"Hatched"},{"Shape":"Triangle", "Color":"Green", "Texture":"Hatched"},
            {"Shape":"Square", "Color":"Red", "Texture":"Dotted"},{"Shape":"Circle", "Color":"Red", "Texture":"Dotted"},{"Shape":"Triangle", "Color":"Red", "Texture":"Dotted"},
            {"Shape":"Square", "Color":"Blue", "Texture":"Dotted"},{"Shape":"Circle", "Color":"Blue", "Texture":"Dotted"},{"Shape":"Triangle", "Color":"Blue", "Texture":"Dotted"},
            {"Shape":"Square", "Color":"Green", "Texture":"Dotted"},{"Shape":"Circle", "Color":"Green", "Texture":"Dotted"},{"Shape":"Triangle", "Color":"Green", "Texture":"Dotted"},
            {"Shape":"Square", "Color":"Red", "Texture":"Wavy"},{"Shape":"Circle", "Color":"Red", "Texture":"Wavy"},{"Shape":"Triangle", "Color":"Red", "Texture":"Wavy"},
            {"Shape":"Square", "Color":"Blue", "Texture":"Wavy"},{"Shape":"Circle", "Color":"Blue", "Texture":"Wavy"},{"Shape":"Triangle", "Color":"Blue", "Texture":"Wavy"},
            {"Shape":"Square", "Color":"Green", "Texture":"Wavy"},{"Shape":"Circle", "Color":"Green", "Texture":"Wavy"},{"Shape":"Triangle", "Color":"Green", "Texture":"Wavy"}
           ]

for index, row in humanData.iterrows():
    if(row['reward'] is None): continue 
    if(math.isnan(row['numSelectedFeatures'])): continue 

    if(row['numSelectedFeatures'] == "NaN"): assert(False)

    d = pd.DataFrame([["Human", row['reward'], row['game'], row['trial'], str(row['numRelevantDimensions']) + " Dimensions", row['numSelectedFeatures']]], columns=performanceDfColumns)
    performanceDf = pd.concat([performanceDf, d], ignore_index=True)
    #assert(False)

    # if 3 total: 0: 20%, 1: 40%, 2:60%, 3:80%
    # if 2 total: 0: 20%, 1: 50%, 2:80%
    # If 1 total  0: 20%, 1: 80%
    rewarding = [row['rewardingFeature_color'], row['rewardingFeature_shape'], row['rewardingFeature_pattern']]
    reward_probability = 0.2

    choice = a.choose(choices)

    for key, value in choice.items():
        if value in rewarding:
            if(row['numRelevantDimensions'] == 1):
                reward_probability += 0.6
            elif(row['numRelevantDimensions'] == 2):
                reward_probability += 0.3
            elif(row['numRelevantDimensions'] == 3):
                reward_probability += 0.2

    reward = np.random.choice([0,1], p=[1-reward_probability, reward_probability])
    a.respond(reward)

    d = pd.DataFrame([["IBL", reward_probability, row['game'], row['trial'], str(row['numRelevantDimensions']) + " Dimensions", row['numSelectedFeatures']]], columns=performanceDfColumns)
    performanceDf = pd.concat([performanceDf, d], ignore_index=True)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

#performanceDf_human = performanceDf[performanceDf["Agent"] == "Human"]
##sns.lineplot(data=performanceDf_human, x="Trial", y="Reward", hue="Relevant Dimensions", ax=axes[0,0])
#sns.lineplot(data=performanceDf_human, x="Trial", y="Selected Features", hue="Relevant Dimensions",  ax=axes[0])

performanceDf_ibl= performanceDf[performanceDf["Agent"] == "IBL"]
sns.lineplot(data=performanceDf_ibl, x="Trial", y="Reward", hue="Relevant Dimensions", ax=axes[0])
sns.lineplot(data=performanceDf_ibl, x="Trial", y="Selected Features", hue="Relevant Dimensions",  ax=axes[1])


axes[0].set_ylabel("Reward Probability", fontsize=16)
axes[1].set_ylabel("Selected Features", fontsize=16)

axes[1].get_legend().remove()

axes[1].set_xlabel("Trial", fontsize=16)
                  
axes[0].set_title("Reward by Trial and Relevant Dimensions", fontsize=18)
axes[1].set_title("Selected Features by Trial and Relevant Dimensions", fontsize=18)

plt.show()
