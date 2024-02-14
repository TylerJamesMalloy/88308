from pyibl import Agent 
import random 
import numpy as np 

seed = 1234
rng = np.random.default_rng(seed=seed)
random.seed(seed) # Used by pyibl model     
a = Agent(name="IBL Model", attributes=["Button"], default_utility=lambda _: 0.5, noise=0) 

q = [0.5, 0.5]
alpha = 0.8

for x in range(2):
    choice = a.choose([{"Button":"Left"}, {"Button":"Right"}], details=True)
    left  = rng.beta(100,100) # E = 0.5000, SD = 0.0353
    right = rng.beta(11,10)   # E = 0.5238, SD = 0.1065
    print(choice[0])
    for option in ["Left", "Right"]:
        if(choice[1][0]['choice'] == {'Button': option}):
            print("Option ", option, " value ", round(choice[1][0]['blended_value'], 2))
            q[0] = ((1-alpha) * q[0]) + alpha * (left)
        if(choice[1][1]['choice'] == {'Button': option}):
            print("Option ", option, " value ", round(choice[1][1]['blended_value'], 2))
            q[1] = ((1-alpha) * q[1]) + alpha * (right)
    
    print("Q-table: ", round(q[0],2), ",", round(q[1],2))
     
    reward = left if choice[0] == "Left" else right

    a.respond(reward)

# [{'choice': 'Right', 'blended_value': 0.9994213771697248, 'retrieval_probabilities': [{'utility': 2.1524723198315336, 'retrieval_probability': 0.021861437004562292}, {'utility': 2.1665081094477325, 'retrieval_probability': 0.027194227920177668}, {'utility': 1.9710906970945614, 'retrieval_probability': 0.0182179642140886}, {'utility': 1.017685925318652, 'retrieval_probability': 0.03262406885367484}, {'utility': 1.6342733642266354, 'retrieval_probability': 0.09488490720293766}, {'utility': 0.8060602135053317, 'retrieval_probability': 0.023621588272708653}, {'utility': 1.5572119654652803, 'retrieval_probability': 0.012931573988092093}, {'utility': 0.7378455522707096, 'retrieval_probability': 0.013387643524619972}, {'utility': 0.5748715441596662, 'retrieval_probability': 0.03599143471061597}, {'utility': 0.8335003706573982, 'retrieval_probability': 0.7192851543085221}]}, {'choice': 'Left', 'blended_value': 0, 'retrieval_probabilities': []}]
choice = a.choose([{"Button":"Left"}, {"Button":"Right"}], details=True)
for option in ["Left", "Right"]:
    if(choice[1][0]['choice'] == {'Button': option}):
        print("Option ", option, " value ", round(choice[1][0]['blended_value'], 2))
    if(choice[1][1]['choice'] == {'Button': option}):
        print("Option ", option, " value ", round(choice[1][1]['blended_value'], 2))