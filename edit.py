import pandas as pd 

df = pd.read_pickle("Data/participants.pkl")
print(df)

assert(False)
human = df[df["Model"] == "Human"]

print(human)