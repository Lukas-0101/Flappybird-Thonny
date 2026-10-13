import pickle

with open ("winner.pkl", "rb") as file:
    data = pickle.load(file)
    
print(data)