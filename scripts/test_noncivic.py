import pickle

with open('Models/model_bbmp.pkl', 'rb') as f:
    pkg = pickle.load(f)
v = pkg['vectorizer']
c = pkg['classifier']

tests = [
    ('i played a game yesterday with my friends', 'Non-Civic'),
    ('hello how are you doing today', 'Non-Civic'),
    ('who won the cricket match yesterday', 'Non-Civic'),
    ('i want to order food from swiggy', 'Non-Civic'),
    ('my favorite movie is playing in theatres', 'Non-Civic'),
    ('water fills the road when it rains', 'Drainage / SWD'),
    ('traffic jam is very high on this route', 'Traffic'),
    ('street light not working near my house', 'Street Light'),
    ('garbage not collected since 3 days', 'Garbage / Sanitation'),
    ('pothole on the main road very dangerous', 'Road Repair'),
]

print(f"{'Input':55s} | {'Predicted':22s} | {'Expected':22s} | Status")
print("-" * 115)
for text, expected in tests:
    pred = c.predict(v.transform([text]))[0]
    ok = "PASS" if pred == expected else "FAIL"
    print(f"{text:55s} | {pred:22s} | {expected:22s} | {ok}")
