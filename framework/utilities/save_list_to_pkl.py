import pickle

F1_GCD = [-2170725,
-2309035,
-2013087,
-2174662,
-2006038,
-2312065,
-2161855,
-2300262,
-2164683,
-2163946,
-2307504,
-2012137]

F2_GCD = [73132.74881505,
77949.53004898,
70787.05068136,
74620.80651651,
69328.3795621,
85829.63124772,
71217.78708358,
74940.05462288,
72965.07618374,
71564.93997058,
76855.01635435,
69675.53244911]
with open("Database/Results_Multi_Optim/functions/case8_cost_pareto.pkl", "wb") as f:   #Pickling
    pickle.dump(F2_GCD, f)