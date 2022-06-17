from joblib import dump, load
import time


scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')   


start_time = time.time()

thrust_force = nn_unit_F.predict(scaler_F.transform([(0, 0.5, 1)]))
fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(0, 0.5, 1)]))

print("--- %s seconds ---" % (time.time() - start_time))
