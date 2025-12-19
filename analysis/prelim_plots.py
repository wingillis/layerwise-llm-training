import matplotlib.pyplot as plt

base_params = 560_988_160

approx_params = 227_659_008


plt.figure(figsize=(3, 6))
plt.bar(["Base model", "Low-rank weights"], [base_params, approx_params])
plt.ylabel("Number of Parameters")
plt.xlabel("Model type")
plt.show()
plt.savefig("params.png")