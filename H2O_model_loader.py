import torch
import torch.nn as nn

class SpectralNN(nn.Module):   #to utiize another type of model architecture, jsut define a class similar to this one
    def __init__(self):
        super(SpectralNN,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),  # Input layer: 1 feature → 64 neurons
            nn.LeakyReLU(0.01),         # Activation function
            nn.Linear(64, 64), # Hidden layer: 64 → 64 neurons
            nn.LeakyReLU(0.01),         # Activation function
            nn.Linear(64, 1)   # Output layer: 64 → 1 output
        )

    def forward(self, x):
        return self.model(x)

#load weights for each model
H2O_model1 = SpectralNN()
H2O_model1.load_state_dict(torch.load('H2O_model1_wts.pth'))
H2O_model1.eval()

CO2_model1 = SpectralNN()
CO2_model1.load_state_dict(torch.load('CO2_model1_wts.pth'))
CO2_model1.eval()

CO2_model2 = SpectralNN()
CO2_model2.load_state_dict(torch.load('CO2_model2_wts.pth'))
CO2_model2.eval()

#load variables for function
import numpy as np

x_mean = np.load('x_mean.npy')
x_std = np.load('x_std.npy')
A_mean = np.load('A_mean.npy')
A_std = np.load('A_std.npy')
A_mean2 = np.load('A_mean2.npy')
A_std2 = np.load('A_std2.npy')

def predict_greatest_absorption(input_wn):
    #normalize wavenumber for each model
    X_test = torch.tensor([[(input_wn - x_mean) / x_std]], dtype=torch.float32).unsqueeze(0)
    a1_test = torch.tensor([[(input_wn - A_mean) / A_std]], dtype=torch.float32).unsqueeze(0)
    a2_test = torch.tensor([[(input_wn - A_mean2) / A_std2]], dtype=torch.float32).unsqueeze(0)
    
    # Step 2: Use the model to predict
    H2O_model1.eval()  # set model to evaluation mode (no dropout, no gradient tracking)
    CO2_model1.eval()
    CO2_model2.eval()
    
    with torch.no_grad():  # don't compute gradients for inference
        y_pred_f = H2O_model1(X_test).item()  # NN predicts log10(absorption coefficient)
    result = y_pred_f

    if input_wn > 400 and input_wn < 1200:
        with torch.no_grad():  # don't compute gradients for inference
            b_pred1f = CO2_model1(a1_test).item()  # NN predicts log10(absorption coefficient)
        if y_pred_f <= b_pred1f:
            result = b_pred1f

    if input_wn > 1800 and input_wn < 2500:
        with torch.no_grad():  # don't compute gradients for inference
            b_pred2f = CO2_model2(a2_test).item()  # NN predicts log10(absorption coefficient)
        if y_pred_f <= b_pred2f:
            result = b_pred2f

    real_result = 10**result
    return result, real_result

