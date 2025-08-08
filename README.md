# Neural-Network-HAPI-Project
You can view all my code and important visualization in the ipynb file, please go there to learn more!

This project develops a set of neural networks that learns to predict absorption coefficients of atmospheric gases as a function of wavenumber, based on data from the high-resolution spectroscopic database HITRAN. It also includes a method to compute the average maximum absorption coefficient across gases at each wavenumber.

Traditional line-by-line radiative transfer models are computationally expensive. A trained neural network can rapidly estimate absorption coefficients with reasonable accuracy, enabling fast approximations for use in climate models, satellite instrument design, and retrieval algorithms. By determining the greatest absorption coefficient at each wavenumber, the model reveals which gas dominates radiative transfer at that spectral region. This is essential for understanding how Earth's atmosphere absorbs and emits energy, which is critical for climate modeling.

In order to utilize the base code yourself, simply downloaded all files and run the ipynb. The second to last cell is the method to compute averge maximum absorption coefficient gases at each wavenumber. 

It is important to note all data is extracted from the database at 260k and 500hPa. This represent the average conditions of the mid-trophosphere, often used as a reference for radiative transfer calculations. If you wish to use the method for other conditions, adjust the parameters in the import_HAPI_data function, which houses the absoprtionCoefficient_Lorentz function. The environment parameters can be adjusted to your preference. 
