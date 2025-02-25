import pandas as pd
data_path=(r'C:\Users\abhin\Desktop\resturant_rating\Dataset .csv')
data=pd.read_csv(data_path)
print(data.head())

from ydata_profiling import ProfileReport
profile=ProfileReport(data,explorative=True)
profile.to_file('output_EDA.html')
