
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import matplotlib.pyplot as plt
import plotly.offline as py
import csv
import os

class Job_fit:
    
    def __init__(self,file_path):
        print("STARTING")
		
		#define essential variables
        self.file = file_path
        self.df = pd.read_csv(self.file)
        self.temp_CSV() # call temp_CSV method to generate temporary CSV file
        
        self.df_new = pd.read_csv('temp.csv') #read csv
        self.m = Prophet(weekly_seasonality=False,yearly_seasonality=False) # remove unwanted methods of fbprophet
        self.m.fit(self.df_new) # fit the data into the model
        self.days = 0 # days required to job fit
        self.value = 0 # positive value percentage
        self.daily_progress() # call the daily_progress function to plot the data

    #temp CSV file generator
    def temp_CSV(self):
        print("CREATING TEMP FILE")
        date = self.df['Date'] # get date from Date column
        prog = self.df['P'] # get positive values from P column

        with open('temp.csv','w',newline='') as file: #open new temporary csv file
            writer = csv.writer(file)
            writer.writerow(['ds','y'])
            
            for i in range(len(date)):
                writer.writerow([date[i],prog[i]]) #write data into temp csv
            
        
    def predict_and_plot(self):
        print("PREDICTING")
        self.df_new.head()
		
		# this model will apply repeatedly until the positiveness value reaches to 100.
		# this while loop will do that and calculate the number of iteration required 
		# then it will decide the number of days(decide according to the iteration count) needed to job fit
        while(self.value <= 100):
            self.days = self.days + 7
            self.future = self.m.make_future_dataframe(periods=self.days)
            self.future.tail()
            forecast = self.m.predict(self.future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            f_cast = forecast['yhat'].tail(1)
            f_cast = f_cast.tolist()
            self.value = f_cast[0]
            print("Days: {}--- Positive value:{}".format(self.days,self.value))
            
            
            
        self.m.plot(forecast) #plotting the data
        self.m.plot_components(forecast) #plotting the data
        py.init_notebook_mode() # initialize the notebook mode
        fig = plot_plotly(self.m, forecast)  # This returns a plotly Figure
        py.iplot(fig)
        print("Number of days to fit the job: {} Days".format(self.days))
        self.remove_temp_files()
        
	# this method uses to remove the generated temp csv file
    def remove_temp_files(self):
        if os.path.exists("temp.csv"):
            os.remove('temp.csv') # remove the temporary csv file from local directory
            
    def daily_progress(self):
        print("DAILY PROGRESS")
        #Positiveness
		#plot the daily positive values of a user
        data = pd.read_csv(self.file, index_col=['Date'], parse_dates=['Date'])
        plt.figure(figsize=(18, 6))
        plt.plot(data.P)
        plt.title('Progress:{}'.format('Positive'))
        plt.grid(True)
        plt.show() # display the plot
        
        #Negativeness
		#plot the daily negative values of a user
        data = pd.read_csv(self.file, index_col=['Date'], parse_dates=['Date'])
        plt.figure(figsize=(18, 6))
        plt.plot(data.N)
        plt.title('Progress:{}'.format('Negative'))
        plt.grid(True)
        plt.show() # display the plot
        
        

t = Job_fit('emotion1.csv')
t.predict_and_plot()

