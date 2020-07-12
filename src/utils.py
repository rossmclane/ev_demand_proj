import random
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class vehicle:
    """
    Vehicle class to simulatie ICE driving
    """
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.remaining_miles = 259
        self.state_of_charge = 100
        self.range = 259
        self.odometer_reading = trajectory.df.odo_read.min()
        self.max_odo = trajectory.df.odo_read.max()
        self.gps = [trajectory.df.decr_lat[0], trajectory.df.decr_lng[0]]
        self.time = trajectory.df.index.min()
        self.charge_events = []

    def __repr__(self):
        return f"""Remaining Miles: {self.remaining_miles}\
                \nGPS: {self.gps} \nTime: {self.time} \nOdometer Reading: {self.odometer_reading}
                   """

    def drive(self, km):
        """
        Drive method movest the vehicle x km and sets the GPS, Time, Odometer_Reading, and Remaining_Miles accordingly
        """

        self.odometer_reading += km
        self.remaining_miles -= km
        self.state_of_charge -= (km / self.range)

        self.trajectory.df['odo_diff'] = abs(self.trajectory.df['odo_read'] - self.odometer_reading)

        closest_ping = self.trajectory.df[self.trajectory.df['odo_diff'] == self.trajectory.df['odo_diff'].min()]
        self.time = closest_ping.index
        self.gps = [closest_ping.decr_lng[0], closest_ping.decr_lat[0]]

    def calculate_next_charge(self):
        
        # Import charging events
        charges = pd.read_csv('../data/raw/charges_derived_joined_charger.csv')
        
        # Randomly sample from distribution of start_soc
        charges['start_soc_prob'] = charges.start_soc / charges.start_soc.sum()
        start_soc = np.random.choice(charges.start_soc, 1, p=list(charges['start_soc_prob']).reverse())
        km_to_next_charge = (1 - start_soc) * self.range

        while km_to_next_charge > self.remaining_miles:
           # redo the random sampling 
           km_to_next_charge = (1 - random.choice(start_socs)) * self.range
           print('km to next charge > remaining miles')

        return km_to_next_charge

    def train_kWh_model(self):
        # Import charging events
        charges = pd.read_csv('../data/raw/charges_derived_joined_charger.csv')
        
        model = LinearRegression()
        x,y = np.array(charges.start_soc).reshape((-1, 1)), np.array(charges.delta_soc).reshape((-1, 1))
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, y_train)
        
        # Set the trained model as a paramter of the object
        self.prediction_model = regr
        

    def charge(self): 
        
        print(self.state_of_charge)

        # predict delta_soc based on state_of_charge
        delta_soc_pred = self.prediction_model.predict(np.array(self.state_of_charge).reshape(-1, 1))

        print(delta_soc_pred)
        miles_of_charge = delta_soc_pred * self.range
        
        print(miles_of_charge)

        while miles_of_charge > (self.range - self.remaining_miles):
            delta_soc_pred = self.prediction_model.predict(np.array(self.state_of_charge).reshape(-1, 1))
            miles_of_charge = delta_soc_pred * self.range

        self.remaining_miles += miles_of_charge
        self.state_of_charge += delta_soc_pred
        self.charge_events.append(self.gps)

def run_simulation(vehicle):
    
    # Train the model
    vehicle.train_kWh_model()

    flag = True
    while flag:

        # Calculate distance to next charging event
        km_to_next_charge = vehicle.calculate_next_charge()

        # if the vehicicle isn't at the end of it's trajectory continue
        if (vehicle.odometer_reading + km_to_next_charge) > vehicle.max_odo:
            flag = False
        else:
            vehicle.drive(km_to_next_charge)
            vehicle.charge()
