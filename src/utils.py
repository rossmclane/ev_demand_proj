import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import geopandas as gpd
from fiona.crs import from_epsg
from datetime import timedelta


class Vehicle:
    """
    A class to represent a vehicle in EV charging simulation

    Attributes
    ----------
    identifier : string
        Unique vehicle identifier
    trajectory : movingpandas object
        The movingpandas trajectory object describing the vehicles movement
    range : int
        Range in miles of vehicle on full battery
    state_of_charge : int
        Simulated battery state of charge of vehicle
    odometer_reading : float
        Reading of odometer at current vehicle state
    time : Datetime
        Current time of vehicle state
    gps : list
        List of GPS [Lat, Lon] for current vehicle state
    charge_events : list
        List of gps lists representing simulated charging events
    max_odo : float
        Maximum odometer reading in the trajectory
    battery_capacity : int
        55 kWh for Bolt EV
   """

    def __init__(self, trajectory):
        self.identifier = trajectory.df.hashed_vin.iloc[0]
        self.trajectory = trajectory
        self.range = 259
        self.state_of_charge = 100
        self.odometer_reading = trajectory.df.odo_read.min()
        self.time = pd.to_datetime(trajectory.df.element_time_local).min()
        self.gps = [trajectory.df.decr_lat[0], trajectory.df.decr_lng[0]]
        self.charge_events = []
        self.max_odo = trajectory.df.odo_read.max()
        self.battery_capacity = 55

    def __repr__(self):
        representation = (
            f"State of Charge: {self.state_of_charge} \n"
            f"Odometer Reading: {self.odometer_reading} \n"
            f"Time: {self.time} \n"
            f"GPS: {self.gps} \n"
        )
        return representation

    def drive(self, miles):
        """
        Moves the vehicle forward the input number of miles and alters vehicle state

        Parameters
        ----------
        self: object
            Python Class Object

        miles : float
            Number of miles to move the vehicle forward
        """

        # Increase odometer and decrease SOC accordingly
        self.odometer_reading += miles
        self.state_of_charge -= (100 * (miles / self.range))

        # Look for the closest odometer reading in the trajectory data
        self.trajectory.df['odo_diff'] = abs(self.trajectory.df['odo_read'] - self.odometer_reading)
        closest_ping = self.trajectory.df[self.trajectory.df['odo_diff'] == self.trajectory.df['odo_diff'].min()]

        # Set time and gps to that in the trajectory data
        self.time = pd.to_datetime(closest_ping.element_time_local.iloc[0])

        self.gps = [closest_ping.decr_lng[0], closest_ping.decr_lat[0]]

    def calculate_miles_to_next_charge(self, start_soc):
        """
        Calculate where the next ICE vehicle charging event will be by randomly sampling from the EV start_soc distribution

        Parameters
        ----------
        self : object
            Python Class Object
        start_soc : int
            The randomly sampled starting state of charge

        Returns
        ----------
        miles_to_next_charge: float
            The predicted number of miles the vehicle will go from its current state to its next charging event

        """

        # If the soc_required according to the sample is more than exists in the tank, re-sample until
        # This requirement is satisfied

        # Calculate the miles to the next charge according to this start_soc
        miles_to_next_charge = ((100 - start_soc) / 100) * self.range

        return miles_to_next_charge

    def charge(self, model):
        """
        Trigger a charging event with kWh defined by the prediction model for the vehicle and change the state accordingly

        Parameters
        ----------
        self: object
            Python Class Object
        model: Scikit-Learn Linear Model
            Trained Prediction Model
        """

        # Predict delta_soc based on state_of_charge
        delta_soc_predicted = model.predict(np.array(self.state_of_charge).reshape(-1, 1))

        # While the delta_soc_predicted is more than the remaining battery capacity, re-predict

        while delta_soc_predicted > (100 - self.state_of_charge):
            print('SOC capacity needed is lower than predicted delta SOC')
            delta_soc_predicted = model.predict(np.array(self.state_of_charge).reshape(-1, 1))

        # Create a new charging event
        energy = (delta_soc_predicted[0][0] / 100) * self.battery_capacity
        event = ChargingEvent(self.gps, self.time, delta_soc_predicted[0][0], self.state_of_charge, energy)

        # Increase the state_of_charge by delta_soc
        self.state_of_charge += delta_soc_predicted[0][0]

        # Append the ChargingEvent to the charge_events list
        self.charge_events.append(event)


class Simulation:
    """
    The simulation object could have a function call run
    In run, the model is trained and then passed to each vehicle object
    For that vehicle to use the prediction in its simulation

    Parameters
    ____________

    prediction_model : scikit-learn linear_model
        The trained delta_soc prediction model
    ev_charging_events : pandas dataframe
        The EV charging event data set used for random sampling and prediction
    vehicles : list
        List of vehicle objects for prediction of charging

    """

    def __init__(self, vehicles):
        self.ev_charging_events = pd.read_csv('../data/raw/charges_derived_joined_charger.csv')
        self.prediction_model = None
        self.vehicles = vehicles

    def train_model(self):
        """
        Train the machine learning model to predict kWh

        Parameters
        ----------
        self: object
            Python Class Object
        """
        # Create input features and target variable
        x, y = np.array(self.ev_charging_events.start_soc).reshape((-1, 1)), np.array(
            self.ev_charging_events.delta_soc).reshape((-1, 1))

        # Split features to train/test on
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # Create linear model
        model = LinearRegression()

        # Train the model using the training sets
        model.fit(x_train, y_train)

        # Set the trained model as a parameter of the object
        self.prediction_model = model

    def randomly_sample_start_soc(self, vehicle):
        """
        Train the machine learning model to predict kWh

        Parameters
        ----------
        self: object
            Python Class Object
        vehicle: Vehicle Object
            Vehicle Object

        Returns
        ----------
        start_soc: int
            The randomly sampled start state of charge
        """

        # Calculate start_soc probabilities
        self.ev_charging_events[
            'start_soc_prob'] = self.ev_charging_events.start_soc / self.ev_charging_events.start_soc.sum()

        # Randomly sample from that distribution
        start_soc = np.random.choice(self.ev_charging_events.start_soc, 1,
                                     p=list(self.ev_charging_events['start_soc_prob']).reverse())[0]

        # If the SOC needed to make this drive is more than the current state, re-sample
        soc_needed = 100 - start_soc
        while soc_needed >= vehicle.state_of_charge:
            start_soc = self.randomly_sample_start_soc(vehicle)
            soc_needed = 100 - start_soc

        return start_soc

    def run(self):
        """
        Train the machine learning model and run the simulation for each vehicle

        Parameters
        ----------
        self: object
            Python Class Object

        """
        self.train_model()

        # For each vehicle in the simulation set
        for vehicle in self.vehicles:

            # Randomly sample start_soc and calculate miles to next charge
            start_soc = self.randomly_sample_start_soc(vehicle)
            miles_to_next_charge = vehicle.calculate_miles_to_next_charge(start_soc)

            # While the vehicle is not at its maximum odometer reading
            while (vehicle.odometer_reading + miles_to_next_charge) <= vehicle.max_odo:
                # Drive and Charge, otherwise move to next vehicle
                vehicle.drive(miles_to_next_charge)

                vehicle.charge(self.prediction_model)

                # Recalculate next charging event
                start_soc = self.randomly_sample_start_soc(vehicle)
                miles_to_next_charge = vehicle.calculate_miles_to_next_charge(start_soc)


class ChargingEvent:
    """
    Class which defines a charging event

    Attributes
    ----------
    gps : list
        [Lat, Lon] in EPSG:4326
    time : Datetime
        Datetime Object
    delta_soc : int
        Change in SOC of charging event
    start_soc : int
        Starting state of charge
    energy : int
        Energy of charging event given battery size
    """

    def __init__(self, gps, time, delta_soc, start_soc, energy):
        self.gps = gps
        self.start_time = time
        self.delta_soc = delta_soc
        self.start_soc = start_soc
        self.energy = energy
        self.duration = timedelta(hours=(((delta_soc / 100) * 55) / 50))
        self.end_time = self.start_time + self.duration

    def __repr__(self):
        representation = (
            f"GPS: {self.gps} \n"
            f"Start Time: {self.start_time} \n"
            f"End Time: {self.end_time} \n"
            f"Delta SOC: {self.delta_soc} \n"
            f"Start SOC: {self.start_soc} \n"
            f"Energy: {self.energy} \n"
        )
        return representation


class ChargingEvents:
    """
    Class which defines all charging events and provides methods for visualization

    Attributes
    ----------
    event_list : list
        List of ChargingEvent Objects as defined above
    """

    def __init__(self, event_list):
        self.event_list = event_list

    def to_geopandas(self):
        """
        Convert the ChargingEvents Object to a pandas dataframe of charging events

        Parameters
        ----------
        self: object
            Python Class Object

        Returns
        ----------
        df : geo pandas dataframe
            GeoPandas dataframe of charging events
        """
        df = gpd.GeoDataFrame()
        for charging_events in self.event_list:

            # iterate through the list and add
            for event in charging_events:
                latitude, longitude = event.gps[0], event.gps[1]
                event_df = pd.DataFrame([[latitude, longitude, event.delta_soc, event.energy, event.start_soc, event.start_time, event.end_time]],
                                        columns=['latitude', 'longitude', 'delta_soc', 'energy', 'start_soc', 'start_time', 'end_time'])

                df = df.append(
                    gpd.GeoDataFrame(event_df, geometry=gpd.points_from_xy(event_df.latitude, event_df.longitude),
                                     crs=from_epsg(4326)))
        return df

    def to_hourly(self):

        # Create charges_gdf
        charge_events_gdf = self.to_geopandas()

        # Create unique identifier
        charge_events_gdf['ID'] = [i for i in range(0, charge_events_gdf.shape[0])]

        # Create dataframe by minutes in this datetime range
        start = charge_events_gdf['start_time'].min()
        end = charge_events_gdf['end_time'].max()
        index = pd.date_range(start=start, end=end, freq='1T')
        df2 = pd.DataFrame(index=index, columns= \
            ['minutes', 'ID', 'latitude', 'longitude', 'delta_soc', 'energy'])

        # Spread the events across minutes
        for index, row in charge_events_gdf.iterrows():
            df2['minutes'][row['start_time']:row['end_time']] = 1
            df2['ID'][row['start_time']:row['end_time']] = row['ID']
            df2['latitude'][row['start_time']:row['end_time']] = row['latitude']
            df2['longitude'][row['start_time']:row['end_time']] = row['longitude']
            df2['delta_soc'][row['start_time']:row['end_time']] = row['delta_soc']
            df2['energy'][row['start_time']:row['end_time']] = row['energy']

        # Clean up dataframe
        df2 = df2[df2.ID.notna()]
        df2['time'] = df2.index
        df2['hour'] = df2['time'].apply(lambda x: x.hour)

        # GroupBy ID and hour
        df3 = df2.groupby(['ID', 'hour']).agg({'minutes': 'count', 'time': 'first', 'latitude': 'first','longitude': 'first', 'delta_soc': 'first',
                                               'energy': 'first'}).reset_index()

        # Recreate time index
        df3['time'] = df3['time'].apply(lambda x: pd.datetime(year=x.year, month=x.month, day=x.day, hour=x.hour))
        df3.set_index('time', inplace=True)
        df3['time'] = df3.index

        # Spread energy and delta_soc
        sums = df3.groupby('ID').agg({'minutes': 'sum'}).rename(columns={'minutes': 'minutes_sum'})
        df4 = pd.merge(df3, sums, on='ID')
        df4.set_index('time', inplace=True)
        df4['delta_soc'] = df4['delta_soc'] * (df4['minutes'] / df4['minutes_sum'])
        df4['energy'] = df4['energy'] * (df4['minutes'] / df4['minutes_sum'])
        df5 = df4.drop(columns=['minutes', 'minutes_sum', 'hour']).sort_values(by='ID')

        return df5

