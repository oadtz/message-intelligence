from pymongo import MongoClient
import os

base_path = './masterdata'

mongo = MongoClient()
db = mongo.planb_prod_main

'''
with open(os.path.join(base_path, 'carriers'), 'w') as f:
    for carrier in db.carriers.find():
        if str(carrier['carrierCode']).strip() != '':
            f.write(carrier['carrierCode'] + '\n')
    f.close()


with open(os.path.join(base_path, 'stations'), 'w') as f:
    for station in db.stations.find():
        if str(station['iataCode']).strip() != '':
            f.write(station['iataCode'] + '\n')
    f.close()


with open(os.path.join(base_path, 'uldGroups'), 'w') as f:
    for uldGroup in db.uldGroups.find():
        if str(uldGroup['iataCode']).strip() != '':
            f.write(uldGroup['iataCode'] + '\n')
    f.close()



with open(os.path.join(base_path, 'message'), 'w') as f:
    for messageType in db.messageLog.find({}).distinct('messageType'):
        if len(messageType) >= 3:
            f.write(messageType + '\n')
    f.close()



with open(os.path.join(base_path, 'ulds'), 'w') as f:
    for uld in db.ulds.find():
        if str(uld['uldCode']).strip() != '':
            f.write(uld['uldCode'] + '\n')
    f.close()
'''

with open(os.path.join(base_path, 'flights'), 'w') as f:
    for fullFlightNbr in db.flights.find({}).distinct('fullFlightNbr'):
        if len(fullFlightNbr) > 3:
            f.write(fullFlightNbr + '\n')
    f.close()

    

db = mongo.planb_prod_stats
with open(os.path.join(base_path, 'flights_changes'), 'w') as f:
    f.write('error,correction\n')
    for log in db.debug_msgQuality.find({ 'chg_flightNbr': { '$ne': None } }):
        f.write(log['carrierCode'] + log['flightNbr'] + ',' + log['carrierCode'] + log['chg_flightNbr']  + '\n')
    f.close()
