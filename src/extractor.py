from pymongo import MongoClient
import os

base_path = os.path.join(os.path.dirname(__file__), '../masterdata')

mongo = MongoClient()

with open(os.path.join(base_path, 'carriers'), 'w') as f:
    db = mongo.planb_prod_main
    for carrier in db.carriers.find():
        if str(carrier['carrierCode']).strip() != '':
            f.write(carrier['carrierCode'] + '\n')
    f.close()


with open(os.path.join(base_path, 'stations'), 'w') as f:
    db = mongo.planb_prod_main
    for station in db.stations.find():
        if str(station['iataCode']).strip() != '':
            f.write(station['iataCode'] + '\n')
    f.close()


with open(os.path.join(base_path, 'uldGroups'), 'w') as f:
    db = mongo.planb_prod_main
    for uldGroup in db.uldGroups.find():
        if str(uldGroup['iataCode']).strip() != '':
            f.write(uldGroup['iataCode'] + '\n')
    f.close()


with open(os.path.join(base_path, 'message'), 'w') as f:
    db = mongo.planb_prod_main
    for messageType in db.messageLog.find({}).distinct('messageType'):
        if len(messageType) >= 3:
            f.write(messageType + '\n')
    f.close()


with open(os.path.join(base_path, 'ulds'), 'w') as f:
    db = mongo.planb_prod_main
    for uld in db.ulds.find():
        if str(uld['uldCode']).strip() != '':
            f.write(uld['uldCode'] + '\n')
    f.close()


with open(os.path.join(base_path, 'flights'), 'w') as f:
    db = mongo.planb_prod_stats
    f.write('error,correction\n')
    for log in db.debug_msgQuality.find({ 'chg_flightNbr': { '$ne': None } }):
        if log['carrierCode'] and log['chg_flightNbr']:
            f.write(log['carrierCode'] + log['flightNbr'] + ',' + 
                    log['carrierCode'] + log['chg_flightNbr']  + '\n')
    
    db = mongo.planb_prod_main
    for flight in db.flights.find({}).distinct('fullFlightNbr'):
        if len(flight) > 3:
            f.write(flight + ',' + flight + '\n')
    f.close()