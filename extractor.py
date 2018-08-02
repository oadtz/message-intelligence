from pymongo import MongoClient
import os

base_path = './masterdata'

mongo = MongoClient()
db = mongo.planb_prod_main

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