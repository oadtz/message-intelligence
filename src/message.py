import os
import re

from src.config import masterdata


def load_list_from_file(file_name):
    with open(os.path.join('./masterdata', file_name)) as f:
        return f.read().splitlines()

def prepare_data(data):
    '''for r in masterdata['reserved']:
        data = data.replace(r, r.lower())
    for r in masterdata['message']:
        data = data.replace(r, r.lower())'''

    data = re.sub('<HEADER>.*?</HEADER>', 'header', data, flags=re.DOTALL)
    data = re.sub('(([0-9]{4})/([0-9]{2})|([0-9]{1,2})([A-Z]{3})([0-9]{0,4})((/([0-9]{4})){0,1}))', 'datetime', data, flags=re.DOTALL)
    data = re.sub('(([A-Z0-9]{2,3})?([0-9]{4,5})([A-Z0-9]{2,3}))', 'uld', data, flags=re.DOTALL)
    data = re.sub('((' + '|'.join(masterdata['carriers']) + ')([0-9]{3,4}))', 'flight', data, flags=re.DOTALL)
    #data = re.sub('(' + '|'.join(masterdata['stations']) + ')', 'station', data, flags=re.DOTALL)
    #data = re.sub('(' + '|'.join(masterdata['uldGroups']) + ')', 'uldgroup', data, flags=re.DOTALL)
    #data = re.sub('(' + '|'.join(masterdata['carriers']) + ')', 'carrier', data, flags=re.DOTALL)'''

    return data