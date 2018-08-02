import os
import re
from src.config import masterdata
from src.message import prepare_data, load_list_from_file

if __name__ == "__main__":
    

    masterdata['carriers']  =   list(map(lambda x: re.escape(x), load_list_from_file('carriers')))
    masterdata['stations']  =   list(map(lambda x: re.escape(x), load_list_from_file('stations')))
    masterdata['uldGroups'] =   list(map(lambda x: re.escape(x), load_list_from_file('uldGroups')))
    masterdata['message']   =   list(map(lambda x: re.escape(x), load_list_from_file('message')))


    
    data = """
<HEADER>
POSTEDDATE:=2018-07-30T11:58:19Z
FROM:=ZRHKUXH <ZRHKUXH@TYPEB.MCONNECT.AERO>
SUBJECT:=
</HEADER>
UCM
CX383/25JUL.BKQH.ZRH
IN
.PLA52938R7
SI
ISOF PLA52398R7 / TK
SI UCM IN STD CHG MINUS 1 DAY

""".upper()


    print(prepare_data(data))
