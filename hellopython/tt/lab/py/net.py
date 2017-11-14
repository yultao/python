from urllib import request


curl =  "https://query1.finance.yahoo.com/v7/finance/download/C?period1=1504107443&period2=1506785843&interval=1d&events=history&crumb=CtrRLGTBCmC"
def downloadcsv():
    response = request.urlopen(curl)
    csv = response.read()
    csv_str = str(csv)
    print(csv_str)
    
downloadcsv()