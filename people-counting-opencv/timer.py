import datetime, time
start = datetime.timedelta(seconds=0)
while start <= datetime.timedelta(seconds=10):
    print(start)
    start += datetime.timedelta(seconds=1)
    time.sleep(1)