import time 

def start_timer(): 
    #Returns start of workout time 
    return time.monotonic()

def get_total_time(start_time):
    #Returns the total workout time shown as MM:SS
    total_seconds = int(time.monotonic() - start_time)

    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return (f"{minutes:02d}:{seconds:02d}")





