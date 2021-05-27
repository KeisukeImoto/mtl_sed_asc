import time

def tic():
    global start_time
    start_time = time.time()

def toc(tag = 'elapsed time'):
    if 'start_time' in globals():
        print('{}: {:.3f} (sec)'.format(tag, time.time()-start_time))
    else:
        print('timer has not been started; call tic()')
