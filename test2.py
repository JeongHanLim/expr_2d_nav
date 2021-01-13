import zmq
import pickle as pkl
import time

if __name__ == '__main__':
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind('tcp://*:{}'.format(23232))
    while True:
        print(pkl.loads(sock.recv()))
        sock.send(pkl.dumps('send!'))
        time.sleep(5)