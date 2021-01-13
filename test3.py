import zmq
import pickle as pkl
import time

if __name__ == '__main__':
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://localhost:{}'.format(23232))
    print('connection!')
    while True:
        sock.send(pkl.dumps('3'))
        print(pkl.loads(sock.recv()))
        time.sleep(5)