import numpy as np
import socket
import  time
import threading



def gen_ProductEvents(maxevent):
    data=[]
    for i in range(maxevent):
        product,price=products[np.random.randint(len(products))]
        user=names[np.random.randint(len(names))]
        data.append((user,product,str(price)))
    return data


def send_event(conn, i):
    while True:
        #print(i)

        time.sleep(1)
        nums=np.random.randint(maxEvent)
        data=gen_ProductEvents(nums)
        for value in data:
            conn.sendall(bytes(",".join(list(value))+"\n",'utf8'))
            #conn.sendall(bytes("\n",'utf8'))
        print("thread {} created {} events..".format(i,nums))


def gen_feature(nums):
    features=[]
    for i in range(nums):
        x = np.random.normal(size=maxFeatures)
        y=weight.dot(x)+noise
    features.append([y,x])
    return features

def send_feature(conn,i):
    while True:
        time.sleep(1)
        nums=np.random.randint(featureEvents)
        features=gen_feature(nums)
        for feature in features:
            x=",".join(list(map(str,feature[1])))
            y=str(feature[0])
            conn.sendall(bytes(y + "\t" + x + "\n", 'utf8'))
        print("thread {} created {} features..".format(i,nums))


if __name__ == '__main__':
    f = open("names.txt")
    names = f.read().split(",")
    products = [("iPhone Cover", 9.99), ("Headphones", 5.49), ("Samsung Galaxy Cover", 8.95), ("iPad Cover", 7.49)]
    maxEvent = 6
    maxFeatures=100
    featureEvents=100

    np.random.seed(42)
    weight = np.random.normal(size=maxFeatures)
    noise = np.random.normal() * 10

    ip_port = ('127.0.0.1', 9999)
    listener = socket.socket()
    listener.bind(ip_port)
    listener.listen(5)
    print("listening on:", listener.getsockname())

    while True:
        conn, addr=listener.accept()
        #print("here:", i)

        t_obj=[]
        for i in range(1):
            #change target func to send_event or send_feature
            t = threading.Thread(target=send_feature, args=(conn,i,))
            t.start()
            t_obj.append(t)

        for tmp in t_obj:
            tmp.join()
        conn.close()
