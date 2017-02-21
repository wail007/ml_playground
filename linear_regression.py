import numpy as np

class LinearRegression(object):

    def __init__(self):
        self.param = None

    def train(self, x, y):
        x  = np.hstack([np.ones([x.shape[0], 1]), x])
        xt = x.transpose()
        
        self.param = np.linalg.inv(np.dot(xt, x)).dot(xt).dot(y)


    def predict(self, x):
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        return np.dot(x, self.param)
    
    def rss(self, x, y):
        """ Residual Sum of Squares """
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        r = y - np.dot(x, self.param)

        return np.sum(np.dot(r.transpose(), r), axis=0)


def main():
    n = 100000
    p = 1000
    k = 1

    x = np.random.random([n, p])
    y = np.random.random([n, k])

    m = LinearRegression()
    m.param = np.random.random([p + 1, 1])

    print(m.rss(x, y))
    m.train(x,y)
    print(m.rss(x, y))

    #yh = m.predict(x)
    #print(yh)



if __name__ == "__main__":
    main()
