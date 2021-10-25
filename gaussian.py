from pylab import *
from scipy.optimize import curve_fit


def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

 




def find_gaussian_curve(x, y):
    expected=(0,.2,0,2,.2,125)
    params,cov=curve_fit(bimodal,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,bimodal(x,*params),color='red',lw=3,label='model')
    plt.show()
    legend()
    print(params,'\n',sigma) 
    pass



if __name__ == "__main__":
    
    data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))
    y,x,_=hist(data,100,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    find_gaussian_curve(x, y)