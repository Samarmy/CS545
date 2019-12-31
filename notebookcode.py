#!/usr/bin/env python
# coding: utf-8

# # Sam Armstrong Assignment 5 CS545

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import neuralnetworks as nn   # from notes 09
import copy


# In[20]:


actions = (-1, 0, 1)   # Possible actions

def reinforcement(s, s1):
    return 0 if abs(s1[0]-s[2]) < 1 else -1
#     return -abs(s1[0] - s[2])

def initialState(goal=None):
    if goal is None:
        goal = np.random.randint(1,10)
    return np.array([10 * np.random.random_sample(), 0.0, goal])

def nextState(s, a):
    s = copy.copy(s)   # s[0] is position, s[1] is velocity. a is -1, 0 or 1
    deltaT = 0.1                           # Euler integration time step
    s[0] += deltaT * s[1]                  # Update position
    s[1] += deltaT * (2 * a - 0.2 * s[1])  # Update velocity. Includes friction
    if s[0] < 0:        # Bound next position. If at limits, set velocity to 0.
        s = [0,0, s[2]]
    elif s[0] > 10:
        s = [10,0,s[2]]
    return s


# In[39]:


def epsilonGreedy(nnetQ, state, actions, epsilon):
    if np.random.uniform() < epsilon:
        # Random Move
        action = np.random.choice(actions)
    else:
        # Greedy Move
        Qs = [nnetQ.use(np.hstack((state, a)).reshape((1, -1))) for a in actions]
        ai = np.argmax(Qs)
        action = actions[ai]
    Q = nnetQ.use(np.hstack((state, action)).reshape((1, -1)))
    return action, Q


# In[40]:


def makeSamples(nnet, initialStateF, nextStateF, reinforcementF,
                validActions, numSamples, epsilon, goal=None):

    X = np.zeros((numSamples, nnet.n_inputs))
    R = np.zeros((numSamples, 1))
    Qn = np.zeros((numSamples, 1))

    s = initialStateF(goal)
    s = nextStateF(s, 0)        # Update state, sn from s and a
    a, _ = epsilonGreedy(nnet, s, validActions, epsilon)

    # Collect data from numSamples steps
    for step in range(numSamples):
        sn = nextStateF(s, a)        # Update state, sn from s and a
        rn = reinforcementF(s, sn)   # Calculate resulting reinforcement
        an, qn = epsilonGreedy(nnet, sn, validActions, epsilon) # Forward pass for time t+1
        X[step, :] = np.hstack((s, a))
        R[step, 0] = rn
        Qn[step, 0] = qn
        # Advance one time step
        s, a = sn, an

    return (X, R, Qn)


# In[49]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plotStatus(net, trial, epsilonTrace, rtrace, goal=5):
    plt.subplot(7, 3, 1)
    plt.plot(epsilonTrace[:trial + 1])
    plt.ylabel('Random Action Probability ($\epsilon$)')
    plt.ylim(0, 1)
    plt.subplot(7, 3, 2)
    plt.plot(X[:, 0])
    plt.plot([0, X.shape[0]], [goal, goal], '--', alpha=0.5, lw=5)
    plt.ylabel('$x$')
    plt.ylim(-1, 11)
    qs = net.use(np.array([[s, 0, goal, a] for a in actions for s in range(11)]))
    plt.subplot(7, 3, 3)
    acts = ['L', '0', 'R']
    actsiByState = np.argmax(qs.reshape((len(actions), -1)), axis=0)
    for i in range(11):
        plt.text(i, 0, acts[actsiByState[i]])
        plt.xlim(-1, 11)
        plt.ylim(-1, 1)
    plt.text(2, 0.2, 'Policy for Zero Velocity')
    plt.axis('off')
    plt.subplot(7, 3, 4)
    plt.plot(rtrace[:trial + 1], alpha=0.5)
    binSize = 20
    if trial + 1 > binSize:
        # Calculate mean of every bin of binSize reinforcement values
        smoothed = np.mean(rtrace[:int(trial / binSize) * binSize].reshape((int(trial / binSize), binSize)),
                           axis=1)
        plt.plot(np.arange(1, 1 + int(trial / binSize)) * binSize, smoothed)
    plt.ylabel('Mean reinforcement')
    plt.subplot(7, 3, 5)
    plt.plot(X[:, 0], X[:, 1])
    plt.plot(X[0, 0], X[0, 1], 'o')
    plt.xlabel('$x$')
    plt.ylabel('$\dot{x}$')
    plt.fill_between([goal-1, goal+1], [-5, -5], [5, 5], color='red', alpha=0.3)
    plt.xlim(-1, 11)
    plt.ylim(-5, 5)
    plt.subplot(7, 3, 6)
    net.draw(['$x$', '$\dot{x}$', '$a$'], ['Q'])
    
    plt.subplot(7, 3, 7)
    n = 20
    positions = np.linspace(0, 10, n)
    velocities =  np.linspace(-5, 5, n)
    xs, ys = np.meshgrid(positions, velocities)
    xsflat = xs.flat
    ysflat = ys.flat
    qs = net.use(np.array([[xsflat[i], ysflat[i], goal, a] for a in actions for i in range(len(xsflat))]))
    qs = qs.reshape((len(actions), -1)).T
    qsmax = np.max(qs, axis=1).reshape(xs.shape)
    cs = plt.contourf(xs, ys, qsmax)
    plt.colorbar(cs)
    plt.xlabel('$x$')
    plt.ylabel('$\dot{x}$')
    plt.title('Max Q')
    plt.subplot(7, 3, 8)
    acts = np.array(actions)[np.argmax(qs, axis=1)].reshape(xs.shape)
    cs = plt.contourf(xs, ys, acts, [-2, -0.5, 0.5, 2])
    plt.colorbar(cs)
    plt.xlabel('$x$')
    plt.ylabel('$\dot{x}$')
    plt.title('Actions')
    
    s = plt.subplot(7, 3, 10)
    rect = s.get_position()
    ax = Axes3D(plt.gcf(), rect=rect)
    ax.plot_surface(xs, ys, qsmax, cstride=1, rstride=1, cmap=cm.jet, linewidth=0)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\dot{x}$')
    plt.title('Max Q')
    
    s = plt.subplot(7, 3, 11)
    rect = s.get_position()
    ax = Axes3D(plt.gcf(), rect=rect)
    ax.plot_surface(xs, ys, acts, cstride=1, rstride=1, cmap=cm.jet, linewidth=0)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\dot{x}$')
    plt.title('Action')
    
    plt.tight_layout()


# In[50]:


def testIt(Qnet, nTrials, nStepsPerTrial):
    xs = np.linspace(0, 10, nTrials)
    for r in range(1, 10):
        plt.subplot(7, 3, 11 + r)
        for x in xs:
            s = [x, 0, r] # 0 velocity
            xtrace = np.zeros((nStepsPerTrial, 3))
            for step in range(nStepsPerTrial):
                a,_ = epsilonGreedy(Qnet, s, actions, 0.0) # epsilon = 0
                s = nextState(s, a)
                xtrace[step, :] = s
            plt.plot(xtrace[:, 0], xtrace[:, 1])
            plt.xlim(-1, 11)
            plt.ylim(-5, 5)
            plt.plot([r, r], [-5, 5], '--', alpha=0.5, lw=5)
            plt.ylabel('$\dot{x}$')
            plt.xlabel('$x$')
            plt.title('State Trajectories With Goal='+str(r))


# In[51]:


def setupStandardization(net, Xmeans, Xstds, Tmeans, Tstds):
    net.Xmeans = Xmeans
    net.XstdsFixed = Xstds
    net.Xconstant = [False] * len(Xmeans)
    net.TstdsFixed = net.Tstds = Tstds
    net.Tmeans = Tmeans
    net.Tconstant = [False] * len(Tstds)


# In[80]:


def test(goal=None, gamma=0.8, nTrials=500, nStepsPerTrial=1000, nSCGIterations=10, finalEpsilon=0.01, nh = [20, 20, 20]):   
    from IPython.display import display, clear_output
    fig = plt.figure(figsize=(10 ,20))

    epsilonDecay =  np.exp(np.log(finalEpsilon) / (nTrials)) # to produce this final value
    print('epsilonDecay is',epsilonDecay)
    nnetQ = nn.NeuralNetwork(4, nh, 1)
    # Inputs are position (1 to 10) velocity (-3 to 3) and action (-1, 0, or 1)
    setupStandardization(nnetQ, [5, 0, 5, 0], [2, 2, 2, 0.5], [0], [1])

    epsilon = 1         # initial epsilon value
    epsilonTrace = np.zeros(nTrials)
    rtrace = np.zeros(nTrials)
    for trial in range(nTrials):
        # Collect nStepsPerRep samples of X, R, Qn, and Q, and update epsilon
        
        X, R, Qn = makeSamples(nnetQ, initialState, nextState, reinforcement, actions, nStepsPerTrial, epsilon)
        nnetQ.train(X, R + gamma * Qn, n_epochs=nSCGIterations)

        # X,R,Qn,Q,epsilon = getSamples(Qnet,actions,nStepsPerTrial,epsilon)
        # Rest is for plotting
        epsilonTrace[trial] = epsilon
        epsilon *= epsilonDecay
        rtrace[trial] = np.mean(R)
        if True and (trial + 1 == nTrials or trial % (nTrials / 10) == 0):
            fig.clf()
            if(goal is None):
                plotStatus(nnetQ, trial, epsilonTrace, rtrace)
            else:
                plotStatus(nnetQ, trial, epsilonTrace, rtrace, goal)
            testIt(nnetQ, 10, 500)
            clear_output(wait=True)
            display(fig)

    clear_output(wait=True)


# ## Goal = 1

# In[81]:


test(goal=1)


# ## Goal = 5

# In[82]:


test(goal=5)


# ## Goal = 9

# In[83]:


test(goal=9)


# ## Experiment Number of Trials

# In[84]:


test(nTrials=250) 


# In[85]:


test(nTrials=750) 


# ## Experiment Number of Steps per Trial

# In[86]:


test(nStepsPerTrial=500)


# In[87]:


test(nStepsPerTrial=1500)


# ## Experiment Number of SCG Iterations in Each Train Call

# In[88]:


test(nSCGIterations=5) 


# In[89]:


test(nSCGIterations=15) 


# ## Experiment Number of Hidden Units

# In[90]:


test(nh = [10, 10, 10], nSCGIterations=5)


# In[91]:


test(nh = [30, 30, 30], nSCGIterations=15)


# ## Experiment Final Epsilon

# In[92]:


test(finalEpsilon=0.1)


# In[93]:


test(finalEpsilon=0.001)


# ## Experiment Gamma

# In[94]:


test(gamma=0.5)


# In[95]:


test(gamma=0.9)


# ## Discussion
# 
# Increasing the number of trials seems to increase the marble's ability to stay in the goal state which is evident in the projections of the model with more trials where the marble circles around the goal state less. Increasing the number of steps per trial seems to effect the marbles ability to get to and stay in the goal state which you can see in the trajectories with the lower steps where the marble is not able to get into some of the goal states at all. Increasing the number of SCG iterations seems to decrease the range of start states that can get to a goal state. Intuitively this makes sense because the model is likely being overtrained and is therefore less versatile in the range of start states and range of goal states (1-9). Increasing the number of hidden layers (also increasing the SCG training iterations to compensate for the increase in layers) seems to increase the marbles ability to get and stay in the goal state which can be seen in the trajectories. Decreasing the final epsilon has a similar effect as increasing the SCG iterations and seems to decrease the range of starting positions that can get to the goal state (likely caused by overfitting). Decreasing gamma seems to decrease the marbles ability to get to and stay in the goal state with the higher gamma model getting to the goal state very quickly/efficiently. Overall it seems that a more complex model performs better, but is susceptible to overfitting. 

# In[ ]:


# !rm A5grader.zip
# !rm A5grader.py
# !wget http://www.cs.colostate.edu/~anderson/cs545/notebooks/A5grader.zip
# !unzip A5grader.zip
get_ipython().run_line_magic('run', '-i A5grader.py')

#!/usr/bin/env python
# coding: utf-8

# <h1>Sam Armstrong Assignment 3 CS545</h1>

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import pandas

import torch
import mlutilities as ml  # for ml.draw
import optimizers as opt  # for opt.sgd, opt.adam, and opt.scg

import neuralnetworks as nn
# import optimizers as opt  # from Lecture Notes


# In[3]:


class NeuralNetworkClassifier(nn.NeuralNetwork):

    # Constructor
    def __init__(self, n_inputs, n_hiddens_list, classes, use_torch=False, use_gpu=False):

        # Force n_hidens_list to be a list
        if(torch.cuda.is_available() and use_gpu):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
        if not isinstance(n_hiddens_list, list):
            raise Exception('NeuralNetworkClassifier: n_hiddens_list must be a list.')
 
        # Call the constructor for NeuralNetwork, passing in the number of unique class names (ints)
        # as the number of outputs
        super().__init__(n_inputs, n_hiddens_list, len(classes), use_torch)

        # Store as member variables other things needed by instances of this class.
        
        self.classes = np.array(classes) # to allow argmax in use()
        
        if use_torch:
            self.log = torch.log
            self.exp = torch.exp
            self.sum = torch.sum
            self.unique = torch.unique
            self.mean = torch.mean
            self.tanh = torch.tanh
        else:
            self.log = np.log
            self.exp = np.exp
            self.sum = np.sum
            self.unique = np.unique
            self.mean = np.mean
            self.tanh = np.tanh
    

    def makeIndicatorVars(self, T):
        # Make sure T is two-dimensiona. Should be nSamples x 1.
        if T.ndim == 1:
            T = T.reshape((-1,1))
        return (T == np.unique(T)).astype(int)
    
    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_list}, {self.classes}, use_torch={self.use_torch})'
        if self.trained:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds. Final objective value is {self.error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str
    
    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
            
    def _standardizeX(self, X):
        if self.use_torch:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float)
            if not isinstance(self.Xmeans, torch.Tensor):
                self.Xmeans = torch.tensor(self.Xmeans, dtype=torch.float) 
            if not isinstance(self.XstdsFixed, torch.Tensor):
                self.XstdsFixed = torch.tensor(self.XstdsFixed, dtype=torch.float) 
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans
    
    def _standardizeT(self, T):
        return T

    def _unstandardizeT(self, Ts):
        return Ts

    def _forward_pass(self, X):
        # Assume weights already unpacked
        if self.use_torch:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float) 
        Z_prev = X  # output of previous layer
        Z = [Z_prev]
        for i in range(self.n_hidden_layers):
            V = self.Vs[i]
            Z_prev = self.tanh(Z_prev @ V[1:, :] + V[0:1, :])
            Z.append(Z_prev)
        Y = self.exp(Z_prev @ self.W[1:, :] + self.W[0:1, :])
        np.seterr(divide='ignore', invalid='ignore')
        softMax = Y/self.sum(Y, axis=1).reshape(-1, 1)
        return softMax, Z

    def _objectiveF(self, w, X, T):
        self._unpack(w)
        if self.use_torch:
            if not isinstance(T, torch.Tensor):
                T = torch.tensor(T, dtype=torch.float) 
        Y, _ = self._forward_pass(X)
        return -self.mean(T * self.log(Y))

    def _objective_to_actual(self, neg_mean_log_likelihood):
        return self.exp(- neg_mean_log_likelihood)
    
    def train(self, X, T, n_epochs, method='scg',
              verbose=False, save_weights_history=False,
              learning_rate=0.001, momentum_rate=0.0): # only for sgd and adam

        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        if isinstance(T, torch.Tensor):
            T = T.detach().numpy()
        if X.shape[1] != self.n_inputs:
            raise Exception(f'train: number of columns in X ({X.shape[1]}) not equal to number of network inputs ({self.n_inputs})')
       
        T = self.makeIndicatorVars(T)
        if self.use_torch:
            X = torch.tensor(X, dtype=torch.float)  # 32 bit
            T = torch.tensor(T, dtype=torch.float)

        self._setup_standardize(X, T)
        X = self._standardizeX(X)
        T = self._standardizeT(T)
        
        try:
            algo = [opt.sgd, opt.adam, opt.scg][['sgd', 'adam', 'scg'].index(method)]
        except:
            raise Exception("train: method={method} not one of 'scg', 'sgd' or 'adam'")            

        result = algo(self._pack(self.Vs, self.W),
                      self._objectiveF,
                      [X, T], n_epochs,
                      self._gradientF,  # not used if scg
                      eval_f=self._objective_to_actual,
                      learning_rate=learning_rate, momentum_rate=momentum_rate,
                      verbose=verbose, use_torch=self.use_torch,
                      save_wtrace=save_weights_history)

        self._unpack(result['w'])
        self.reason = result['reason']
        self.error_trace = result['ftrace'] # * self.Tstds # to _unstandardize the MSEs
        self.n_epochs = len(self.error_trace) - 1
        self.trained = True
        self.weight_history = result['wtrace'] if save_weights_history else None
        self.training_time = result['time']
        return self

    def use(self, X, all_outputs=False):
        if self.use_torch:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float)
        X = self._standardizeX(X)
        Y, Z = self._forward_pass(X)
        Y = self._unstandardizeT(Y)
        if self.use_torch:
            Y = Y.detach().cpu().numpy()
            Z = [Zi.detach().cpu().numpy() for Zi in Z]
        Y_classes = self.classes[np.where((Y == Y.max(axis=1)[:,None]).astype(int) == 1)[1].reshape(-1, 1)]
        return (Y_classes, Y, Z[1:]) if all_outputs else (Y_classes, Y)


# In[4]:


np.random.seed(42)
X = np.arange(20).reshape((10, 2)) * 0.1
T = np.hstack((X[:, 0:1]**2, np.sin(X[:, 1:2])))
X.shape, T.shape


# In[5]:


X


# In[6]:


T


# In[7]:


plt.plot(T)
plt.xlabel('Sample Index')
plt.legend(['Square', 'Sine']);


# In[8]:


nnet = nn.NeuralNetwork(2, [10, 10], 2)
nnet


# In[9]:


nnet.train(X, T, 50, method='scg')
nnet


# In[10]:


plt.plot(nnet.get_error_trace())
plt.xlabel('Epoch')
plt.ylabel('RMSE');
print(nnet.get_error_trace())


# In[11]:


nnet


# In[12]:


Y = nnet.use(X)
plt.plot(T)
plt.plot(Y, '--')
plt.legend(['$T_0$', '$T_1$', '$Y_0$', '$Y_1']);


# In[13]:


np.random.seed(42)  # only to help you compare your output to mine.  Do not use otherwise.

nnet = nn.NeuralNetwork(2, [10, 10], 2)
nnet.train(X, T, 50, method='sgd', learning_rate=0.1, momentum_rate=0.5)
print(nnet)
plt.plot(nnet.get_error_trace())
plt.xlabel('Epoch')
plt.ylabel('RMSE');


# In[14]:


Y = nnet.use(X)
plt.plot(T)
plt.plot(Y, '--')
plt.legend(['$T_0$', '$T_1$', '$Y_0$', '$Y_1']);


# In[15]:


np.random.seed(42)  # only to help you compare your output to mine.  Do not use otherwise.

nnet = nn.NeuralNetwork(2, [10, 10], 2)
nnet.train(X, T, 50, method='adam', learning_rate=0.1)
print(nnet)
plt.plot(nnet.get_error_trace())
plt.xlabel('Epoch')
plt.ylabel('RMSE');


# In[16]:


Y = nnet.use(X)
plt.plot(T)
plt.plot(Y, '--')
plt.legend(['$T_0$', '$T_1$', '$Y_0$', '$Y_1']);


# In[17]:


np.random.seed(42)  # only to help you compare your output to mine.  Do not use otherwise.

nnet = nn.NeuralNetwork(2, [10, 10], 2, use_torch=True)
nnet.train(X, T, 50, method='adam', learning_rate=0.1)
print(nnet)
plt.plot(nnet.get_error_trace())
plt.xlabel('Epoch')
plt.ylabel('RMSE');


# In[18]:


Y = nnet.use(X)
plt.plot(T)
plt.plot(Y, '--')
plt.legend(['$T_0$', '$T_1$', '$Y_0$', '$Y_1']);


# In[19]:


np.random.seed(42)  # only to help you compare your output to mine.  Do not use otherwise.

n_samples = 20
X = np.random.choice(3, (n_samples, 2))
T = (X[:, 0:1] == X[:, 1:2]).astype(int)  # where the two inputs are equal
classes = [0, 1]

for x, t in zip(X, T):
    print(f'x = {x}, t = {t}')


# In[20]:


print(f'{np.sum(T==0)} not equal, {np.sum(T==1)} equal')


# In[21]:


np.random.seed(42)  # only to help you compare your output to mine.  Do not use otherwise.

nnet_new = NeuralNetworkClassifier(2, [10, 10], [0, 1])
nnet_new


# In[22]:


nnet_new._standardizeT(T)


# In[23]:


nnet_new._unstandardizeT(T)


# In[24]:


nnet_new._setup_standardize(X, T)
Xst = nnet_new._standardizeX(X)
Xst


# In[25]:


Y, Z = nnet_new._forward_pass(Xst)
Y


# In[26]:


w = nnet_new._pack(nnet_new.Vs, nnet_new.W)

T_indicator_vars = np.hstack((1 - T, T))  # this only works for this particular two-class toy data

nnet_new._objectiveF(w, X, T_indicator_vars)


# In[27]:


nnet_new.train(X, T, 100)


# In[28]:


Y_classes, Y = nnet_new.use(X)
Y_classes.shape, Y.shape


# In[29]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(nnet_new.get_error_trace())
plt.xlabel('epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T, 'o-')
plt.plot(Y_classes + 0.05, 'o-')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend(['Target Class', 'Predicted Class']);


# <h2>Real Data Set</h2>

# In[30]:


# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00312/dow_jones_index.zip
# !unzip dow_jones_index.zip


# In[31]:


data = pandas.read_csv('dow_jones_index.data').fillna(0.0).drop(['date'], axis=1)
data[['open','high','low','close','next_weeks_open','next_weeks_close']] = data[['open','high','low','close','next_weeks_open','next_weeks_close']].replace('[\$,]', '', regex=True).astype(float)
classes = np.unique(data.as_matrix(columns=['stock']))
data_X = data[['quarter','open','high','low','close','volume','percent_change_price','percent_change_volume_over_last_wk','previous_weeks_volume','next_weeks_open','next_weeks_close','percent_change_next_weeks_price','days_to_next_dividend','percent_return_next_dividend']]
X = np.array(data_X)
stock = data['stock']
T = np.array(stock).reshape(-1, 1)
data


# <h2>Different Sized Networks</h2>

# <h3>Network 1 [30, 30]</h3>

# In[32]:


dj_nnet1_scg = NeuralNetworkClassifier(14, [30, 30], classes)
dj_nnet1_adam = NeuralNetworkClassifier(14, [30, 30], classes)
dj_nnet1_sgd = NeuralNetworkClassifier(14, [30, 30], classes)
dj_nnet1_scg


# In[33]:


dj_nnet1_scg.train(X, T, 1400, method='scg', verbose=True)
dj_nnet1_adam.train(X, T, 800, method='adam', verbose=True, learning_rate= 0.1)
dj_nnet1_sgd.train(X, T, 3000, method='sgd', verbose=True, learning_rate= 2.0)
dj_nnet1_scg


# In[34]:


Y_classes1, Y1 = dj_nnet1_scg.use(X)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(dj_nnet1_scg.get_error_trace())
plt.xlabel('epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T.reshape(-1), 'o-')
plt.plot(Y_classes1.reshape(-1), 'x-')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend(['Target Class', 'Predicted Class']);


# <h3>Network 2 [20, 20]</h3>

# In[35]:


dj_nnet2_scg = NeuralNetworkClassifier(14, [20, 20], classes)
dj_nnet2_adam = NeuralNetworkClassifier(14, [20, 20], classes)
dj_nnet2_sgd = NeuralNetworkClassifier(14, [20, 20], classes)
dj_nnet2_scg


# In[36]:


dj_nnet2_scg.train(X, T, 2000, method='scg', verbose=True)
dj_nnet2_adam.train(X, T, 1000, method='adam', verbose=True, learning_rate= 0.1)
dj_nnet2_sgd.train(X, T, 3000, method='sgd', verbose=True, learning_rate= 2.0)
dj_nnet2_scg


# In[37]:


Y_classes2, Y2 = dj_nnet2_scg.use(X)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(dj_nnet2_scg.get_error_trace())
plt.xlabel('epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T.reshape(-1), 'o-')
plt.plot(Y_classes2.reshape(-1), 'x-')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend(['Target Class', 'Predicted Class']);


# <h3>Network 3 [10, 10]</h3>

# In[38]:


dj_nnet3_scg = NeuralNetworkClassifier(14, [10, 10], classes)
dj_nnet3_adam = NeuralNetworkClassifier(14, [10, 10], classes)
dj_nnet3_sgd = NeuralNetworkClassifier(14, [10, 10], classes)
dj_nnet3_scg


# In[39]:


dj_nnet3_scg.train(X, T, 4000, method='scg', verbose=True)
dj_nnet3_adam.train(X, T, 4000, method='adam', verbose=True, learning_rate= 0.1)
dj_nnet3_sgd.train(X, T, 4000, method='sgd', verbose=True, learning_rate= 2.0)
dj_nnet3_scg


# In[40]:


Y_classes3, Y3 = dj_nnet3_scg.use(X)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(dj_nnet3_scg.get_error_trace())
plt.xlabel('epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T.reshape(-1), 'o-')
plt.plot(Y_classes3.reshape(-1), 'x-')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend(['Target Class', 'Predicted Class']);


# <h2>Discussion</h2>
# 
# The dataset I choose are the stocks that make up the Dow Jones Index. There are 30 different stocks each with 14 different attributes and 750 total samples (duplicate stocks were recorded at different points in time). I used three different sized networks Network1 [30, 30], Network2 [20, 20], and Network 3 [10, 10]. I choose these 3 different sizes because I wanted to test if the size of the hidden layers needs to be at least the size of the number of classes. 
# 
# Network1 performed the best with the fastest increase in the objective function. Network1's best performing training was Adam (able to train to an objective function of 1.0 in 800 epochs), its second best performing training was SCG (able to train to an objective function of 1.0 in 1400 epochs), and its worst performing training was SGD (able to train to an objective function of 0.99 in 3000 epochs). Network1 was not able to reach an objective function of 1.0 using SGD with 3000 or less epochs. 
# 
# Network2 was the second best performing network with the second fastest increasing objective function. Network2's best performing training was Adam (able to train to an objective function of 1.0 in 900 epochs), its second best performing training was SCG (able to train to an objective function of 1.0 in 2000 epochs), and its worst performing training was SGD (able to train to an objective function of 0.99 in 3000 epochs). Reducing the hidden layers from 30 to 20 increased the amount of epochs needed to train SCG and Adam to the same objective function of 1.0, but they were still able to reach the objective function of 1.0. Network2 was not able to reach an objective function of 1.0 using SGD with 3000 or less epochs. 
# 
# Network3 performed the worst with the slowest increasing objective function. Network3's best performing training was Adam (able to train to an objective function of 1.0 in 2400 epochs), its second best performing training was SCG (able to train to an objective function of 0.999 in 4000 epochs), and its worst performing training was SGD (able to train to an objective function of 0.98 in 4000 epochs). Reducing the hidden layers from 20 to 10 again increased the amount of epochs needed to train Adam to the same objective function of 1.0, but Adam was still able to reach the objective function of 1.0. Network3 was not able to reach an objective function of 1.0 using SCG or SGD using 4000 or less epochs. 
# 
# Although using the same size of the hidden layers as number of classes (30) performed the best, reducing the size of the hidden layers didn't limit the objective function to lower than 1.0. Adam consistently trained the best on all three Networks, follwed by SCG and than SGD. The step size of Adam was set to 0.1 and the step size of SGD was set to 2.0 for all the trainings. Using a step size of 2.0 for SGD seemed large, but using a step size of 0.1 resulted in the model training way to slow.

# <h2>Extra Credit 1</h2>

# In[41]:


I = {}
for s in classes:
    temp, _ = np.where(T == s)
    I[s] = np.random.permutation(temp)

N = {}
for x, y in I.items():
    N[x] = round(0.8*len(y))

Ltrain = [] 
Ltest = []  
for s2 in classes:
    Ltrain.append(I[s2][:N[s2]])
    Ltest.append(I[s2][N[s2]:])   
    
rowsTrain = np.hstack(Ltrain)
Xtrain = X[rowsTrain, :]
Ttrain = T[rowsTrain, :]
rowsTest = np.hstack(Ltest)
Xtest =  X[rowsTest, :]
Ttest =  T[rowsTest, :]


# <h3>Network 1 [30, 30] SCG</h3>

# In[42]:


par_nnet1_scg = NeuralNetworkClassifier(14, [30, 30], classes)
par_nnet1_scg.train(Xtrain, Ttrain, 500, method='scg', verbose=True)
Yc, Y = par_nnet1_scg.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet1_scg.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 1 [30, 30] Adam</h3>

# In[43]:


par_nnet1_adam = NeuralNetworkClassifier(14, [30, 30], classes)
par_nnet1_adam.train(Xtrain, Ttrain, 500, method='adam', verbose=True, learning_rate= 0.1)
Yc, Y = par_nnet1_adam.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet1_adam.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 1 [30, 30] SGD</h3>

# In[44]:


par_nnet1_sgd = NeuralNetworkClassifier(14, [30, 30], classes)
par_nnet1_sgd.train(Xtrain, Ttrain, 2000, method='sgd', verbose=True, learning_rate= 2.0)
Yc, Y = par_nnet1_sgd.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet1_sgd.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 2 [20, 20] SCG</h3>

# In[45]:


par_nnet2_scg = NeuralNetworkClassifier(14, [20, 20], classes)
par_nnet2_scg.train(Xtrain, Ttrain, 1000, method='scg', verbose=True)
Yc, Y = par_nnet2_scg.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet2_scg.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 2 [20, 20] Adam</h3>

# In[46]:


par_nnet2_adam = NeuralNetworkClassifier(14, [20, 20], classes)
par_nnet2_adam.train(Xtrain, Ttrain, 1000, method='adam', verbose=True, learning_rate= 0.1)
Yc, Y = par_nnet2_adam.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet2_adam.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 2 [20, 20] SGD</h3>

# In[47]:


par_nnet2_sgd = NeuralNetworkClassifier(14, [20, 20], classes)
par_nnet2_sgd.train(Xtrain, Ttrain, 4000, method='sgd', verbose=True, learning_rate= 2.0)
Yc, Y = par_nnet2_sgd.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet2_sgd.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 3 [10, 10] SCG</h3>

# In[48]:


par_nnet3_scg = NeuralNetworkClassifier(14, [10, 10], classes)
par_nnet3_scg.train(Xtrain, Ttrain, 1500, method='scg', verbose=True)
Yc, Y = par_nnet3_scg.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet3_scg.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 3 [10, 10] Adam</h3>

# In[49]:


par_nnet3_adam = NeuralNetworkClassifier(14, [10, 10], classes)
par_nnet3_adam.train(Xtrain, Ttrain, 1500, method='adam', verbose=True, learning_rate= 0.1)
Yc, Y = par_nnet3_adam.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet3_adam.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h3>Network 3 [10, 10] SGD</h3>

# In[50]:


par_nnet3_sgd = NeuralNetworkClassifier(14, [10, 10], classes)
par_nnet3_sgd.train(Xtrain, Ttrain, 6000, method='sgd', verbose=True, learning_rate= 2.0)
Yc, Y = par_nnet3_sgd.use(Xtrain)
n_correct = (Yc == Ttrain).sum()
print(f'Training data {n_correct} out of {Ttrain.shape[0]} samples, or {n_correct/Ttrain.shape[0]*100:.2f} percent.')
Yc, Y = par_nnet3_sgd.use(Xtest)
n_correct = (Yc == Ttest).sum()
print(f'Testing Data {n_correct} out of {Ttest.shape[0]} samples, or {n_correct/Ttest.shape[0]*100:.2f} percent.')


# <h2>Discussion Extra Credit 1</h2>
# 
# I re-used the three different sized networks Network1 [30, 30], Network2 [20, 20], and Network 3 [10, 10] from the main assignment. I also used the same Dow Jones data (600 training, 150 testing) with an 80/20 split. All of the networks using SCG or Adam got a 600/600 (100%) on the training data. While all the networks using SGD got 580-590/600 (\~97-98%). Suprisingly all of the networks and training methods resulted in an 140-148/150 (~94-98%) success rate on the testing data making it hard to compare them using only one NN initialization. After initializing each network multiple times it seems that the best to worst networks were Network1, Network2, Network3 and that the best to worst training methods were SCG, Adam, SGD based on their testing success rate. The amount of epochs I used on Network 1 for both SCG and Adam was 500 and for SGD it was 2000. The amount of epochs I used on Network 2 for both SCG and Adam was 1000 and for SGD it was 4000. The amount of epochs I used on Network 3 for both SCG and Adam was 1500 and for SGD it was 6000. The reasoning behind increasing the epochs is that the smaller hidden layers neural networks need more training to achieve the same objective function level as the larger hidden layers neural networks.

# <h2>Extra Credit 2</h2>
# *Had same problem mentioned in class where using the gpu slows down alot in the higher iterations of training

# In[51]:


gpu_nnet = NeuralNetworkClassifier(14, [30, 30], classes, use_torch=True, use_gpu=True)
gpu_nnet.train(X, T, 200, method='scg', verbose=True)
Y_classes, Y = gpu_nnet.use(X)


# In[52]:


# !rm A3grader.zip
# !rm A3grader.py
# !wget https://www.cs.colostate.edu/~anderson/cs545/notebooks/A3grader.zip
# !unzip A3grader.zip
# %run -i A3grader.py

#!/usr/bin/env python
# coding: utf-8

# Sam Armstrong Assignment 1 CS545 

# In[7]:


import numpy as np
import pandas

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Download Data

# In[38]:


# !curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip
# !unzip -o AirQualityUCI.zip


# Download A1 Grader

# In[39]:


# !curl -O http://www.cs.colostate.edu/~anderson/cs545/notebooks/A1grader.zip
# !unzip -o A1grader.zip


# Root Mean Square Error

# In[1]:


def rmse(model, X, T, W):
    Y = model(X, W)
    return np.sqrt(np.mean((T - Y)**2))


# Linear Model

# In[2]:


def linear_model(X, W):
    return np.array(X) @ W[1:, :] + W[0,:]


# Quadratic Model

# In[3]:


def quadratic_model(X, W):
    return np.hstack((X, X**2)) @ W[1:, :] + W[0, :]


# Cubic Model

# In[4]:


def cubic_model(X, W):
    return np.hstack((X, X**2, X**3)) @ W[1:, :] + W[0, :]


# Quartic Model

# In[5]:


def quartic_model(X, W):
    return np.hstack((X, X**2, X**3, X**4)) @ W[1:, :] + W[0, :]


# Linear Model Gradient

# In[6]:


def linear_model_gradient(X, T, W):
    dEdY = -2 * (T - linear_model(X, W))
    all_but_bias = np.array(X)
    dYdW = np.insert(all_but_bias, 0, 1, axis=1)
    result = dEdY.T @ dYdW / X.shape[0]
    return result.T


# In[26]:


xx = np.array((1, 2)).reshape(-1, 1)
yy = np.array((2, 2)).reshape(-1, 1)
ww = np.zeros((2, 1))
ww = np.array((1, 0)).reshape(-1, 1)
linear_model(yy, ww)
print(linear_model_gradient(xx, yy, ww))
plt.plot(xx, yy, 'o');


# Quadratic Model Gradient

# In[46]:


def quadratic_model_gradient(X, T, W):
    dEdY = -2 * (T - quadratic_model(X, W))
    all_but_bias = np.hstack((X, X**2))
    dYdW = np.insert(all_but_bias, 0, 1, axis=1)
    result = dEdY.T @ dYdW / X.shape[0]
    return result.T


# Cubic Model Gradient

# In[47]:


def cubic_model_gradient(X, T, W):
    dEdY = -2 * (T - cubic_model(X, W))
    all_but_bias = np.hstack((X, X**2, X**3))
    dYdW = np.insert(all_but_bias, 0, 1, axis=1)
    result = dEdY.T @ dYdW / X.shape[0]
    return result.T


# Quartic Model Gradient

# In[48]:


def quartic_model_gradient(X, T, W):
    dEdY = -2 * (T - quartic_model(X, W))
    all_but_bias = np.hstack((X, X**2, X**3, X**4))
    dYdW = np.insert(all_but_bias, 0, 1, axis=1)
    result = dEdY.T @ dYdW / X.shape[0]
    return result.T


# Gradient Descent Adam

# In[49]:


def gradient_descent_adam(model_f, gradient_f, rmse_f, X, T, W, rho, n_steps):
    # Commonly used parameter values
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = 0
    v = 0
    
    error_sequence = []
    W_sequence = []
    
    for step in range(n_steps):
        error_sequence.append(rmse_f(model_f, X, T, W))
        W_sequence.append(W.flatten())
        
        g = gradient_f(X, T, W)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1 ** (step+1))
        vhat = v / (1 - beta2 ** (step+1))
        W -= rho * mhat / (np.sqrt(vhat) + epsilon)
        
    return W, error_sequence, W_sequence


# Format and Plot Data

# In[50]:


data = pandas.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',',
                        usecols=range(15), na_values=-200)
data = data[['Time', 'CO(GT)']]
# data = data[:46]  # only use the first 46 samples
data = data.dropna(axis=0)
    
hour = np.array([int(t[:2]) for t in data['Time']])
CO = np.array(data['CO(GT)'])

T = CO.reshape(-1, 1)
Tnames = ['CO']
X = hour.reshape(-1, 1)
Xnames = ['Hour']
print('X.shape =', X.shape, 'Xnames =', Xnames)
print('T.shape =', T.shape, 'Tnames =', Tnames)
plt.ylabel("Air Quality")
plt.xlabel("Hour of the Day")
plt.plot(X, T, 'o');


# Find Best Weights for the Linear Model

# In[56]:


rhoAry = np.array((1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10))
error_linear_best = 1000
n_steps = 3000
for rho in rhoAry:
    W = np.zeros((2, 1))
    W, error_sequence, W_sequence = gradient_descent_adam(linear_model, linear_model_gradient, rmse, X, T, W, rho, n_steps)
    print(np.sqrt(error_sequence[-1]))
    if(np.sqrt(error_sequence[-1]) < error_linear_best):
        W_best = W
        error_sequence_best = error_sequence
        W_sequence_best = W_sequence
        rho_linear_best = rho
        error_linear_best = np.sqrt(error_sequence[-1])

print(f'The lowest RMSE is {error_linear_best} for rho {rho_linear_best} at step {n_steps}')


# Plot the Linear Model Results

# In[57]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(error_sequence_best)
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.subplot(1, 3, 2)
plt.plot(W_sequence_best)
plt.xlabel("Number of Steps")
plt.ylabel("Weight")
plt.subplot(1, 3, 3)
plt.plot(X, T, 'o')
plt.ylabel("Air Quality")
plt.xlabel("Hour of the Day")
xs = np.linspace(0, 24, 100).reshape(-1, 1)
plt.plot(xs, linear_model(xs, W_best));


# <h2>Linear Model Discussion</h2>
# 
# The best learning rate value for the linear model is 0.01 with a RMSE of 1.1647985282595315. Since the linear model only has two weighted variables the gradient descent probably has less local minimums then the quadratic, cubic, and quartic models. Training it on the highest learning rate (0.01) yields the bests results and using a lower learning rate probably results in the gradient descent being too slow and not able to reach the same minimum in the same number of steps.

# Find Best Weights for the Quadratic Model

# In[58]:


error_quadratic_best = 1000
for rho in rhoAry:
    W = np.zeros((3, 1))
    W, error_sequence, W_sequence = gradient_descent_adam(quadratic_model, quadratic_model_gradient, rmse, X, T, W, rho, n_steps)
    print(np.sqrt(error_sequence[-1]))
    if(np.sqrt(error_sequence[-1]) < error_quadratic_best):
        W_best = W
        error_sequence_best = error_sequence
        W_sequence_best = W_sequence
        rho_quadratic_best = rho
        error_quadratic_best = np.sqrt(error_sequence[-1])

print(f'The lowest RMSE is {error_quadratic_best} for rho {rho_quadratic_best} at step {n_steps}')


# Plot the Quadratic Model Results

# In[59]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(error_sequence_best)
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.subplot(1, 3, 2)
plt.plot(W_sequence_best)
plt.xlabel("Number of Steps")
plt.ylabel("Weight")
plt.subplot(1, 3, 3)
plt.plot(X, T, 'o')
plt.ylabel("Air Quality")
plt.xlabel("Hour of the Day")
xs = np.linspace(0, 24, 100).reshape(-1, 1)
plt.plot(xs, quadratic_model(xs, W_best));


# <h2>Quadratic Model Discussion</h2>
# 
# The best learning rate value for the quadratic model is 0.01 with a RMSE of 1.1591889144427283. The quadratic model uses the same learning rate as the linear model but has a lower RMSE because a quadratic equation (curved line) can fit the same data more accuratly then a straight linear line. The likely reason the linear and quadratic models share the best learning rate of 0.01 is because they have similarly scaled gradient descents with a low number of local minimums they could get stuck in. Training it on the highest learning rate (0.01) yields the bests results and using a lower learning rate probably results in the gradient descent being too slow and not able to reach the same minimum in the same number of steps.

# Find Best Weights for the Cubic Model

# In[60]:


error_cubic_best = 1000
for rho in rhoAry:
    W = np.zeros((4, 1))
    W, error_sequence, W_sequence = gradient_descent_adam(cubic_model, cubic_model_gradient, rmse, X, T, W, rho, n_steps)
    print(np.sqrt(error_sequence[-1]))
    if(np.sqrt(error_sequence[-1]) < error_cubic_best):
        W_best = W
        error_sequence_best = error_sequence
        W_sequence_best = W_sequence
        rho_cubic_best = rho
        error_cubic_best = np.sqrt(error_sequence[-1])

print(f'The lowest RMSE is {error_cubic_best} for rho {rho_cubic_best} at step {n_steps}')


# Plot the Cubic Model Results

# In[61]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(error_sequence_best)
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.subplot(1, 3, 2)
plt.plot(W_sequence_best)
plt.xlabel("Number of Steps")
plt.ylabel("Weight")
plt.subplot(1, 3, 3)
plt.plot(X, T, 'o')
plt.ylabel("Air Quality")
plt.xlabel("Hour of the Day")
xs = np.linspace(0, 24, 100).reshape(-1, 1)
plt.plot(xs, cubic_model(xs, W_best));


# <h2>Cubic Model Discussion</h2>
# 
# The best learning rate value for the cubic model is 0.01 with a RMSE of 1.1489330924746517. The cubic model uses the same learning rate as the linear and quadratic model but has a lower RMSE because a cubic equation (double curved line) can fit the data more accuratly then a straight linear or single curved quadratic line. The likely reason the linear, quadratic, and cubic models share the best learning rate of 0.01 is because they have similarly scaled gradient descents with a low number of local minimums they could get stuck in. Training it on the highest learning rate (0.01) yields the bests results and using a lower learning rate probably results in the gradient descent being too slow and not able to reach the same minimum in the same number of steps.

# Find Best Weights for the Quartic Model

# In[62]:


error_quartic_best = 1000
for rho in rhoAry:
    W = np.zeros((5, 1))
    W, error_sequence, W_sequence = gradient_descent_adam(quartic_model, quartic_model_gradient, rmse, X, T, W, rho, n_steps)
    print(np.sqrt(error_sequence[-1]))
    if(np.sqrt(error_sequence[-1]) < error_quartic_best):
        W_best = W
        error_sequence_best = error_sequence
        W_sequence_best = W_sequence
        rho_quartic_best = rho
        error_quartic_best = np.sqrt(error_sequence[-1])

print(f'The lowest RMSE is {error_quartic_best} for rho {rho_quartic_best} at step {n_steps}')


# Plot the Quartic Model Results

# In[63]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(error_sequence_best)
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.subplot(1, 3, 2)
plt.plot(W_sequence_best)
plt.xlabel("Number of Steps")
plt.ylabel("Weight")
plt.subplot(1, 3, 3)
plt.plot(X, T, 'o')
plt.ylabel("Air Quality")
plt.xlabel("Hour of the Day")
xs = np.linspace(0, 24, 100).reshape(-1, 1)
plt.plot(xs, quartic_model(xs, W_best));


# <h2>Quartic Model Discussion</h2>
# 
# The best learning rate value for the quartic model is 0.001 with a RMSE of 1.1708838115369375. The quartic model uses a lower learning rate than the linear, quadratic, and cubic models but has a higher RMSE probably because the learning rate of 0.01 is too high of a step (and overshoots the minimum) and the next step rate of 0.001 is low enough (to not overshoot the minimum) but doesn't have enough iterations to reach the minimum. The likely reason the quartic model has a higher learning rate of 0.001 than the linear, quadratic, and cubic models is because it has more local minimums or a less steep gradient descent.

# <h1>Extra Credit</h1>

# Linear Model Plus Sine

# In[64]:


def linear_sine_model(X, W):
    return np.hstack((X, np.sin(X))) @ W[1:, :] + W[0, :]


# Linear Model Plus Sine Gradient

# In[65]:


def linear_sine_model_gradient(X, T, W):
    dEdY = -2 * (T - linear_sine_model(X, W))
    all_but_bias = np.hstack((X, np.sin(X)))
    dYdW = np.insert(all_but_bias, 0, 1, axis=1)
    result = dEdY.T @ dYdW / X.shape[0]
    return result.T


# Find Best Weights for the Linear Model Plus Sine

# In[66]:


error_linear_sine_best = 1000
for rho in rhoAry:
    W = np.zeros((3, 1))
    W, error_sequence, W_sequence = gradient_descent_adam(linear_sine_model, linear_sine_model_gradient, rmse, X, T, W, rho, n_steps)
    print(np.sqrt(error_sequence[-1]))
    if(np.sqrt(error_sequence[-1]) < error_linear_sine_best):
        W_best = W
        error_sequence_best = error_sequence
        W_sequence_best = W_sequence
        rho_linear_sine_best = rho
        error_linear_sine_best = np.sqrt(error_sequence[-1])

print(f'The lowest RMSE is {error_linear_sine_best} for rho {rho_linear_sine_best} at step {n_steps}')


# Plot the Linear Model Plus Sine Results

# In[67]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(error_sequence_best)
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.subplot(1, 3, 2)
plt.plot(W_sequence_best)
plt.xlabel("Number of Steps")
plt.ylabel("Weight")
plt.subplot(1, 3, 3)
plt.plot(X, T, 'o')
plt.ylabel("Air Quality")
plt.xlabel("Hour of the Day")
xs = np.linspace(0, 24, 100).reshape(-1, 1)
plt.plot(xs, linear_sine_model(xs, W_best));


# <h2>Linear Model Plus Sine Discussion</h2>
# 
# The best learning rate value for the linear model plus sine is 0.01 with a RMSE of 1.1614571617336993. The linear model plus sine performed better then the linear (1.1647985282595315) model but worse than the quadratic (1.1591889144427283) and cubic (1.1489330924746517) models. The reason for this is probably the linear model plus sine has one more weight than the linear model but having the weight attached to a sine function is less affective then attaching it to a squared function. The best learning rate (0.01) is the same as the linear, quadratic, and cubic best learning rates, this is likely due to having similarly scaled gradient descents with a low number of local minimums they could get stuck in. Training it on the highest learning rate (0.01) yields the bests results and using a lower learning rate probably results in the gradient descent being too slow and not able to reach the same minimum in the same number of steps.

# In[68]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# <h1>Sam Armstrong Assignment 2 CS545</h1>

# In[2]:


import numpy as np
import pandas
import optimizers as opt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# import optimizers as opt  # from Lecture Notes


# <h2>Matrix Equations</h2>

# \begin{align}
# N & = \text{number of samples} \\
# I & = \text{the number of attributes in each sample} \\
# K & = \text{number of units in output layer} \\
# H_1 & = \text{number of units in first hidden layer} \\
# H_2 & = \text{number of units in second hidden layer} \\
# \\
# \mathbf{Z}_1 & = \tanh(\hat{\mathbf{X}}\ \mathbf{U}) \\
# \mathbf{Z}_2 & = \tanh(\hat{\mathbf{Z}}_1\ \mathbf{V}) \\
# \mathbf{Y} & = \hat{\mathbf{Z}}_2\ \mathbf{W} \\
# \mathbf{E} & = \frac{1}{NK}\sum_{n=1}^{N}\sum_{k=1}^{K}(\mathbf{T}_{n,k} - \mathbf{Y}_{n,k})^2 \\
# \\
# \nabla_{\mathbf{Y}}{E_{n,k}} & = \frac{-2}{NK}(\mathbf{T}_{n,k} - \mathbf{Y}_{n,k}) \\
# \delta_{\mathbf{Y}} & = \frac{-2}{NK}(\mathbf{T} - \mathbf{Y}) \\
# \nabla_{\mathbf{W}}{E} & = \underbrace{\underbrace{\hat{\mathbf{Z}}_2^T}_{H_2+1 \times N} \underbrace{\delta_{\mathbf{Y}}}_{N \times K}}_{H_2+1 \times K} \\
# \\
# \nabla_{\mathbf{V}}{E} & = \underbrace{\underbrace{\hat{\mathbf{Z}}_1^T}_{H_1+1 \times N} \underbrace{\delta_{\mathbf{Z}_2}}_{N \times H_2}}_{H_1+1 \times H_2} \text{ where } \delta_{\mathbf{Z}_2} = (\delta_{\mathbf{Y}} \mathbf{W}_{1:}^T) \cdot (1-\mathbf{Z}_2^2) \ \text{ if } f(\hat{\mathbf{X}} \mathbf{V}) = \tanh{(\hat{\mathbf{X}} \mathbf{V})} \\
# \\
# \nabla_{\mathbf{U}}{E} & = \underbrace{\underbrace{\hat{\mathbf{X}}^T}_{I+1 \times N} \underbrace{\delta_{\mathbf{Z}_1}}_{N \times H_1}}_{I+1 \times H_1} \text{ where } \delta_{\mathbf{Z}_1} = (\delta_{\mathbf{Z}_2} \mathbf{V}_{1:}^T) \cdot (1-\mathbf{Z}_1^2) \ \text{ if } f(\hat{\mathbf{X}} \mathbf{V}) = \tanh{(\hat{\mathbf{X}} \mathbf{V})} \\
# \end{align}

# <h2>Network Function</h2>

# In[3]:


def network(w, n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, all_outputs=False):
    n_U = (n_inputs + 1) * n_hiddens_1
    n_V = (n_hiddens_1 + 1) * n_hiddens_2
    n_W = (n_hiddens_2 + 1) * n_outputs
    U = w[:n_U].reshape((n_inputs + 1, n_hiddens_1))
    V = w[n_U:(n_U+n_V)].reshape((n_hiddens_1 + 1, n_hiddens_2))
    W = w[(n_U+n_V):].reshape((n_hiddens_2 + 1, n_outputs))
    Z1 = np.tanh(U[0:1, :] + X @ U[1:, :])
    Z2 = np.tanh(V[0:1, :] + Z1 @ V[1:, :])
    Y = W[0:1, :] + Z2 @ W[1:, :]
    return (Y, Z1, Z2) if all_outputs else Y


# <h2>Error Gradient Function</h2>

# In[4]:


def error_gradient(w, n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T):
    Y, Z1, Z2 = network(w, n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, all_outputs=True)
    n_samples = X.shape[0]
    delta_Y = -2 / (n_samples * n_outputs) * (T - Y)
    Z2_hat = np.insert(Z2, 0, 1, axis=1)
    dEdW = Z2_hat.T @ delta_Y
    
    n_W = (n_hiddens_2 + 1) * n_outputs
    W = w[-n_W:].reshape((n_hiddens_2 + 1, n_outputs))
    delta_Z2 = (delta_Y @ W[1:, :].T) * (1 - Z2**2)
    Z1_hat = np.insert(Z1, 0, 1, axis=1)
    dEdV = Z1_hat.T @ delta_Z2
    
    n_U = (n_inputs + 1) * n_hiddens_1
    n_V = (n_hiddens_1 + 1) * n_hiddens_2
    V = w[n_U:(n_U+n_V)].reshape((n_hiddens_1 + 1, n_hiddens_2))
    delta_Z1 = (delta_Z2 @ V[1:, :].T) * (1 - Z1**2)
    X_hat = np.insert(X, 0, 1, axis=1)
    dEdU = X_hat.T @ delta_Z1
 
    dEdw = np.hstack((dEdU.flatten(), dEdV.flatten(), dEdW.flatten()))

    return dEdw


# <h2>Mean Squared Error Function</h2>

# In[5]:


def mse(w, n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T):
    Y = network(w, n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X)
    return np.mean((T - Y)**2)


# <h2>Import Data</h2>

# In[6]:


data = pandas.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',', usecols=range(15), na_values=-200)
data = data[['Time', 'CO(GT)']]
data = data [:23 * 20]  # first 20 days of data
data = data.dropna(axis=0)
print('data.shape =', data.shape)

hour = [int(t[:2]) for t in data['Time']]
X = np.array(hour).reshape(-1, 1)
CO = data['CO(GT)']
T = np.array(CO).reshape(-1, 1)
np.hstack((X, T))[:10]  # show the first 10 samples of hour, CO


# <h2>Define the Network Parameters</h2>

# In[7]:


n_inputs = X.shape[1]
n_hiddens_1 = 5
n_hiddens_2 = 5
n_outputs = T.shape[1]


# <h2>Intialize Weight Vector</h2>

# In[8]:


n_U = (n_inputs + 1) * n_hiddens_1
n_V = (n_hiddens_1 + 1) * n_hiddens_2
n_W = (n_hiddens_2 + 1) * n_outputs

initial_w = np.random.uniform(-0.1, 0.1, n_U + n_V + n_W)  # range of weights is -0.1 to 0.1


# <h2>Standardize the Input Values</h2>

# In[9]:


standardize = True

if standardize:
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    X = (X - X_mean) / X_std
    
print(f'X mean is {X.mean(axis=0)[0]:.3f} and its standard deviation is {X.std(axis=0)[0]:.3f}')


# <h2>Train Network with Three Optimization Algorithms</h2>

# In[10]:


n_iterations = 2000

result_sgd = opt.sgd(initial_w,
                     mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                     n_iterations=n_iterations, learning_rate=1e-1, momentum_rate=0.2, 
                     save_wtrace=True)
print(f'SGD final error is {result_sgd["ftrace"][-1]:.3f} and it took {result_sgd["time"]:.2f} seconds')

result_adam = opt.adam(initial_w, 
                       mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                       n_iterations=n_iterations, learning_rate=1e-2, 
                       save_wtrace=True)
print(f'Adam final error is {result_adam["ftrace"][-1]:.3f} and it took {result_adam["time"]:.2f} seconds')

result_scg = opt.scg(initial_w,
                     mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                     n_iterations=n_iterations,
                     save_wtrace=True)
print(f'SCG final error is {result_scg["ftrace"][-1]:.3f} and it took {result_scg["time"]:.2f} seconds')


# <h2>Plot Error Curves and Model Fits</h2>

# In[11]:


plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(result_sgd['ftrace'], label='SGD')
plt.plot(result_adam['ftrace'], label='Adam')
plt.plot(result_scg['ftrace'], label='SCG')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.ylim(0, 4)

plt.subplot(1, 2, 2)
if standardize:
    plt.plot(((X * X_std) + X_mean), T, 'k.')  # unstandardize X
else:
    plt.plot(X, T, 'k.')
xs = np.linspace(0, 23, 100).reshape((-1, 1))
xs_standardized = (xs - X_mean) / X_std if standardize else xs
plt.plot(xs, network(result_sgd['w'], n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, xs_standardized), label='SGD')
plt.plot(xs, network(result_adam['w'], n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, xs_standardized), label='Adam')
plt.plot(xs, network(result_scg['w'], n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, xs_standardized), label='SCG')
plt.legend()
plt.xlabel('Hour')
plt.ylabel('CO');


# <h3>Discussion</h3>
# SCG performs the best with a MSE of 0.912. SCG is also the slowest running with a 2.05 second runtime. The SCG line in the Iterations by MSE graph drops the quickest and goes horizontal at about 250 iterations, showing SCG performs the best within a small amount of iterations. Adam performs the second best with a MSE 0.913. Adam is also the second quickest to run at 1.02 seconds. The Adam line goes horizontal at about 1250 iterations showing Adam performs about the same at 1250 iterations as SCG performs at 250 iterations. The SGD performs the worse with a MSE of 0.988. SGD has the quickest run time of 0.97 seconds. Although SGD performs the worst its line in the Iterations by MSE is still dropping which indicates it needs more iterations than 2000 before it's line becomes horizontal. The lines in the Hour by CO graph are very similar for all three (SCG, Adam, SGD) which makes sense since all three have very close MSE and have identical inputs.

# <h2>Find Best Parameters for n_iterations, n_hiddens_1, n_hiddens_2 and learning_rate</h2>
# <h3>*This Takes a While to Run (~1 hour)</h3>

# In[11]:


results = []
for n_iterations in [2000, 3000, 4000]:
    print("Iterations for " + str(n_iterations))
    for nh1 in [10, 11, 12, 13, 14, 15]:
        for nh2 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            n_U = (n_inputs + 1) * nh1
            n_V = (nh1 + 1) * nh2
            n_W = (nh2 + 1) * n_outputs
            initial_w = np.random.uniform(-0.1, 0.1, n_U + n_V + n_W)

            result_scg = opt.scg(initial_w, mse, error_gradient, fargs=[n_inputs, nh1, nh2, n_outputs, X, T],
                                 n_iterations=n_iterations)
            
            results.append([n_iterations, nh1, nh2, "N/A", 'scg', result_scg['ftrace'][-1]])
            
            for lr in [1e-1, 1e-2]:
                result_sgd = opt.sgd(initial_w, mse, error_gradient, fargs=[n_inputs, nh1, nh2, n_outputs, X, T],
                     n_iterations=n_iterations, learning_rate=lr, momentum_rate=0)
                result_adam = opt.adam(initial_w, mse, error_gradient, fargs=[n_inputs, nh1, nh2, n_outputs, X, T],
                                       n_iterations=n_iterations, learning_rate=lr)
                
                
                results.append([n_iterations, nh1, nh2, lr, 'sgd', result_sgd['ftrace'][-1]])
                results.append([n_iterations, nh1, nh2, lr, 'adam', result_adam['ftrace'][-1]])
        print(str(round((((nh1 - 9)/6)*100), 2)) + "%")
results = pandas.DataFrame(results, columns=('Iterations', 'nh1', 'nh2', 'lr', 'algo', 'mse'))
results.sort_values('mse').head(20)


# <h3>SCG Results</h3>

# In[12]:


scg_results = results[results['algo'] =='scg'].sort_values('mse').head(20)
scg_results


# <h3>SGD Results</h3>

# In[13]:


sgd_results = results[results['algo'] =='sgd'].sort_values('mse').head(20)
sgd_results


# <h3>Adam Results</h3>

# In[14]:


adam_results = results[results['algo'] =='adam'].sort_values('mse').head(20)
adam_results


# <h3>Discussion</h3>
# The range for n_iterations was [2000, 3000, 4000], the range for nh1 was [10, 11, 12, 13, 14, 15], the range for nh2 was [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], and the range for lr was [1e-1, 1e-2]. Adam performed the best with an MSE of 0 (less than 1e-35) in 2000 iterations, a learning rate of 0.01, a first hidden layer of 14, and a second hidden layer of 8. SCG performed the second best with an MSE of 7.167041e-19 in 3000 iterations, a first hidden layer of 12, and a second hidden layer of 15. SGD performed the worse with an MSE of 3.423091e-11 in 4000 iterations, a learning rate of 0.1, a first hidden layer of 13, and a second hidden layer of 2.

# <h2>5 Samples and 2 Outputs</h2>

# In[24]:


X = np.arange(15).reshape((5, 3))
T = np.hstack((X[:, 0:1] * 0.1 * X[:, 1:2], X[:, 2:]**2)) # making two target values for each sample
T = T.reshape((5, 2))
print('  Input            Target')
for x, t in zip(X, T):
    print(x, '\t', t)


# <h2>50 Units Hidden Layer 1 and 3 Units Hidden Layer 2</h2>

# In[19]:


n_hiddens_1 = 50
n_hiddens_2 = 3
n_iterations = 1000


n_inputs = X.shape[1]
n_outputs = T.shape[1]

n_U = (n_inputs + 1) * n_hiddens_1
n_V = (n_hiddens_1 + 1) * n_hiddens_2
n_W = (n_hiddens_2 + 1) * n_outputs

initial_w = np.random.uniform(-0.1, 0.1, n_U + n_V + n_W)  # range of weights is -0.1 to 0.1

result_sgd = opt.sgd(initial_w,
                     mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                     n_iterations=n_iterations, learning_rate=1e-1, momentum_rate=0.2, 
                     save_wtrace=True)
print(f'SGD final error is {result_sgd["ftrace"][-1]:.3f} and it took {result_sgd["time"]:.2f} seconds')

result_adam = opt.adam(initial_w, 
                       mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                       n_iterations=n_iterations, learning_rate=1e-2, 
                       save_wtrace=True)
print(f'Adam final error is {result_adam["ftrace"][-1]:.3f} and it took {result_adam["time"]:.2f} seconds')

result_scg = opt.scg(initial_w,
                     mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                     n_iterations=n_iterations,
                     save_wtrace=True)
print(f'SCG final error is {result_scg["ftrace"][-1]:.3f} and it took {result_scg["time"]:.2f} seconds')


# <h2>Results</h2>

# In[20]:


w = result_scg['w']

Y = network(w, n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X)
Y.shape


# <h2>Plot</h2>

# In[21]:


def plot_diagonal(T, Y):
    a = min(T.min(), Y.min())
    b = max(T.max(), Y.max())
    plt.plot([a, b], [a, b], '-', lw=3, alpha=0.5)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(result_scg['ftrace'])
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('SCG')

plt.subplot(1, 3, 2)
plt.plot(T[:, 0], Y[:, 0], '.')
plot_diagonal(T[:, 0], Y[:, 0])
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('$Y_1$')

plt.subplot(1, 3, 3)
plt.plot(T[:, 1], Y[:, 1], '.')
plot_diagonal(T[:, 1], Y[: 1])
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('$Y_2$');


# <h3>Discussion</h3>
# SCG had the best MSE of 0.640 and slowest run time of 0.36 seconds. Based on the first graph (Iterations by MSE) it looks like the right amount of iterations are to reached the minimumize the MSE. However after running the training several times with the randomized initial weights the MSE varies quite a lot suggesting that higher iterations might be needed to consistently get a low MSE. The Y1 graph plots a linear line for the first output which plots very well for all five values. The Y2 graph is identical to the Y1 graph except the range of the x, y axes is now 0-200 instead of 0-15. Y2 seems to plot well with a linear line for all five values. 
# 
# When the training results in a higher MSE the Y1 graph plots a linear line for the first output which doesn't plot very well for the last value (15, 13), but well for the first four values. The Y2 graph is identical to the Y1 graph except the range of the x, y axes is now 0-200 instead of 0-15. Y2 seems to plot well with a linear line except for the last value (200, 170).

# <h2>Extra Credit</h2>

# <h3>Train Model</h3>

# In[22]:


hour = [int(t[:2]) for t in data['Time']]
X = np.array(hour).reshape(-1, 1)
CO = data['CO(GT)']
T = np.array(CO).reshape(-1, 1)

n_hiddens_1 = 5
n_hiddens_2 = 5
n_iterations = 3000

n_inputs = X.shape[1]
n_outputs = T.shape[1]

n_U = (n_inputs + 1) * n_hiddens_1
n_V = (n_hiddens_1 + 1) * n_hiddens_2
n_W = (n_hiddens_2 + 1) * n_outputs

initial_w = np.random.uniform(-0.1, 0.1, n_U + n_V + n_W)  # range of weights is -0.1 to 0.1

result_adam = opt.adam(initial_w, 
                       mse, error_gradient, fargs=[n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, X, T],
                       n_iterations=n_iterations, learning_rate=1e-2, 
                       save_wtrace=True)
print(f'Adam final error is {result_adam["ftrace"][-1]:.3f} and it took {result_adam["time"]:.2f} seconds')


# <h3>Plot</h3>

# In[23]:


xs_standardized = np.linspace(0, 23, 100).reshape((-1, 1))
Y, Z1, Z2 = network(result_adam['w'], n_inputs, n_hiddens_1, n_hiddens_2, n_outputs, xs_standardized, True)

plt.figure(figsize=(20, 18))
plt.subplot(3, 1, 1)
[a, b, c, d, e] = plt.plot(xs, Z1)
plt.title("Hidden Layer 1")
plt.legend([a, b, c, d, e], ["Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5"])
plt.xlabel('Hour')
plt.ylabel('Unit Output')

plt.subplot(3, 1, 2)
plt.title("Hidden Layer 2")
[a, b, c, d, e] = plt.plot(xs, Z2)
plt.legend([a, b, c, d, e], ["Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5"])
plt.xlabel('Hour')
plt.ylabel('Unit Output')

plt.subplot(3, 1, 3)
plt.title("Output")
plt.plot(xs, Y)
plt.plot(X, T, 'k.')
plt.xlabel('Hour')
plt.ylabel('Predicted CO');


# <h3>Discussion</h3>
# The first plot shows the outputs of the first layer given the hour. They look like different forms of the tanh function. The second plot shows the weights of the second layer. The second layer has a few outputs that look like the tanh function, but a few that are more complex. This makes sense given the second layers inputs are from the first layers outputs (a tanh funtion) that then go into the second layer (a second tanh function) creating a more complex relationship (than a single tanh function) between the input hours and the outputs of the second layer. The final graph shows the predicted output of the final layer. The predicted ouput is a more complex function than the second hidden layer for the same reason the second hidden layer is more complex than the first hidden layer except their is no tanh function on the last layer.

# In[1]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# <h1>Sam Armstrong Assignment 4 CS545</h1>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import pandas

import torch
import mlutilities as ml  # for ml.draw
import optimizers as opt  # for opt.sgd, opt.adam, and opt.scg

import time
import copy
import pickle, gzip

import neuralnetworks as nn
# import optimizers as opt  # from Lecture Notes


# In[2]:


class NeuralNetwork_Convolutional():
    
    def __init__(self, n_channels_in_image, image_size,
                 n_units_in_conv_layers, kernels_size_and_stride,
                 n_units_in_fc_hidden_layers,
                 classes, use_gpu=False, verbose=True):

        if not isinstance(n_units_in_conv_layers, list):
            raise Exception('n_units_in_conv_layers must be a list')

        if not isinstance(n_units_in_fc_hidden_layers, list):
            raise Exception('n_units_in_fc_hidden_layers must be a list')
        
        if use_gpu and not torch.cuda.is_available():
            print('\nGPU is not available. Running on CPU.\n')
            use_gpu = False

        self.n_channels_in_image = n_channels_in_image
        self.image_size = image_size 
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.n_units_in_fc_hidden_layers = n_units_in_fc_hidden_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = len(classes)
        self.classes = np.array(classes)
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        self.n_conv_layers = len(self.n_units_in_conv_layers)
        self.n_fc_hidden_layers = len(self.n_units_in_fc_hidden_layers)

        # Build the net layers
        self.nnet = torch.nn.Sequential()

        # Add convolutional layers

        n_units_previous = self.n_channels_in_image
        output_size_previous = self.image_size
        n_layers = 0
        if self.n_conv_layers > 0:

            for (n_units, kernel) in zip(self.n_units_in_conv_layers, self.kernels_size_and_stride):
                n_units_previous, output_size_previous = self._add_conv2d_tanh(n_layers,
                                        n_units_previous, output_size_previous, n_units, kernel)
                n_layers += 1 # for text label in layer
                
        # A4.3 version moved following statement left one indent level
        
        self.nnet.add_module('flatten', torch.nn.Flatten())  # prepare for fc layers

        n_inputs = output_size_previous ** 2 * n_units_previous
        if self.n_fc_hidden_layers > 0:
            for n_units in self.n_units_in_fc_hidden_layers:
                n_inputs = self._add_fc_tanh(n_layers, n_inputs, n_units)
                n_layers += 1

        self.nnet.add_module(f'output_{n_layers}', torch.nn.Linear(n_inputs, self.n_outputs))

        # Member variables for standardization
        self.Xmeans = None
        self.Xstds = None

        if self.use_gpu:
            self.nnet.cuda()

        self.n_epochs = 0
        self.error_trace = []

    def _add_conv2d_tanh(self, n_layers, n_units_previous, output_size_previous,
                   n_units, kernel_size_and_stride):
        kernel_size, kernel_stride = kernel_size_and_stride
        self.nnet.add_module(f'conv_{n_layers}', torch.nn.Conv2d(n_units_previous, n_units,
                                                                 kernel_size, kernel_stride))
        self.nnet.add_module(f'output_{n_layers}', torch.nn.Tanh())
        output_size_previous = (output_size_previous - kernel_size) // kernel_stride + 1
        n_units_previous = n_units                
        return n_units_previous, output_size_previous
    
    def _add_fc_tanh(self, n_layers, n_inputs, n_units):
        self.nnet.add_module(f'linear_{n_layers}', torch.nn.Linear(n_inputs, n_units))
        self.nnet.add_module(f'output_{n_layers}', torch.nn.Tanh())
        n_inputs = n_units
        return n_inputs

    def __repr__(self):
        str = f'''{type(self).__name__}(
                            n_channels_in_image={self.n_channels_in_image},
                            image_size={self.image_size},
                            n_units_in_conv_layers={self.n_units_in_conv_layers},
                            kernels_size_and_stride={self.kernels_size_and_stride},
                            n_units_in_fc_hidden_layers={self.n_units_in_fc_hidden_layers},
                            classes={self.classes},
                            use_gpu={self.use_gpu})'''

        str += self.nnet
        if self.n_epochs > 0:
            str += f'\n   Network was trained for {self.n_epochs} epochs that took {self.training_time:.4f} seconds.'
            str += f'\n   Final objective value is {self.error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str
        
    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

    def train(self, X, T, n_epochs, learning_rate=0.01):

        start_time = time.time()
        
        self.learning_rate = learning_rate

        if T.ndim == 1:
            T = T.reshape((-1, 1))

        _, T = np.where(T == self.classes)  # convert to labels from 0

        self._setup_standardize(X, T)
        X = self._standardizeX(X)

        X = torch.tensor(X)
        T = torch.tensor(T.reshape(-1))
        if self.use_gpu:
            X = X.cuda()
            T = T.cuda()

            
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=learning_rate)
        self.loss_F = torch.nn.CrossEntropyLoss()

        for epoch in range(1, n_epochs + 1):

            self.optimizer.zero_grad()

            Y = self.nnet(X)

            error = self.loss_F(Y, T)
            
            if epoch % int(n_epochs/10) == 0 and self.verbose:
                print(f'Epoch {epoch} error {error:.5f}')

            error.backward()

            self.optimizer.step()
            
            self.error_trace.append(error)
        
        self.training_time = time.time() - start_time
        
    def get_error_trace(self):
        return self.error_trace
    
    def _softmax(self, Y):
        mx = Y.max()
        expY = np.exp(Y - mx)
        denom = expY.sum(axis=1).reshape((-1, 1)) + sys.float_info.epsilon
        return expY / denom
    
    def use(self, X):
        self.nnet.eval()  # turn off gradients and other aspects of training
        X = self._standardizeX(X)
        X = torch.tensor(X)
        if self.use_gpu:
            X = X.cuda()

        Y = self.nnet(X)

        if self.use_gpu:
            Y = Y.cpu()
        Y = Y.detach().numpy()
        Yclasses = self.classes[Y.argmax(axis=1)].reshape((-1, 1))

        return Yclasses, self._softmax(Y)


# <h2>Simple Example with Squares and Diamonds</h2>
# 
# Repeating the example from lecture notes.

# In[3]:


def make_images(nEach):
    images = np.zeros((nEach * 2, 1, 20, 20))  # nSamples, nChannels, rows, columns
    radii = 3 + np.random.randint(10 - 5, size=(nEach * 2, 1))
    centers = np.zeros((nEach * 2, 2))
    for i in range(nEach * 2):
        r = radii[i, 0]
        centers[i, :] = r + 1 + np.random.randint(18 - 2 * r, size=(1, 2))
        x = int(centers[i, 0])
        y = int(centers[i, 1])
        if i < nEach:
            # squares
            images[i, 0, x - r:x + r, y + r] = 1.0
            images[i, 0, x - r:x + r, y - r] = 1.0
            images[i, 0, x - r, y - r:y + r] = 1.0
            images[i, 0, x + r, y - r:y + r + 1] = 1.0
        else:
            # diamonds
            images[i, 0, range(x - r, x), range(y, y + r)] = 1.0
            images[i, 0, range(x - r, x), range(y, y - r, -1)] = 1.0
            images[i, 0, range(x, x + r + 1), range(y + r, y - 1, -1)] = 1.0
            images[i, 0, range(x, x + r), range(y - r, y)] = 1.0
            # images += np.random.randn(*images.shape) * 0.5
            T = np.ones((nEach * 2, 1))
            T[nEach:] = 2
    return images.astype(np.float32), T.astype(np.int)

Xtrain, Ttrain = make_images(500)
Xtest, Ttest = make_images(10)

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[4]:


nnet = NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                   image_size=Xtrain.shape[2],
                                   n_units_in_conv_layers=[5], # , 5],
                                   n_units_in_fc_hidden_layers=[2], # 10, 10],
                                   classes=[1, 2],
                                   kernels_size_and_stride=[[5, 2]], # , [4, 1]],
                                   use_gpu=False)

nnet.train(Xtrain, Ttrain, 50, learning_rate=0.01)


# In[5]:


plt.plot(nnet.get_error_trace())
plt.xlabel('Epochs')
plt.ylabel('MSE');


# In[6]:


Yclasses, Y = nnet.use(Xtest)

print(f'{np.sum(Ttest == Yclasses)} out of {Ttest.shape[0]} test samples correctly classified.', end='')
print(f'  Training took {nnet.training_time:.3f} seconds.')


# In[7]:


def show_layer_output(nnet, X_sample, layer):
    outputs = []
    reg = nnet.nnet[layer * 2].register_forward_hook(
        lambda self, i, o: outputs.append(o))
    nnet.use(X_sample)
    reg.remove()
    output = outputs[0]

    n_units = output.shape[1]
    nplots = int(np.sqrt(n_units)) + 1
    for unit in range(n_units):
        plt.subplot(nplots, nplots, unit+1)
        plt.imshow(output[0, unit, :, :].detach(),cmap='binary')
        plt.axis('off')
    return output

def show_layer_weights(nnet, layer):
    W = nnet.nnet[layer*2].weight.detach()
    n_units = W.shape[0]
    nplots = int(np.sqrt(n_units)) + 1
    for unit in range(n_units):
        plt.subplot(nplots, nplots, unit + 1)
        plt.imshow(W[unit, 0, :, :], cmap='binary')
        plt.axis('off')
    return W


# In[8]:


X_sample = Xtest[0:1, :, :, :]
plt.imshow(X_sample[0, 0, :, :], cmap='binary')
plt.axis('off')


# In[9]:


show_layer_output(nnet, X_sample, 0);


# In[10]:


show_layer_weights(nnet, 0);


# <h2>MNIST Digits</h2>
# 
# Investigate the application of your code to the classification of MNIST digits, which you may download from this site. http://deeplearning.net/tutorial/gettingstarted.html

# In[11]:


# !wget http://deeplearning.net/data/mnist/mnist.pkl.gz

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


# In[12]:


Xtrain = train_set[0]
Ttrain = train_set[1]
Xtrain.shape, Ttrain.shape


# In[13]:


a = Xtrain[0, :].reshape(28, 28)
a.shape


# In[14]:


for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(Xtrain[i, :].reshape(28, 28), cmap='binary')
    plt.title(Ttrain[i])
    plt.axis('off');


# ## CNN Classifier

# In[15]:


Xtrain = train_set[0].reshape(50000, 1, 28, 28)
Ttrain = train_set[1].reshape(50000, 1)
Xvalid = valid_set[0].reshape(10000, 1, 28, 28)
Tvalid = valid_set[1].reshape(10000, 1)
Xtest = test_set[0].reshape(10000, 1, 28, 28)
Ttest = test_set[1].reshape(10000, 1)
Xtrain.shape, Ttrain.shape, Xvalid.shape, Tvalid.shape, Xtest.shape, Ttest.shape


# In[16]:


def testModel(conv_layers, hidden_layers, kernels_size_and_stride, epochs, learning_rate=0.1, use_gpu=True):
    nnet1 = NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                   image_size=Xtrain.shape[2],
                                   n_units_in_conv_layers=conv_layers,
                                   n_units_in_fc_hidden_layers=hidden_layers, 
                                   classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   kernels_size_and_stride=kernels_size_and_stride,
                                   use_gpu=use_gpu)
    nnet1.train(Xtrain, Ttrain, epochs, learning_rate=learning_rate)
    YclassesTrain, Ytrain = nnet1.use(Xtrain)
    YclassesValid, Yvalid = nnet1.use(Xvalid)
    YclassesTest, Ytest = nnet1.use(Xtest)
    print(f'{np.sum(Ttrain == YclassesTrain)/Ttrain.shape[0]*100:.2f}% or {np.sum(Ttrain == YclassesTrain)} out of {Ttrain.shape[0]} training samples correctly classified.')
    print(f'{np.sum(Tvalid == YclassesValid)/Tvalid.shape[0]*100:.2f}% or {np.sum(Tvalid == YclassesValid)} out of {Tvalid.shape[0]} validation samples correctly classified.')
    print(f'{np.sum(Ttest == YclassesTest)/Ttest.shape[0]*100:.2f}% or {np.sum(Ttest == YclassesTest)} out of {Ttest.shape[0]} testing samples correctly classified.')


# <h3>Model 1 conv_layers=[10], fc_hidden_layers=[10], kernels_size_and_stride=[[5, 5]], epochs = 100</h3>

# In[21]:


testModel([10], [10], [[5, 5]], 100)


# <h3>Model 2 conv_layers=[10], fc_hidden_layers=[10], kernels_size_and_stride=[[5, 3]], epochs = 100</h3>

# In[19]:


testModel([10], [10], [[5, 3]], 100)


# <h3>Model 3 conv_layers=[10], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3]], epochs = 100</h3>

# In[20]:


testModel([10], [20], [[5, 3]], 100)


# <h3>Model 4 conv_layers=[10, 10], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3], [4, 2]], epochs = 200</h3>

# In[21]:


testModel([10, 10], [20], [[5, 3], [4, 2]], 200)


# <h3>Model 5 conv_layers=[10, 20], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3], [4, 2]], epochs = 200</h3>

# In[22]:


testModel([10, 20], [20], [[5, 3], [4, 2]], 200)


# <h3>Model 6 conv_layers=[10, 20, 30], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3], [4, 2], [3, 1]], epochs = 200</h3>

# In[23]:


testModel([10, 20, 30], [20], [[5, 3], [4, 2], [3, 1]], 200)


# <h3>Model 7 conv_layers=[10, 20, 30, 40], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3], [4, 2], [3, 1], [1, 1]], epochs = 200</h3>

# In[24]:


testModel([10, 20, 30, 40], [20], [[5, 3], [4, 2], [3, 1], [1, 1]], 200)


# <h3>Model 8 conv_layers=[10, 20, 30, 40], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3], [4, 2], [3, 1], [1, 2]], epochs = 300</h3>

# In[25]:


testModel([10, 20, 30, 40], [20], [[5, 3], [4, 2], [3, 1], [1, 2]], 200)


# <h3>Model 9 conv_layers=[10, 20, 30, 40], fc_hidden_layers=[20], kernels_size_and_stride=[[5, 3], [4, 2], [3, 2], [1, 2]], epochs = 200</h3>

# In[26]:


testModel([10, 20, 30, 40], [20], [[5, 3], [4, 2], [3, 2], [1, 2]], 200)


# <h3>Model 10 conv_layers=[10, 20, 30, 40], fc_hidden_layers=[10], kernels_size_and_stride=[[5, 3], [4, 2], [3, 2], [1, 2]], epochs = 200</h3>

# In[27]:


testModel([10, 20, 30, 40], [10], [[5, 3], [4, 2], [3, 2], [1, 2]], 200)


# ## Discussion
# 
# Starting from model 1 I changed at least one of the parameters each time I created a new model. The models mostly increase in success rate from model 1 to model 10. This is what the incremental changes to each model looks like this.
# 
# 1. conv = [10], hidden = [10], kernel stride = [[5, 5]], epochs 100
# 2. kernel stride = [[5, 3]]
# 3. hidden = [20]
# 4. conv = [10, 10] kernel stride = [[5, 3], [4, 2]] epoch 200
# 5. conv = [10, 20]
# 6. conv = [10, 20, 30] kernel stride = [[5, 3], [4, 2], [3, 1]]
# 7. conv = [10, 20, 30, 40] kernel stride = [[5, 3], [4, 2], [3, 1], [1, 1]]
# 8. kernel stride = [[5, 3], [4, 2], [3, 1], [1, 2]]
# 9. kernel stride = [[5, 3], [4, 2], [3, 2], [1, 2]]
# 10. hidden = [10]
# 
# Increasing the number of convolutional layers seemed to imporve the performance (testing, validation, and training success rate). Increasing the stride and epochs also seemed to improve the performance. The number of hidden layers seemed less important and increasing the number of hidden layers to more than one or the units to more than 10 didn't seem to affect the success rate. The training success rate was the easiest to manipulate (compared to validation and testing success rate) which could give a maximum result of 99%. The validation and testing data was more difficult to increase and seemed to respond less to the changes to the CNN structure, each giving a max success rate of 95%. One change that did seem to directly affect the validation and testing success rate was increasing the number of epochs. Overall it seems like a more complex CNN is better on this dataset.

# ## Extra Credit 1

# In[20]:


cnnet = NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                   image_size=Xtrain.shape[2],
                                   n_units_in_conv_layers=[10, 20, 30, 40],
                                   n_units_in_fc_hidden_layers=[10], 
                                   classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   kernels_size_and_stride=[[5, 3], [4, 2], [3, 2], [1, 2]],
                                   use_gpu=False)
cnnet.train(Xtrain, Ttrain, 200, learning_rate=0.01)
YclassesTrain, Ytrain = cnnet.use(Xtrain)
YclassesValid, Yvalid = cnnet.use(Xvalid)
YclassesTest, Ytest = cnnet.use(Xtest)


# In[21]:


show_layer_output(cnnet, Xtrain, 0);


# In[22]:


show_layer_weights(cnnet, 0);


# ### Discussion
# 
# The output of the convolutional looks like its either capturing straight lines (the top of the 5 or the bottom of the 5) or curved lines (the rightside/bottom curve of the 5). The layer weights seemed to be focused on the bottom area (the line on the bottom of the five), the right middle area (the rightside/bottom curve of the 5), or the top area (the line on the top of the five). It looks like the layer weights focus on one or two areas in the picture and if a line appears in that area it is accentuated in the output images of the convalutional layer.

# ## Extra Credit 2

# In[23]:


cnnet_non_gpu = NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                   image_size=Xtrain.shape[2],
                                   n_units_in_conv_layers=[10, 20, 30, 40],
                                   n_units_in_fc_hidden_layers=[10], 
                                   classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   kernels_size_and_stride=[[5, 3], [4, 2], [3, 2], [1, 2]],
                                   use_gpu=False)
cnnet_non_gpu.train(Xtrain, Ttrain, 200, learning_rate=0.01)
cnnet_non_gpu.training_time


# In[24]:


cnnet_gpu = NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                   image_size=Xtrain.shape[2],
                                   n_units_in_conv_layers=[10, 20, 30, 40],
                                   n_units_in_fc_hidden_layers=[10], 
                                   classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   kernels_size_and_stride=[[5, 3], [4, 2], [3, 2], [1, 2]],
                                   use_gpu=True)
cnnet_gpu.train(Xtrain, Ttrain, 200, learning_rate=0.01)
cnnet_gpu.training_time


# ### Discussion
# 
# Without the gpu the time was ~208 seconds and with the gpu it was ~26 seconds. Running a convalutional layers seems really computationally expensive especially when the kernel size and stride are small. This is likely because the CNN has to look at more sub-areas of the picture which requires alot of matrix calculations. So running matrix calculations on a gpu really speeds up the training of a CNN.

# In[25]:


# !rm A4grader.zip
# !rm A4grader.py
# !wget https://www.cs.colostate.edu/~anderson/cs545/notebooks/A4grader.zip
# !unzip A4grader.zip
# %run -i A4grader.py


# In[ ]:




