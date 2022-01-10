from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dimensions
    N = X.shape[0]
    C = W.shape[1]

    # compute scores and substract maximum class score for normalization
    s = X @ W              # (N, C)
    s = s - (np.max(s, axis=1).reshape(N, 1))
    s = np.math.e ** s

    for i in range(N):
        partial_loss = 0.0
        summation = 0.0
        for j in range(C):
            summation += s[i, j]
        partial_loss = - np.log(s[i, y[i]] / summation)
        loss += partial_loss

        for j in range(C):
            partial_gradient = (X[i] * s[i, j]) / summation         # (D, )
            if j == y[i]:
                partial_gradient -= X[i]
            dW[:, j] += partial_gradient

    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dimensions
    N = X.shape[0]
    C = W.shape[1]

    # compute scores and substract maximum class score for normalization
    s = X @ W              # (N, C)
    s = s - (np.max(s, axis=1).reshape(N, 1))
    s = np.math.e ** s

    # compute loss
    summation = np.sum(s, axis=1)       # (N, )
    right_class_scores = s[range(N), y]
    loss = (np.sum(np.log(summation) - np.log(right_class_scores))) / \
        N + reg * np.sum(W * W)

    # compute gradient
    s = (s.T / summation).T
    s[range(N), y] -= 1
    dW = (X.T @ s) / N + 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
