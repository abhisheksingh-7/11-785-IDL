import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
		Initialize instance variables

		Argument(s)
		-----------

		BLANK (int, optional): blank label index. Default 0.

		"""

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
		"""

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # initialize skip_connect to be all zeros.
        skip_connect = np.zeros((N,), dtype=int)
        '''
        Skips are permitted across a blank, but only if the symbols on
        either side are different because a blank is mandatory between
        repetitions of a symbol but not required between distinct symbols
        '''
        for i in range(N-2):
            if ((extended_symbols[i] != self.BLANK) and (extended_symbols[i] != extended_symbols[i+2])):
                skip_connect[i+2] = 1

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        T = logits.shape[0]
        K = extended_symbols.shape[0]
        alpha = np.zeros(shape=(T, K))

        # Intialize alpha(0,0) and alpha(0,r) for all r > 0
        alpha[0][0] = logits[0][extended_symbols[0]]
        for r in range(1, K):
            alpha[0, r] = 0

        # Intialize alpha(0,1)
        alpha[0][1] = logits[0][extended_symbols[1]]

        # Compute all values for alpha(t, l) where 1 <= t < T and 1 <= l < S
        # IMP: Remember to check for skipConnect when calculating alpha
        for t in range(1, T):
            alpha[t][0] = alpha[t-1][0] * logits[t][extended_symbols[0]]
            for r in range(1, K):
                alpha[t][r] = alpha[t-1][r] + alpha[t-1][r-1]
                if (r > 1 and skip_connect[r] == 1):
                    alpha[t][r] += alpha[t-1][r-2]
                alpha[t][r] *= logits[t][extended_symbols[r]]
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
		
		"""

        T = logits.shape[0]
        K = extended_symbols.shape[0]

        beta = np.zeros((T, K))
        beta_hat = np.zeros((T, K))

        beta_hat[T-1][K-1] = logits[T-1][extended_symbols[K-1]]
        beta_hat[T-1][K-2] = logits[T-1][extended_symbols[K-2]]
        for i in reversed(range(K-3)):
            beta_hat[T-1][i] = 0

        for t in reversed(range(T-1)):
            beta_hat[t][K-1] = beta_hat[t+1][K-1] * logits[t][extended_symbols[K-1]]
            for i in reversed(range(K-1)):
                beta_hat[t][i] = beta_hat[t+1][i] + beta_hat[t+1][i+1]

                if ((i < K - 2) and (skip_connect[i+2] == 1)):
                    beta_hat[t][i] += beta_hat[t+1][i+2]

                beta_hat[t][i] *= logits[t][extended_symbols[i]]

        # compute beta from betahat
        for t in reversed(range(T)):
            for i in reversed(range(K)):
                beta[t][i] = beta_hat[t][i] / logits[t][extended_symbols[i]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

		"""

        T, K = alpha.shape
        gamma = np.zeros((T, K))
        sumgamma = np.zeros((T,))

        for t in range(T):
            sumgamma[t] = 0
            for i in range(K):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                sumgamma[t] += gamma[t][i]
            for i in range(K):
                gamma[t][i] = gamma[t][i] / sumgamma[t]

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
        
		"""
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # Computing CTC Loss for single batch

            # Truncate the target to target length
            target_batch = target[batch_itr, :target_lengths[batch_itr]]

            # Truncate the logits to input length
            logits_batch = logits[:input_lengths[batch_itr], batch_itr]

            # Extend target sequence with blank
            self.extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_batch)

            # Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logits_batch, self.extended_symbols, skip_connect)

            # Compute backward probabilities
            beta = self.ctc.get_backward_probs(logits_batch, self.extended_symbols, skip_connect)

            # Compute posteriors using total probability function
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            # Compute expected divergence for each batch and store it in total_loss
            for i, log_probability in enumerate(logits_batch):
                for j, symbol in enumerate(self.extended_symbols):
                    total_loss[batch_itr] -= np.log(log_probability[symbol]) * gamma[i][j]

        # Take an average over all batches and return final result
        total_loss = np.sum(total_loss) / B
        return total_loss

    def backward(self):
        """
		
		CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
		w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # Computing CTC Derivative for single batch

            # Truncate the target to target length
            target_batch = self.target[batch_itr, :self.target_lengths[batch_itr]]

            # Truncate the logits to input length
            logits_batch = self.logits[:self.input_lengths[batch_itr], batch_itr]

            # Extend target sequence with blank
            self.extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_batch)

            # Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logits_batch, self.extended_symbols, skip_connect)

            # Compute backward probabilities
            beta = self.ctc.get_backward_probs(logits_batch, self.extended_symbols, skip_connect)

            # Compute posteriors using total probability function
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            # Compute derivative of divergence and store them in dY
            for i in range(self.input_lengths[batch_itr]):
                for j, symbol in enumerate(self.extended_symbols):
                    dY[i, batch_itr, symbol] -= gamma[i][j] / logits_batch[i][symbol]
        return dY