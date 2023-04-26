import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # Iterate over batch
        for batch in range(y_probs.shape[2]):
            # Iterate over sequence length - len(y_probs[0])
            for t in range(len(y_probs[0])):
                # Update path probability, by multiplying with the current max probability
                path_prob *= np.max(y_probs[:, t, batch])
                # Select most probable symbol and append to decoded_path (symbol set doesn't contain blank)
                most_probable_symbol = np.argmax(y_probs[:, t, batch])
                decoded_path.append(blank if most_probable_symbol == 0 else self.symbol_set[most_probable_symbol - 1])

        compressed_decoded_path = compress_sequence(decoded_path)
        return compressed_decoded_path, path_prob

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]

        # first time instant: initialize paths with each of the symbols including blank, using score at time t=1
        new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score = initialize_paths(
            self.symbol_set, y_probs[:, 0]
            )

        # subsequent time steps
        for t in range(1, T):
            # prune the collection down to the beam width
            paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score = prune(
                new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score, self.beam_width
                )

            # first extend paths by a blank
            new_paths_with_terminal_blank, new_blank_path_score = extend_with_blank(
                paths_with_terminal_blank, paths_with_terminal_symbol, y_probs[:, t], path_score, blank_path_score
                )

            # next extend paths by a symbol
            new_paths_with_terminal_symbol, new_path_score = extend_with_symbol(
                paths_with_terminal_blank, paths_with_terminal_symbol, self.symbol_set, y_probs[:, t], path_score, blank_path_score
                )

        # merge identical paths differing only by the final blank
        merged_paths, final_path_score = merge_identical_paths(
            new_paths_with_terminal_blank, new_blank_path_score, new_paths_with_terminal_symbol, new_path_score
            )
        
        # pick best path
        best_path = max(final_path_score, key=final_path_score.get)
        
        return best_path, final_path_score

def compress_sequence(list):
        """
        compressing the sequence to remove the blanks
        and repetitions in-between blanks and return a string
        """
        blank = 0
        compressed_sequence = []
        previous = None
        compressed_string = ''
        for symbol in list:
            compressed_string += symbol if (symbol != previous and symbol != blank) else ''
            previous = symbol
        return compressed_string

def initialize_paths(symbol_set, y):
    initial_blank_path_score = {}
    initial_path_score = {}

    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = ''
    initial_blank_path_score[path] = y[0] # Score of blank at t=1
    initial_paths_with_final_blank = [path]

    # Push rest of the symbols into a path-ending-with-symbol stack
    initial_paths_with_final_symbol = []
    for c in range(len(symbol_set)): # This is the entire symbol set, without the blank
        path = symbol_set[c]
        initial_path_score[path] = y[c+1] # Score of symbol c at t=1
        initial_paths_with_final_symbol.append(path) # Set addition

    return initial_paths_with_final_blank, initial_paths_with_final_symbol, initial_blank_path_score, initial_path_score

def prune(paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score, beam_width):
    pruned_blank_path_score = {}
    pruned_path_score = {}
    score_list = []

    # First gather all the relevant scores
    for p in paths_with_terminal_blank:
        score_list.append(blank_path_score[p])

    for p in paths_with_terminal_symbol:
        score_list.append(path_score[p])

    # Sort and find cutoff score that retains exactly BeamWidth paths
    score_list.sort(reverse=True) # In decreasing order
    cutoff = score_list[beam_width - 1] if beam_width < len(score_list) else score_list[-1]

    pruned_paths_with_terminal_blank = []
    for p in paths_with_terminal_blank:
        if blank_path_score[p] >= cutoff:
            pruned_paths_with_terminal_blank.append(p) # Set addition
            pruned_blank_path_score[p] = blank_path_score[p]

    pruned_paths_with_terminal_symbol = []
    for p in paths_with_terminal_symbol:
        if path_score[p] >= cutoff:
            pruned_paths_with_terminal_symbol.append(p) # Set addition
            pruned_path_score[p] = path_score[p]

    return pruned_paths_with_terminal_blank, pruned_paths_with_terminal_symbol, pruned_blank_path_score, pruned_path_score

def extend_with_blank(paths_with_terminal_blank, paths_with_terminal_symbol, y, path_score, blank_path_score):
    updated_paths_with_terminal_blank = []
    updated_blank_path_score = {}

    # First work on paths with terminal blanks
    # (This represents transitions along horizontal trellis edges for blanks)
    for path in paths_with_terminal_blank:
        # Repeating a blank doesn’t change the symbol sequence
        updated_paths_with_terminal_blank.append(path) # Set addition
        updated_blank_path_score[path] = blank_path_score[path] * y[0]

    # Then extend paths with terminal symbols by blanks
    for path in paths_with_terminal_symbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
        # simply add the score. If not create a new entry
        if path in updated_paths_with_terminal_blank:
            updated_blank_path_score[path] += path_score[path] * y[0]
        else:
            updated_paths_with_terminal_blank.append(path)
            updated_blank_path_score[path] = path_score[path] * y[0]

    return updated_paths_with_terminal_blank, updated_blank_path_score

def extend_with_symbol(paths_with_terminal_blank, paths_with_terminal_symbol, symbol_set, y, path_score, blank_path_score):
    updated_paths_with_terminal_symbol = []
    updated_path_score = {}

    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in paths_with_terminal_blank:
        for c in range(len(symbol_set)): # SymbolSet does not include blanks
            new_path = path + symbol_set[c] # Concatenation
            updated_paths_with_terminal_symbol.append(new_path) # Set addition
            updated_path_score[new_path] = blank_path_score[path] * y[c + 1]

    # Next work on paths with terminal symbols
    for path in paths_with_terminal_symbol:
        # Extend the path with every symbol other than blank
        for c in range(len(symbol_set)): # SymbolSet does not include blanks
            new_path = path if symbol_set[c] == path[-1] else path + symbol_set[c] # Horizontal transitions don’t extend the sequence
            if new_path in updated_paths_with_terminal_symbol: # Already in list, merge paths
                updated_path_score[new_path] += path_score[path] * y[c + 1]
            else: # Create new path
                updated_paths_with_terminal_symbol.append(new_path) # Set addition
                updated_path_score[new_path] = path_score[path] * y[c + 1]

    return updated_paths_with_terminal_symbol, updated_path_score

def merge_identical_paths(paths_with_terminal_blank, blank_path_score, paths_with_terminal_symbol, path_score):
    # All paths with terminal symbols will remain
    merged_paths = paths_with_terminal_symbol
    final_path_score = path_score

    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in paths_with_terminal_blank:
        if p in merged_paths:
            final_path_score[p] += blank_path_score[p]
        else:
            merged_paths.add(p) # Set addition
            final_path_score[p] = blank_path_score[p]

    return merged_paths, final_path_score