
import numpy as np
import nltk
from nltk.corpus import brown
from collections import defaultdict
from sklearn.metrics import accuracy_score


class HMM:

    def __init__(self, transition_prob, emission_prob, state_name, output_name, pi):
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.state_name = state_name
        self.output_name = output_name
        self.pi = pi

    def transformation_data(self):
        """Modify matrix of probabilities into nested dictionaries to use them easier"""

        # Create dictionaries from data to use it easier
        states = self.state_name
        obs = self.output_name
        pi = dict(zip(states, self.pi))
        emit_p = {}
        for i in range(len(states)):
            emit_p[states[i]] = dict(zip(obs, self.emission_prob[i, ]))
        trans_p = {}
        for i in range(len(states)):
            trans_p[states[i]] = dict(zip(states, self.transition_prob[i, ]))
        return states, obs, pi, trans_p, emit_p

    def generate(self, length):
        """generate randomly a sequence of states and observations of a certain length
        according to the initial probabilities"""

        # modify data to dict for a better use
        states, obs, pi, trans_p, emit_p = self.transformation_data()

        # create empty list to store estimated states and observations
        generated_states = []
        generated_obs = []

        # initialize the state
        t = 0
        st_t = np.random.choice(states, p=list(pi.values()))
        p = pi[st_t]
        generated_states.append(st_t)

        while t < length:
            # go forward
            t = t + 1
            # generate the observation at t+1
            probs = list(emit_p[st_t].values())
            # pr = np.array(probs)/np.sum(probs)  # normalize probability
            obs_t = np.random.choice(obs, p=probs)
            generated_obs.append(obs_t)
            # calculate the probability of this observation according the previous state
            p = p * emit_p[st_t][obs_t]
            # calculate the new state according the previous observation
            probs = list(trans_p[st_t].values())
            pr_normalize = np.array(probs)/np.sum(probs)  # normalize probability due to calculation precision
            st_t = np.random.choice(states, p=pr_normalize)
            generated_states.append(st_t)

        return generated_states, generated_obs, p

    def viterbi(self, obs):
        """Viterbi Algorithm
        From observations, estimate the states"""

        # modify data
        states, observations, pi, trans_p, emit_p = self.transformation_data()

        # initialize the vector of probabilities of most likely sequence
        # from a initial state and a following observation
        # V list of dict
        V = [{}]
        for st in states:
            V[0][st] = {"prob": pi[st] * emit_p[st][obs[0]], "prev": None}

        # Start the Viterbi calculation
        for t in range(1, len(obs)):
            # V will have index added successively (new dict)
            V.append({})
            for st in states:
                # max transition probability
                max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                # take prev state
                prev_st_selected = states[0]

                # loop to choose the prev state and the max transition probability
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob * emit_p[st][obs[t]]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        # The highest probability of the last vector calculated
        max_prob = max(val["prob"] for val in V[-1].values())
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        return opt, max_prob

    def forward_backward(self, obs):
        '''Forward-Backward algorithm
        return the posterior probability for each state at
        each steps of the generation of observations'''

        # modify data
        states, observations, pi, trans_p, emit_p = self.transformation_data()
        end_st = states[-1]

        # forward part of the algorithm
        fwd = []
        f_prev = {}
        for i, obs_i in enumerate(obs):
            f_curr = {}
            for st in states:
                if i == 0:
                    # initialize the forward
                    prev_f_sum = pi[st]
                else:
                    prev_f_sum = sum(f_prev[s] * trans_p[s][st] for s in states)

                f_curr[st] = emit_p[st][obs_i] * prev_f_sum

            fwd.append(f_curr)
            f_prev = f_curr
        # alpha_fwd = sum(f_prev[k] * trans_p[k][end_st] for k in states)

        # backward part of the algorithm
        bkw = []
        b_prev = {}
        for i, obs_i in enumerate(reversed(obs[1:] + [None, ])):
            b_curr = {}
            for st in states:
                if i == 0:
                    # initialize the backward
                    b_curr[st] = 1 #trans_p[st][end_st]
                else:
                    b_curr[st] = sum(b_prev[s] * emit_p[s][obs_i] * trans_p[st][s] for s in states)

            bkw.insert(0, b_curr)
            b_prev = b_curr

        beta_bkw = sum(pi[l] * emit_p[l][obs[0]] * b_prev[l] for l in states)

        # merging the two parts
        posterior = []
        for i in range(len(obs)):
            posterior.append({st: fwd[i][st] * bkw[i][st] / beta_bkw for st in states})

        # assert alpha_fwd == beta_bkw
        return fwd, bkw, posterior


def test():
    """Is a test function to see what happen"""
    transition_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission_prob = np.array([[0.9, 0.1], [0.6, 0.4]])
    pi = np.array([0.5, 0.5])
    state_name = ['Rainy', 'Sunny']
    output_name = ['Dirty', 'Clean']
    hmm = HMM(transition_prob, emission_prob, state_name, output_name, pi)
    for i in range(10):
        states, observables, p = hmm.generate(10)
        print('States:      ', states)
        print('Observabes:  ', observables)
        print('Probability: ', p)
        print('\n')
        opt, max_prob = hmm.viterbi(observables)
        print('The states: ', opt)
        print('Max proba:  ', max_prob)
        print('\n')
        fwd, bkw, posterior = hmm.forward_backward(observables)
        # print(fwd)
        # print(bkw)
        print(posterior)
        print('\n')

###############################################
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')


def states(tags):
    return list(set(tags))


def observations(words):
    return list(set(words))


def transition_matrix(tags):
    """Calculate Transition Matrix a"""

    # get unique tags vector
    vect_tags = list(set(tags))
    # define size of matrix a
    n = len(vect_tags)
    a = np.zeros((n, n))

    # for each row of a
    for i in range(n):
        # get the cnt of appearance of the tag
        n_tags_t = np.sum(np.array(tags) == vect_tags[i])
        # for each tags
        for t in range(len(tags)-1):
            # if the tag is the tag represented by the index of the row
            if tags[t] == vect_tags[i]:
                # take the index of the following tag
                index_t_1 = vect_tags.index(tags[t+1])
                # add 1/n_tags_t
                a[i, index_t_1] += 1/n_tags_t
    return a


def emission_matrix(tags, words, words_tags):
    """Calculate emission matrix b"""

    # get unique words
    vect_words = list(set(words))
    # get unique tags
    vect_tags = list(set(tags))
    # get unique tuple of (tag, word)
    vect_words_tags = list(set(words_tags))
    # define size matrix b
    n = len(vect_tags)
    m = len(vect_words_tags)
    b = np.zeros((n, m))

    # same process than the transition matrix
    # but we not taking the following tag
    # we add 1/n_tags_t if the tag of the word is the tag
    # represented by the row of the matrix
    for i in range(n):
        n_tags_t = np.sum(np.array(tags) == vect_tags[i])
        for t in range(len(tags)):
            if vect_tags[i] == tags[t]:
                index_w_t = vect_words.index(words[t])
                b[i, index_w_t] += 1 / n_tags_t
    return b


def initial_probabilities(tags, words_tags=None):
    """calculate initial probability"""

    vect_tags = list(set(tags))

    # if we it s the train set => equi probability
    if words_tags is None:
        # equi probability for each states
        return np.array([1/len(vect_tags) for _ in range(len(vect_tags))])
    # if we set specific words to start
    else:
        pi = np.zeros(len(vect_tags))
        for i in range(len(pi)):
            for v in words_tags:
                if vect_tags[i] == v[1]:
                    pi[i] += 1/len(words_tags)
    return pi


def scores(generate_states, viterbi_state):
    return accuracy_score(generate_states[:-1], viterbi_state)


def hmm_train_model(words_tags):
    """generate the state_name, obs_name, transition matrix A
    emission matrix B and initial probabilities from the
    list of words in the the text and their tags
    and return too the list of tags and words"""
    # get list of tags
    tags = [t[1] for t in words_tags]
    words = [t[0] for t in words_tags]
    # get dictionary of count of tags
    tags_cnt = defaultdict(int)
    for t in tags:
        tags_cnt[t] += 1
    state_name = states(tags)
    obs_name = observations(words)
    a = transition_matrix(tags)
    b = emission_matrix(tags, words, words_tags)
    pi = initial_probabilities(tags)

    return tags, words, state_name, obs_name, a, b, pi


def main():

    print('--------TEST---------')
    test()
    print('\n')

    print('--------Brown Corpus Part--------')
    # get brown corpus
    words = brown.words()
    # get list tuples [(tag0, word0), . . . , (tagn, wordn)]
    words_tags = nltk.pos_tag(words[:50000])

    # get list of tags, words from the corpus, states (unique tags), observations (unique words)
    # get the transition and emission matrices and the initial probabilities
    tags, words, state_name, obs_name, a, b, pi = hmm_train_model(words_tags)
    print('Print the matrix A to show that the sum of probabilities may be 1-err from computer precision')
    print(np.sum(a, axis=1))
    print('\n')

    # generate the HMM object according to its definition space (A, B, states, observations, pi)
    hmm_train = HMM(a, b, state_name, obs_name, pi)

    # From brown corpus: generate 150 words with an equi probabilities for each states
    generate_state, generate_obs, p = hmm_train.generate(100)
    print('generated states:       ', generate_state)
    print('generated observations: ', generate_obs)

    # apply Vertibi to estimate states from the previous generated observations
    viterbi_state, max_prob = hmm_train.viterbi(generate_obs)
    print('viterbi states:         ', viterbi_state)

    # calculate the accuracy
    # the accuracy score is depending on the number of states we generate (here 150)
    accuracy = scores(generate_state, viterbi_state)
    print('accuracy : ', accuracy)

    # Print the posterior probability for each state at each steps of the genration
    fwd, bkw, posterior = hmm_train.forward_backward(generate_obs)
    print('posterior: ', posterior)
    print('\n')

    print('------3 first words of lyrics "Here we are" of Queen-----')
    t = ['here', 'we', 'are']
    words_tags_lyrics = nltk.pos_tag(t)
    print('Initial words: ', t)
    print('Tag and Words: ', words_tags_lyrics)
    # calculated initial probability according to the sarting words
    pi = initial_probabilities(tags, words_tags=words_tags_lyrics)
    # generate HMM class with A and B of the train set
    hmm = HMM(a, b, state_name, obs_name, pi)
    for i in range(10):
        generate_state, generate_obs, p = hmm.generate(15)
        print('lyrics sentence number: ', i)
        print('lyrics generated states:       ', generate_state)
        print('lyrics generated observations: ', generate_obs)


if __name__ == '__main__':
    main()