from copy import deepcopy

class Wordle:
    def __init__(self, word, rows=6, letters=5):
        self.g_count = 0
        self.word = word
        self.w_hash_table = {}
        if word is not None:
            for x, l in enumerate(word):
                if l in self.w_hash_table:
                    self.w_hash_table[l]['count'] += 1
                    self.w_hash_table[l]['pos'].append(x)
                else:
                    self.w_hash_table[l] = {'count':1, 'pos':[x]}
        self.rows = rows
        self.letters = letters
        self.board = [['' for _ in range(letters)] for _ in range(rows)]
        self.colours = [['B' for _ in range(letters)] for _ in range(rows)]
        self.alph = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def is_end(self):
        if self.board[-1] != ['' for _ in range(self.letters)]:
            return True
        else:
            r = self.game_result()
            if r[0] == True:
                return True
            else:
                return False

    def game_result(self):
        win = (False, 100)
        for i, r in enumerate(self.board):
            if self.word == ''.join(r):
                win = (True, i)
                break
        return win

    def update_board(self, u_inp):
        w_hash_table = deepcopy(self.w_hash_table)
        i_hash_table = {}
        for x, l in enumerate(str(u_inp).upper()):
            self.board[self.g_count][x] = l
            if l in i_hash_table:
                i_hash_table[l].append(x)
            else:
                i_hash_table[l] = [x]
        colours = {'G':[],'B':[],'Y':[]}
        for l in i_hash_table:
            if l in w_hash_table:
                g_hold = []
                for p in i_hash_table[l]:
                    if p in w_hash_table[l]['pos']:
                        g_hold.append(p)
                for p in g_hold:
                    i_hash_table[l].remove(p)
                colours['G'] += g_hold
                if len(g_hold) < w_hash_table[l]['count']:
                    y_hold = []
                    for p in i_hash_table[l]:
                        y_hold.append(p)
                        if len(y_hold) == w_hash_table[l]['count']:
                            break
                    for p in y_hold:
                        i_hash_table[l].remove(p)
                    colours['Y'] += y_hold
                for p in i_hash_table[l]:
                    colours['B'].append(p)
            else:
                colours['B'] += i_hash_table[l]
                i_hash_table[l] = []
        for c in colours:
            for p in colours[c]:
                self.colours[self.g_count][p] = c
        self.g_count += 1

    def valid_guess(self, u_inp):
        if len(u_inp) == 5 and False not in [False for s in str(u_inp).upper() if s not in self.alph]:
            return True
        else:
            return False