import os


####################
# ##### GeNN ##### #
####################z
difficulties = ['easy', 'med', 'hard']
puzzles = [1, 2, 3]
bugFix = [0, 1]
enhanced = [0, 1]

for d in difficulties:
    for p in puzzles:
        for b in bugFix:
            for e in enhanced:
                os.system(f'python3 1-sudokuGENN.py -d {d} -p {p} -t 300 -b {b} -e {e} -g 1')
                print(f'Difficulty: {d} | Puzzle: {p} | Bug Fix: {b} | Enhanced {e}')


#########################
# ##### SpiNNaker ##### #
#########################
difficulties = ['easy', 'med', 'hard']
puzzles = [1, 2, 3]
bugFix = [0, 1]
enhanced = [0, 1]
trials = 1

for d in difficulties:
    for p in puzzles:
        for b in bugFix:
            for e in enhanced:
                name = f'SpiNNaker{d}{p}_b{b}_e{e}'
                try:
                    file = open(f'../results/{name}.csv', 'r')
                    start = len(file.readlines())
                    file.close()
                except:
                    start = 0
                if start < trials:
                    os.system(f'python3 2-sudokuSpiNNaker.py -d {d} -p {p} -t {trials} -b {b} -e {e}')
                    print(f'Difficulty: {d} | Puzzle: {p} | Bug Fix: {b} | Enhanced {e}')
