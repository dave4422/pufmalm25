#gen datasets script
import numpy as np
import pandas as pd
import random
import os
import csv
from pathlib import Path
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--test-boards', default=1, type=int)

#parser.add_argument('--boards-per-task', default=25, type=int)



args = parser.parse_args()

random.seed( 1 )
number_of_test_boards = args.test_boards
number_of_untouched_test_boards = 3* number_of_test_boards



# directory of frequency data
directory = os.fsencode('data/roPUF/')
k_shot_board_names = random.sample((os.listdir(directory)),k=number_of_test_boards+number_of_untouched_test_boards)
print(k_shot_board_names)


#make experiemnt dir if not exists
if not os.path.exists("data/roPUF/experiments"):
    Path("data/roPUF/experiments").mkdir(parents=True, exist_ok=True)

test_boards = k_shot_board_names[:number_of_test_boards]
untouched_test_boards = k_shot_board_names[number_of_test_boards:]

diraname = os.path.dirname(__file__)

number_challenges_per_ro = 100

#del everythin in experiments direct
shutil.rmtree(directory.decode("utf-8")+'experiments')


#dataset = tf.data.Dataset()

for challenge_size in [64,128]:
    for index in range(1,len(test_boards)+1):

        dont_touch = untouched_test_boards[(index-1)*(number_of_untouched_test_boards//number_of_test_boards):(index-1)*(number_of_untouched_test_boards//number_of_test_boards)+(number_of_untouched_test_boards//number_of_test_boards)]
        #print(dont_touch)

        dont_touch = [item.decode('UTF-8') for item in dont_touch]
        print('Dont touch test set:')
        print(dont_touch)
        print("progress: "+str(challenge_size)+" "+str(index+((challenge_size//64-1)*number_of_test_boards))+"/"+str(len([64,128])*len(test_boards)))
        dir_name = "experiments/"+(test_boards[index-1].decode("utf-8")).strip(".csv")+"_"+str(challenge_size)

        if os.path.exists(directory.decode("utf-8")+dir_name):
            shutil.rmtree(directory.decode("utf-8")+dir_name)



        os.makedirs(os.path.join(diraname, directory.decode("utf-8")+dir_name) )

        training = None
        test = None
        test_untouched = None
        # iterate over all boards
        for file in os.listdir(directory):
            filename = os.fsdecode(os.path.join(directory, file))

            if filename.endswith(".csv"):
                board_data = pd.read_csv(filename, header=None)

                # generate number_challenges_per_ro challenge vectors
                challenges = np.random.choice([-1,1], (challenge_size,number_challenges_per_ro), p=[0.5, 0.5])

                #challenges = np.random.choice([-1,1], (challenge_size), p=[0.5, 0.5]) #same challenge for each meassurement
                #challenges =  np.tile(challenges, (100, 1))

                #print(board_data)
                # substract frequencies of two consecutive ROs
                frequencies = np.subtract(board_data[0::2],board_data[1::2])[0:challenge_size]
                #print(frequencies)
                #print(challenges)
                #print(challenges.shape)
                #product = np.multiply(challenges.T,frequencies)
                product = np.multiply(challenges,frequencies)
                #print(product)
                sum_challenges = np.sum(product,axis=0)

                lables = np.multiply((np.sign(sum_challenges) + 1),0.5)

                #convert to int
                lables = lables.astype(np.int64)


                #print(lables)
                #print(challenges)

                lables_and_challenges = np.column_stack([np.repeat([os.fsdecode(file).strip('.csv')],number_challenges_per_ro),challenges.T, lables])

                #print(lables_and_challenges.shape)
                #print(lables_and_challenges)
                #print()
                if filename.endswith(test_boards[index-1].decode("utf-8")):
                    #print(classes)
                    test = lables_and_challenges
                    #print("LABLES:")
                    #print(lables)
                    #print("\n\n")
                    #print(challenges)
                elif filename[len('data/roPUF/'):] in dont_touch:
                    #print(filename)
                    if test_untouched is None:
                        test_untouched = lables_and_challenges
                    else:

                        test_untouched = np.row_stack([test_untouched, lables_and_challenges])
                else:
                    if training is None:
                        training = lables_and_challenges
                    else:

                        training = np.row_stack([training, lables_and_challenges])


            else:
                continue

        pd.DataFrame(training).to_csv(os.path.join(diraname, directory.decode("utf-8")+dir_name)+"/training.csv",header=False, index = False)
        pd.DataFrame(test).to_csv(os.path.join(diraname, directory.decode("utf-8")+dir_name)+"/test.csv",header=False, index = False)
        pd.DataFrame(test_untouched).to_csv(os.path.join(diraname, directory.decode("utf-8")+dir_name)+"/test_untouched.csv",header=False, index = False)
