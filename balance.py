import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

listDicts = []

for i in range(1,2):
            file_name = 'E:/Aqib khan/Final Year Project/Training/training_data-{0}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)   
            zero_lefts = []
            zero_rights = []

            shuffle(train_data)

            for data in train_data:
                 img = data[0]
                 choice = data[1]

                 if choice == [0,0,1,0,0,0,0,0,0] or choice == [0,0,0,0,1,0,0,0,0] or choice == [0,0,0,0,0,0,1,0,0]:
                     zero_lefts.append([img,choice])
                 elif choice == [0,0,0,1,0,0,0,0,0] or choice == [0,0,0,0,0,1,0,0,0] or choice == [0,0,0,0,0,0,0,1,0]:
                     zero_rights.append([img,choice])
                     
            zero_lefts = zero_lefts * (167 // len(zero_lefts))
            zero_rights = zero_rights * (167 // len(zero_rights))

for i in range(1,195):
            file_name = 'E:/Aqib khan/Final Year Project/Training/training_data-{0}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)   
            lefts = []
            rights = []
            forwards = []
            nokeys = []
            brakes = []
            shuffle(train_data)

            for data in train_data:
                 img = data[0]
                 choice = data[1]

                 if choice == [0,0,1,0,0,0,0,0,0] or choice == [0,0,0,0,1,0,0,0,0] or choice == [0,0,0,0,0,0,1,0,0]:
                     lefts.append([img,choice])
                 elif choice == [1,0,0,0,0,0,0,0,0]:
                     forwards.append([img,choice])
                 elif choice == [0,0,0,1,0,0,0,0,0] or choice == [0,0,0,0,0,1,0,0,0] or choice == [0,0,0,0,0,0,0,1,0]:
                     rights.append([img,choice])
                 elif choice == [0,0,0,0,0,0,0,0,1]:
                     nokeys.append([img,choice])
                 elif choice == [0,1,0,0,0,0,0,0,0]:
                     brakes.append([img,choice])
                 else:
                     None
            forwards = forwards[:167]
            if len(lefts) == 0:
                 lefts = zero_lefts
            else:
                 lefts = lefts * (167 // len(lefts))
            if len(rights) == 0:
                 rights = zero_rights
            else:
                 rights = rights * (167 // len(rights))

            final_data = forwards + lefts + rights + brakes + nokeys
            shuffle(final_data)

            save_file_name = 'E:/Aqib khan/Final Year Project/Training new/training_data_v2-{0}.npy'.format(i)

            np.save(save_file_name, final_data)
            df = pd.DataFrame(final_data)
            dd = Counter(df[1].apply(str))
            listDicts.append(dd)
            print(i)

counter = Counter() 

for d in listDicts:  
    counter.update(d)

print(counter)




