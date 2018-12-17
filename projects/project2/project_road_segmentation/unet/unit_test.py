
from utrain import main as train
from utest import main as test

if __name__=="__main__":
    hdf5_file = train(256, 2, 1, 1, False, aug=True)
    print('hdf5_file', hdf5_file)
    hdf5_file = train(400, 2, 1, 1, True, aug=False)
    print('hdf5_file', hdf5_file)
    csv_file = test(hdf5_file, t=False, four_split=False)
    print('csv_file', csv_file)
    csv_file = test(hdf5_file, t=True, four_split=False)
    print('csv_file', csv_file)
    csv_file = test(hdf5_file, t=False, four_split=True)
    print('csv_file', csv_file)
