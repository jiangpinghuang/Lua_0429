# Generate data files and execute model training and evaluation.

python script/data.py --src_train data/src_train.txt --tar_train data/tar_train.txt --src_valid data/src_valid.txt --tar_valid data/tar_valid.txt --output data/demo
th train.lua -train_file data/demo_train.hdf5 -valid_file data/demo_valid.hdf5 -save_file data/demo_model
th test.lua -model data/demo_model_final.t7 -valid_file data/src_valid.txt -output data/pred.txt -src_dict data/demo.src.dict -tar_dict data/demo.tar.dict
