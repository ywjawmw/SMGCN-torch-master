# SMGCN pytorch version
## Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.2.0
* numpy == 1.17.4
* scipy == 1.4.1
* temsorboardX == 2.0

## Usage
The hyperparameter search range and optimal settings have been clearly stated in the codes (see the details in utils/parser.py).
* Train

```
python smgcn_main.py 
```


## Dataset
We provide two processed datasets: Herb and Netease.
Herb:
* `train.txt`
  * Train file.
  * Each line is 'symptom set\t herb set\n'.
* `test.txt`
  * Test file.
  * Each line is 'symptom set\t herb set\n'.
Netease:
* `user_bundle_train.txt`
  * Train file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.

* `user_item.txt`
  * Train file.
  * Each line is 'userID\t itemID\n'.
  * Every observed interaction means user u once interacted item i. 

* `bundle_item.txt`
  * Train file.
  * Each line is 'bundleID\t itemID\n'.
  * Every entry means bundle b contains item i.

* `Netease_data_size.txt`
  * Assist file.
  * The only line is 'userNum\t bundleNum\t itemNum\n'.
  * The triplet denotes the number of users, bundles and items, respectively.

* `user_bundle_tune.txt`
  * Tune file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.

* `user_bundle_test.txt`
  * Test file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.
  
