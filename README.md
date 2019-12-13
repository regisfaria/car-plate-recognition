# Car plate recognition

## OVERVIEW
This code was developed for a final test on my AI 2 class.
The objective is to my Neural Network MLP recognize the characters on a car's plate.

The steps taken in my code are:
1. From a given car image, extract it's plate
2. Do some pre processing on it's plate and then segment each char
3. Make some pos processing after segmentation and send it to the NN
 
The output should be the car's plate. 

*NOTE: There are a few bugs and unsolved problems.
1. The NN mistakes a lot
2. The characters are not recognized by order. So if a plate is: "ABC1234" the output may be "12AB4C3"

My NN MLP trained in 120 epochs achieved the following accuracy:
And it's respective loss:
![Alt Text](https://raw.githubusercontent.com/regisfaria/car-plate-recognition/master/model_accuracy.png)
![Alt Text](https://raw.githubusercontent.com/regisfaria/car-plate-recognition/master/model_loss.png)

## REQUIREMENTS

To install the code depencencies, just run the following bash script

```
$ ./install
```

*NOTE: I've made it creates a virtualenv, because there are some unnecessary libs being used in the req.txt
So you can delete it afterwards

## CONTACT
Any questions or ideas are welcome
Email: regisprogramming@gmail.com

[LinkedIn](https://www.linkedin.com/in/regissfaria/) and [GitHub](https://github.com/regisfaria) profiles
