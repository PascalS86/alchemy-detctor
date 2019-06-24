# Alchemy-Detector

This is going to be my #AlchemyFriends card-detector.
It uses a tensorflow model (http://tensorfow.org), which was created with azure custom vision (https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/export-model-python)

The current sources show, how you can detect Alchemy With Friends Cards.

I followed the installation for tensorflow with pip. I also use tensorflow 1.5 currently.

To make it run the same way, i did, do the following (tested with windows 10):
* Ensure you installed python 3.6.x (I used 3.6.8)
* install virtualenv
```
python3 -m pip install virtualenv
```
* create a virtualenv
```
virtualenv --system-site-packages -p python3 ./venv
```
* activate the virtualenv
```
.\venv\Scripts\activate

```
* install tensorflow
```
pip install --upgrade tensorflow
```
* install opencv
```
pip install --upgrade opencv
```

* make it run
```
python card-detector.py
```


that's basically it. if you cloned this repos, and followed those instructions, you should have your own object detection first shot.

