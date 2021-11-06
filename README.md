
# introduction  

Apple Game([fruit_box_a](https://www.gamesaien.com/game/fruit_box_a/)) macro

python3.9 and selenium and cv2 are required for this project

test: windows10 64bit

![game](game.gif)


# required program

* Chrome and [ChromeDriver](https://chromedriver.chromium.org/downloads)(same version)
    * Show ChromeVersion URL: [chrome://settings/help](chrome://settings/help)
*  python3 and pip


# install
1. Download ChromeDriver which is same version as Chrome and save it to the project folder
 2. run pip command

 ```shell
pip install selenium
pip install numpy
pip install opencv-python
```

  # run
  
  1.  Run Chrome debug mode
```
chrome.exe --remote-debugging-port=9222 --user-data-dir="C:/Users/%USERNAME%/Desktop/ChromeDebug"
```
  2. goto url https://www.gamesaien.com/game/fruit_box_a/
  3. game start button
  4. and run script
  

  # Goal
Eliminate all apples or get the highest score
  
 ## todo
1. back tracking
1. Graph modeling
1. Optimization and structuring
